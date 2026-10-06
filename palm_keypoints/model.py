"""Small convolutional keypoint regressor, forward/backprop/Adam in NumPy only."""
import json
from pathlib import Path
import numpy as np
import cv2
from . import SCHEMA,LINES,TOTAL_POINTS,LINE_POINTS,COORDS

VERSION='numpy-palm-pose-cnn-v2'
SIZE=64
OUTPUT=TOTAL_POINTS*2+len(LINES)+1


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))


def prepare_image(rgb,size=SIZE):
    h,w=rgb.shape[:2];scale=min(size/w,size/h);nw=max(2,round(w*scale));nh=max(2,round(h*scale));x=(size-nw)//2;y=(size-nh)//2
    canvas=np.zeros((size,size,3),np.float32)
    canvas[y:y+nh,x:x+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)/255.0
    return (canvas-.5)*2,{'nw':nw,'nh':nh,'x':x,'y':y,'size':size}


def to_model(points,meta):
    p=np.asarray(points,float).copy();p=p*np.array([meta['nw']-1,meta['nh']-1])+[meta['x'],meta['y']]
    return p/(meta['size']-1)


def from_model(points,meta):
    return (np.asarray(points)*(meta['size']-1)-[meta['x'],meta['y']])/[meta['nw']-1,meta['nh']-1]


def targets(annotation,meta):
    coords=np.zeros((TOTAL_POINTS,2),np.float32);mask=np.zeros_like(coords);presence=[]
    for i,name in enumerate(LINES):
        line=annotation['lines'][name];visible=line['status']=='present';presence.append(float(visible))
        if visible:coords[i*6:(i+1)*6]=to_model(line['points'],meta);mask[i*6:(i+1)*6]=1
    if len(annotation['palm_width_points'])==2:
        coords[LINE_POINTS:]=to_model(annotation['palm_width_points'],meta);mask[LINE_POINTS:]=1
    apparent_right=annotation['handedness']=='right'
    target=np.r_[coords.ravel(),presence,float(apparent_right)].astype(np.float32)
    known_side=annotation['handedness']!='unknown'
    return target,np.r_[mask.ravel(),[float(annotation['lines'][name]['status']!='unreviewed') for name in LINES],float(known_side)].astype(np.float32)


def conv(x,w,b):
    B,H,W,C=x.shape;p=np.pad(x,((0,0),(1,1),(1,1),(0,0)));oh=(H+1)//2;ow=(W+1)//2
    columns=np.concatenate([p[:,dy:dy+oh*2:2,dx:dx+ow*2:2,:] for dy in range(3) for dx in range(3)],axis=-1)
    return columns@w+b,(columns,x.shape)


def conv_back(dy,w,cache,need_input=True):
    cols,shape=cache;dw=cols.reshape(-1,cols.shape[-1]).T@dy.reshape(-1,dy.shape[-1]);db=dy.sum(axis=(0,1,2))
    if not need_input:return None,dw,db
    B,H,W,C=shape;oh,ow=dy.shape[1:3];dc=dy@w.T;dp=np.zeros((B,H+2,W+2,C),np.float32)
    for y in range(3):
        for x in range(3):dp[:,y:y+oh*2:2,x:x+ow*2:2,:]+=dc[...,((y*3+x)*C):((y*3+x+1)*C)]
    return dp[:,1:-1,1:-1],dw,db


class PoseCNN:
    def __init__(self,seed=42):
        self.rng=np.random.default_rng(seed);self.metadata={};self.step=0;self.m={};self.v={}
        self.p={}
        for key,shape in [('w1',(27,8)),('w2',(72,16)),('w3',(1024,96)),('w4',(96,OUTPUT))]:
            self.p[key]=(self.rng.standard_normal(shape)*np.sqrt(2/shape[0])).astype(np.float32)
            self.p['b'+key[1:]]=np.zeros(shape[1],np.float32)
        self.p['w4']*=.05
        # Start near broad plausible locations rather than the same collapsed point.
        base=[]
        for y in (.36,.50,.63):base.extend([[x,y] for x in np.linspace(.25,.75,6)])
        base.extend([[.2,.6],[.8,.6]])
        q=np.asarray(base).ravel();self.p['b4'][:COORDS]=np.log(q/(1-q))

    def features(self,x):
        z1,c1=conv(x,self.p['w1'],self.p['b1']);a1=np.maximum(z1,0)
        pool=a1.reshape(len(x),16,2,16,2,8).mean(axis=(2,4))
        z2,c2=conv(pool,self.p['w2'],self.p['b2']);a2=np.maximum(z2,0);flat=a2.reshape(len(x),-1)
        z3=flat@self.p['w3']+self.p['b3'];hidden=np.maximum(z3,0)
        return hidden,(z1,c1,z2,c2,flat,z3)

    def loss_grad(self,x,target,mask,dropout=.15):
        hidden,cache=self.features(x)
        drop=(self.rng.random(hidden.shape)>=dropout).astype(np.float32)/(1-dropout) if dropout else np.ones_like(hidden)
        hd=hidden*drop;logits=hd@self.p['w4']+self.p['b4'];out=sigmoid(logits)
        d=out[:,:COORDS]-target[:,:COORDS];valid=mask[:,:COORDS];den=max(1,float(valid.sum()));delta=.05
        huber=np.where(np.abs(d)<delta,.5*d*d/delta,np.abs(d)-.5*delta)
        coordinate_loss=float((huber*valid).sum()/den)
        class_mask=mask[:,COORDS:];class_den=max(1,float(class_mask.sum()))
        classification_loss=float(((np.logaddexp(0,logits[:,COORDS:])-target[:,COORDS:]*logits[:,COORDS:])*class_mask).sum()/class_den)
        grad=np.zeros_like(out);grad[:,:COORDS]=np.clip(d/delta,-1,1)*valid/den*out[:,:COORDS]*(1-out[:,:COORDS])
        grad[:,COORDS:]=.15*(out[:,COORDS:]-target[:,COORDS:])*class_mask/class_den
        grads={'w4':hd.T@grad,'b4':grad.sum(axis=0)};dh=(grad@self.p['w4'].T)*drop
        z1,c1,z2,c2,flat,z3=cache;dz3=dh*(z3>0);grads['w3']=flat.T@dz3;grads['b3']=dz3.sum(axis=0)
        dz2=(dz3@self.p['w3'].T).reshape(z2.shape)*(z2>0)
        dp,grads['w2'],grads['b2']=conv_back(dz2,self.p['w2'],c2)
        dz1=np.repeat(np.repeat(dp,2,axis=1),2,axis=2)/4*(z1>0)
        _,grads['w1'],grads['b1']=conv_back(dz1,self.p['w1'],c1,False)
        return coordinate_loss+.15*classification_loss,grads

    def update(self,grads,lr=.001):
        if not all(np.isfinite(g).all() for g in grads.values()):raise ValueError('Gradient không hữu hạn; checkpoint tốt trước đó được giữ.')
        norm=np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()));scale=min(1,5/max(norm,1e-8));self.step+=1
        for k,g in grads.items():
            g=g*scale
            self.m[k]=.9*self.m.get(k,np.zeros_like(g))+.1*g
            self.v[k]=.999*self.v.get(k,np.zeros_like(g))+.001*g*g
            self.p[k]-=lr*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)

    def predict_tensor(self,x):
        hidden,_=self.features(x);return sigmoid(hidden@self.p['w4']+self.p['b4'])

    def predict(self,rgb,mc_samples=8):
        x,meta=prepare_image(rgb);hidden,_=self.features(x[None]);rng=np.random.default_rng(173)
        draws=[]
        for _ in range(mc_samples):
            drop=(rng.random(hidden.shape)>=.15)/.85
            draws.append(sigmoid((hidden*drop)@self.p['w4']+self.p['b4'])[0])
        draws=np.asarray(draws);mean=draws.mean(axis=0);coords=from_model(mean[:COORDS].reshape(-1,2),meta)
        std=draws[:,:COORDS].std(axis=0).reshape(-1,2)*(SIZE-1)/[meta['nw']-1,meta['nh']-1]
        return {'points':coords,'std':std,'presence':mean[COORDS:COORDS+len(LINES)],'right_hand_score':float(mean[-1])}

    def save(self,path,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        doc={**metadata,'schema':SCHEMA,'model_version':VERSION,'input_size':SIZE,'lines':list(LINES),'keypoints':6}
        tmp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(tmp,metadata=np.asarray(json.dumps(doc,allow_nan=False)),**self.p);tmp.replace(path)

    @classmethod
    def load(cls,path,expected_sha256):
        from prototype_common import sha256,validate_npz_size
        path=Path(path)
        if not expected_sha256 or len(expected_sha256)!=64 or sha256(path)!=expected_sha256:raise ValueError('Checkpoint cần SHA-256 tin cậy khớp nội dung.')
        validate_npz_size(path,32*1024*1024)
        model=cls()
        with np.load(path,allow_pickle=False) as z:
            metadata=json.loads(str(z['metadata'].item()))
            if not isinstance(metadata,dict) or metadata.get('schema')!=SCHEMA or metadata.get('model_version')!=VERSION or metadata.get('lines')!=list(LINES) or metadata.get('input_size')!=SIZE or metadata.get('keypoints')!=6:raise ValueError('Checkpoint không phải Tâm đạo/Trí đạo/Sinh đạo keypoint CNN v2.')
            for k,template in model.p.items():
                value=np.asarray(z[k],np.float32)
                if value.shape!=template.shape or not np.isfinite(value).all():raise ValueError('Checkpoint shape/giá trị không hợp lệ.')
                model.p[k]=value.copy()
        model.metadata=metadata;return model
