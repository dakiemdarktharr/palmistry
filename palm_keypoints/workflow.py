"""Training, physical measurements and conservative pseudo-label review routing."""
from pathlib import Path
import json
import time
import numpy as np
from prototype_common import read_rgb,sha256,perceptual_hash
from . import SCHEMA,LINES,LINE_POINTS,TOTAL_POINTS
from .data import audited_rows,load_project,read_json,write_json,safe_path,annotation_path
from .model import PoseCNN,prepare_image,targets,from_model
from .spline import trace_line,width_from_points


def evaluate(model,rows,project):
    stats={name:{'present':0,'absent':0,'unlabeled':0,'tp':0,'fp':0,'fn':0,'errors':[]} for name in LINES}
    widths=[];hands=[]
    for row in rows:
        rgb=read_rgb(safe_path(project,row['image_path']));h,w=rgb.shape[:2];a=row['annotation']
        pred=model.predict(rgb);width=None
        if len(a['palm_width_points'])==2:
            width=width_from_points(a['palm_width_points'],(w,h))
            widths.append(float(np.linalg.norm((pred['points'][LINE_POINTS:]-a['palm_width_points'])*[w-1,h-1],axis=1).mean()/width))
        if a['handedness']!='unknown':hands.append((pred['right_hand_score']>=.5)==(a['handedness']=='right'))
        for i,name in enumerate(LINES):
            s=stats[name];line=a['lines'][name]
            if line['status']=='unreviewed':s['unlabeled']+=1;continue
            positive=line['status']=='present';detected=pred['presence'][i]>=.5
            s['present' if positive else 'absent']+=1;s['tp']+=int(positive and detected);s['fp']+=int(not positive and detected);s['fn']+=int(positive and not detected)
            if positive and width:s['errors'].extend((np.linalg.norm((pred['points'][i*6:(i+1)*6]-line['points'])*[w-1,h-1],axis=1)/width).tolist())
    for s in stats.values():
        errors=s.pop('errors');s['mean_error_palm_width']=float(np.mean(errors)) if errors else None
        s['pck_at_005']=float(np.mean(np.asarray(errors)<=.05)) if errors else None
        s['precision']=s['tp']/max(1,s['tp']+s['fp']);s['recall']=s['tp']/max(1,s['tp']+s['fn'])
        s['auto_label_supported']=bool(s['present']>=3 and s['absent']>=3 and s['precision']>=.95 and s['recall']>=.9 and s['pck_at_005'] is not None and s['pck_at_005']>=.9)
    return {'samples':len(rows),'lines':stats,'reference_samples':len(widths),'reference_error_palm_width':float(np.mean(widths)) if widths else None,
            'hand_samples':len(hands),'hand_accuracy':float(np.mean(hands)) if hands else None,'confidence_semantics':'MC-dropout ranking, not calibrated probability'}


def train(project,out,epochs=60,batch_size=8,seed=42,lr=.001,patience=12):
    if not 1<=epochs<=500 or not 1<=batch_size<=32 or not 0<lr<=.01:raise ValueError('epochs 1..500, batch 1..32, learning rate (0,.01].')
    project=Path(project).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Dùng thư mục train mới để giữ checkpoint cũ.')
    rows=audited_rows(project,seed);out.mkdir(parents=True)
    snapshot=[]
    for row in rows:
        snapshot.append({k:v for k,v in row.items() if k!='annotation'}|{'annotation':row['annotation'],'annotation_sha256':sha256(annotation_path(project,row['image_id']))})
    write_json(out/'split.json',snapshot)
    arrays={}
    for split in ('train','val','test'):
        group=[r for r in rows if r['split']==split];xx=[];yy=[];mm=[]
        for row in group:
            x,meta=prepare_image(read_rgb(safe_path(project,row['image_path'])));y,m=targets(row['annotation'],meta);xx.append(x);yy.append(y);mm.append(m)
        arrays[split]=tuple(np.asarray(v,np.float32) for v in (xx,yy,mm))
    model=PoseCNN(seed);rng=np.random.default_rng(seed);best=float('inf');stale=0;history=[]
    metadata={'project':str(project),'seed':seed,'split_manifest':'split.json','split_sha256':sha256(out/'split.json'),
              'training_kind':'human_approved_only','training_count':len(arrays['train'][0]),
              'positive_counts':{name:sum(r['split']=='train' and r['annotation']['lines'][name]['status']=='present' for r in rows) for name in LINES}}
    x,y,m=arrays['train'];start=time.monotonic()
    for epoch in range(1,epochs+1):
        losses=[]
        for ids in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/batch_size)))):
            loss,grad=model.loss_grad(x[ids],y[ids],m[ids]);model.update(grad,lr);losses.append(loss)
        vx,vy,vm=arrays['val'];vl=[]
        for offset in range(0,len(vx),batch_size):
            end=offset+batch_size;loss,_=model.loss_grad(vx[offset:end],vy[offset:end],vm[offset:end],dropout=0);vl.append((loss,len(vx[offset:end])))
        val=sum(loss*n for loss,n in vl)/len(vx)
        if not np.isfinite(val):raise ValueError('Loss không hữu hạn; giữ best.npz trước đó để kiểm tra.')
        record={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':float(val)};history.append(record)
        print(json.dumps(record),flush=True);write_json(out/'history.json',history)
        if val<best-1e-6:best=val;stale=0;model.save(out/'best.npz',metadata|{'best_epoch':epoch})
        else:stale+=1
        if stale>=patience:break
    model=PoseCNN.load(out/'best.npz',sha256(out/'best.npz'))
    validation=evaluate(model,[r for r in rows if r['split']=='val'],project)
    test=evaluate(model,[r for r in rows if r['split']=='test'],project)
    metadata=model.metadata|{'validation':validation};model.save(out/'best.npz',metadata)
    result={'ok':True,'schema':SCHEMA,'checkpoint':str(out/'best.npz'),'sha256':sha256(out/'best.npz'),'epochs':len(history),
            'validation':validation,'test':test,'seconds':time.monotonic()-start}
    write_json(out/'summary.json',result);return result


def infer(model,rgb,mirrored='unknown'):
    if mirrored not in ('unknown','yes','no'):raise ValueError('mirrored phải là unknown/yes/no.')
    h,w=rgb.shape[:2];pred=model.predict(rgb);points=np.asarray(pred['points']);std=np.asarray(pred['std']);reasons=[]
    if points.shape!=(TOTAL_POINTS,2) or std.shape!=(TOTAL_POINTS,2) or not np.isfinite(points).all() or not np.isfinite(std).all():raise ValueError('Model trả keypoint sai shape hoặc không hữu hạn.')
    scores=np.r_[pred['presence'],pred['right_hand_score']]
    if scores.shape!=(len(LINES)+1,) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)) or np.any(std<0):raise ValueError('Confidence model không hợp lệ.')
    validation=model.metadata.get('validation',{});line_validation=validation.get('lines',{})
    try:width=width_from_points(points[LINE_POINTS:],(w,h))
    except ValueError:width=0;reasons.append('invalid_palm_reference')
    uncertainty=np.linalg.norm(std*[w-1,h-1],axis=1)
    confidence=np.exp(-uncertainty/max(.05*width,1e-6))
    if confidence[LINE_POINTS:].min()<.90:reasons.append('uncertain_palm_reference')
    if validation.get('reference_error_palm_width') is None or validation['reference_error_palm_width']>.05:reasons.append('reference_validation_insufficient')
    right=float(pred['right_hand_score']);side='unknown'
    if max(right,1-right)>=.90 and (validation.get('hand_accuracy') or 0)>=.95 and validation.get('hand_samples',0)>=6:
        side='right' if right>=.5 else 'left'
    lines={}
    for i,name in enumerate(LINES):
        probability=float(pred['presence'][i]);supported=line_validation.get(name,{}).get('auto_label_supported',False)
        if probability<=.05 and supported:
            lines[name]={'status':'absent','presence_score':probability,'points':[]};continue
        q=points[i*6:(i+1)*6];conf=confidence[i*6:(i+1)*6]*probability
        traced=trace_line(q,(w,h),width,confidence=conf)
        issues=list(traced['reasons'])
        if not supported:issues.append('validation_support_insufficient')
        if not model.metadata.get('positive_counts',{}).get(name,0):issues.append('no_positive_training_examples')
        valid=traced['valid'] and not issues and not reasons
        lines[name]={'status':'present' if valid else 'review','points':q.tolist(),'keypoint_confidence':conf.tolist(),
                     'presence_score':probability,'trace':traced,'reasons':issues}
    needs_review=bool(reasons) or any(v['status']=='review' for v in lines.values())
    return {'schema':SCHEMA,'image_size':[w,h],'mirrored':mirrored,'handedness':side,'hand_score':right,
            'palm_width_points':points[LINE_POINTS:].tolist(),'palm_width_px':width,'lines':lines,'needs_review':needs_review,
            'reasons':reasons,'confidence_semantics':'uncalibrated MC-dropout ranking; validation and geometry gates also required'}


def pseudo(project,checkpoint,digest,out,limit=0):
    project=Path(project).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Dùng thư mục pseudo mới.')
    model=PoseCNN.load(checkpoint,digest)
    if model.metadata.get('project')!=str(project):raise ValueError('Checkpoint không thuộc project này.')
    splitfile=Path(checkpoint).parent/'split.json'
    if sha256(splitfile)!=model.metadata.get('split_sha256'):raise ValueError('Split manifest đã đổi; không thể audit chống leakage.')
    split=read_json(splitfile);held=[r for r in split if r['split']!='train'];doc=load_project(project)
    known={r['source_sha256'] for r in doc['images']};known_paths={str(Path(r['source_path']).resolve()) for r in doc['images']};held_phash=[int(r['phash'],16) for r in held]
    held_subjects={r['subject_id'] for r in held if r.get('subject_id')};held_groups={r['source_group'] for r in held}
    remaining=read_json(project/'remaining_sources.json');paths=list(dict.fromkeys(remaining.get('paths',[])))
    if limit<0:raise ValueError('limit phải >=0.')
    if limit:paths=paths[:limit]
    root=Path(remaining.get('source_root',project)).resolve();metadata={}
    for source,meta in remaining.get('metadata',{}).items():
        key=str((root/str(source)).resolve())
        if key in metadata and metadata[key]!=meta:raise ValueError('Metadata có đường dẫn trùng sau chuẩn hóa với thông tin người khác nhau.')
        metadata[key]=meta
    out.mkdir(parents=True)
    counts={'accepted':0,'review':0,'skipped_known_or_duplicate':0};seen=set()
    with (out/'auto_labels.jsonl').open('w',encoding='utf-8') as auto,(out/'manual_review.jsonl').open('w',encoding='utf-8') as manual:
        for index,source in enumerate(paths):
            record={'source_path':str(source),'provenance':'pseudo_model','checkpoint_sha256':digest,'training_partition':'train_only','human_approved':False}
            try:
                path=(root/str(source)).resolve()
                if not path.is_relative_to(root):raise ValueError('source_outside_dataset_root')
                if str(path) in known_paths:counts['skipped_known_or_duplicate']+=1;continue
                if not path.is_file():raise ValueError('source_image_missing')
                meta=metadata.get(str(path),{})
                if not meta.get('subject_id'):raise ValueError('subject_identity_required_before_pseudo_training')
                rawhash=sha256(path)
                if rawhash in known or rawhash in seen:counts['skipped_known_or_duplicate']+=1;continue
                seen.add(rawhash);record['source_sha256']=rawhash
                rgb=read_rgb(path);phash=int(perceptual_hash(rgb),16)
                if meta['subject_id'] in held_subjects or meta.get('source_group',str(path.parent)) in held_groups or any(((phash^h).bit_count()<=6) for h in held_phash):raise ValueError('held_out_subject_source_or_duplicate')
                record['subject_id']=meta['subject_id'];record['prediction']=infer(model,rgb,meta.get('mirrored','unknown'))
                if record['prediction']['needs_review']:record['reason']='confidence_geometry_or_validation_review'
            except (ValueError,OSError,KeyError) as exc:record['reason']=str(exc)
            accepted='reason' not in record;counts['accepted' if accepted else 'review']+=1
            (auto if accepted else manual).write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            if (index+1)%25==0:print(f'Pseudo {index+1}/{len(paths)}: {counts}',flush=True)
    result={'ok':True,**counts,'out':str(out),'total_candidates':len(paths),'note':'Pseudo nhãn riêng, không tự đổi annotation người dùng thành approved. Thiếu metadata/độ tin cậy → manual_review.jsonl.'}
    write_json(out/'summary.json',result);return result
