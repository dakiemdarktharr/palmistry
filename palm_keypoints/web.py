"""Local keypoint annotation and training UI; no commands entered by the user."""
import os
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from PIL import Image,ImageOps
import numpy as np
from flask import request,jsonify,render_template,send_file
from . import LINES
from .data import load_project,read_json,write_json,annotation_path,validate_annotation,audited_rows,safe_path
from .model import PoseCNN
from .spline import trace_line,width_from_points
from .workflow import infer


def register_keypoint_ui(app,project_dir,csrf,nonce,legacy_busy=lambda:False):
    root=Path(project_dir);base=root/'artifacts/keypoints';base.mkdir(parents=True,exist_ok=True)
    lock=threading.RLock();statefile=base/'ui_state.json'
    state={'status':'idle','log':[]}
    if statefile.exists():
        try:state.update(read_json(statefile))
        except (ValueError,OSError):pass
    if state['status']=='running':state.update(status='interrupted',error='Ứng dụng đã khởi động lại. Dữ liệu/checkpoint đã lưu vẫn giữ nguyên; xem log rồi chạy lại bằng thư mục mới.')

    def name(value):
        value=str(value)
        if not re.fullmatch('[A-Za-z0-9_-]{1,80}',value):raise ValueError('Tên project không hợp lệ.')
        return value

    def project(value):
        path=safe_path(base,name(value));doc=load_project(path);return path,doc

    def item_for(doc,image_id):
        item=next((r for r in doc['images'] if r['image_id']==image_id),None)
        if item is None:raise ValueError('Ảnh không thuộc project.')
        return item

    def persist():write_json(statefile,state)

    def model_for(project_name):
        path,_=project(project_name)
        reports=sorted((path/'runs').glob('train_*/summary.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for report in reports:
            result=read_json(report)
            if result.get('ok') and result.get('checkpoint'):
                checkpoint=Path(result['checkpoint']).resolve()
                if checkpoint.is_relative_to(path.resolve()) and checkpoint.is_file():return {'project':project_name,'checkpoint':str(checkpoint),'sha256':result['sha256']}
        raise ValueError('Chưa có checkpoint keypoint được train cho project này.')


    def begin(kind,args,out,project_name):
        with lock:
            if state['status']=='running' or legacy_busy():raise ValueError('Một pipeline đang chạy; đợi hoàn thành trước khi bắt đầu tác vụ mới.')
            job_id=uuid.uuid4().hex
            state.update(status='running',job_id=job_id,kind=kind,project=project_name,log=[],error='',output=str(out),result={},returncode=None);persist()
        def worker():
            try:
                flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
                with (base/'job.log').open('w',encoding='utf-8') as logfile:
                    process=subprocess.Popen([sys.executable,'-u',str(root/'keypoint_pipeline.py'),*args],cwd=root,env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONUTF8":"1"},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=flags)
                    with lock:state['pid']=process.pid;persist()
                    for line in process.stdout:
                        logfile.write(line);logfile.flush()
                        with lock:state['log']=(state['log']+[line.rstrip()])[-50:]
                    code=process.wait()
                with lock:
                    state['status']='completed' if code==0 else 'failed';state['returncode']=code
                    if code==0:
                        result=read_json(Path(out)/'summary.json');state['result']=result
                        if kind=='train':state['model']={'project':project_name,'checkpoint':result['checkpoint'],'sha256':result['sha256']}
                    else:state['error']='Tác vụ thất bại; xem log. Ảnh gốc và các bản đã lưu được giữ nguyên.'
                    persist()
            except Exception as exc:
                with lock:state.update(status='failed',error=str(exc));persist()
        threading.Thread(target=worker,daemon=True).start()
        return job_id

    def api_error(fn):
        from functools import wraps
        @wraps(fn)
        def call(*args,**kwargs):
            try:return fn(*args,**kwargs)
            except (ValueError,KeyError,TypeError,OSError) as exc:return jsonify(ok=False,error=str(exc)),400
        return call

    def body():
        if request.content_length and request.content_length>128*1024:raise ValueError('JSON annotation vượt 128 KiB.')
        value=request.get_json(silent=True)
        if not isinstance(value,dict):raise ValueError('Cần JSON object hợp lệ.')
        return value

    @app.get('/keypoints')
    def kp_page():return render_template('keypoints.html',csrf=csrf,nonce=nonce,lines=LINES)

    @app.get('/api/keypoints/state')
    @api_error
    def kp_state():
        projects=[];project_errors=[]
        for path in sorted(base.glob('*/project.json')):
            try:
                doc=load_project(path.parent)
                if doc.get('test_fixture'):continue
                counts={'pending':0,'approved':0,'rejected':0}
                for row in doc['images']:
                    a=read_json(annotation_path(path.parent,row['image_id']));counts[a['status']]+=1
                projects.append({'name':path.parent.name,'count':len(doc['images']),'counts':counts})
            except (ValueError,OSError,KeyError,TypeError,AttributeError) as exc:
                project_errors.append({'name':path.parent.name,'error':str(exc)})
        runs=[p.parent.parent.name for p in (root/'artifacts').glob('*/review/review.csv')]
        with lock:snapshot=json.loads(json.dumps(state))
        return jsonify(ok=True,projects=projects,project_errors=project_errors,runs=sorted(runs),job=snapshot)

    @app.post('/api/keypoints/create')
    @api_error
    def kp_create():
        b=body();run=safe_path(root/'artifacts',name(b['run']));n=name(b['name']);out=safe_path(base,n)
        if out.exists():raise ValueError('Project đã tồn tại; chọn tên khác.')
        count=int(b.get('count',400))
        if not 1<=count<=400:raise ValueError('Queue thủ công từ 1 đến 400 ảnh.')
        if not (run/'review/review.csv').is_file():raise ValueError('Queue nguồn không tồn tại.')
        job_id=begin('create',['from-review','--run',str(run),'--out',str(out),'--count',str(count)],out,n)
        return jsonify(ok=True,job_id=job_id),202

    @app.get('/api/keypoints/project/<project_name>')
    @api_error
    def kp_project(project_name):
        _,doc=project(project_name)
        return jsonify(ok=True,images=[{k:r[k] for k in ('image_id','width','height')} for r in doc['images']],tutorial_only=doc.get('tutorial_only',False),tutorial_note=doc.get('tutorial_note',''))

    @app.get('/keypoints/assets/<project_name>/<image_id>')
    @api_error
    def kp_asset(project_name,image_id):
        path,doc=project(project_name);item=item_for(doc,image_id)
        return send_file(safe_path(path,item['image_path']),mimetype='image/jpeg')

    @app.get('/api/keypoints/annotation/<project_name>/<image_id>')
    @api_error
    def kp_annotation(project_name,image_id):
        path,doc=project(project_name);item_for(doc,image_id)
        return jsonify(ok=True,annotation=read_json(annotation_path(path,image_id)))

    @app.post('/api/keypoints/annotation/<project_name>/<image_id>')
    @api_error
    def kp_save(project_name,image_id):
        b=body();path,doc=project(project_name);item=item_for(doc,image_id)
        with lock:
            if state['status']=='running' and state.get('project')==project_name:raise ValueError('Đợi tác vụ hiện tại xong trước khi đổi nhãn đã dùng cho train/export.')
            previous=read_json(annotation_path(path,image_id))
            if b.get('revision')!=previous['revision']:return jsonify(ok=False,error='Nhãn đã được sửa ở tab khác. Tải lại ảnh trước khi lưu.'),409
            if b.get('image_id')!=image_id:raise ValueError('Image ID không khớp.')
            a=validate_annotation(b,(item['width'],item['height']));a['revision']=previous['revision']+1;a['provenance']='tutorial_visual_example' if doc.get('tutorial_only') else 'human_ui'
            write_json(annotation_path(path,image_id),a)
        return jsonify(ok=True,annotation=a)

    @app.post('/api/keypoints/preview/<project_name>/<image_id>')
    @api_error
    def kp_preview(project_name,image_id):
        b=body();_,doc=project(project_name);item=item_for(doc,image_id);size=(item['width'],item['height'])
        b['status']='pending';a=validate_annotation(b,size)
        try:width=width_from_points(a['palm_width_points'],size)
        except ValueError:return jsonify(ok=True,lines={},message='Đặt 2 điểm bề rộng lòng bàn tay để xem B-spline và độ dài chuẩn hóa.')
        result={k:trace_line(v['points'],size,width,min_confidence=0) for k,v in a['lines'].items() if v['status']!='absent' and v['points']}
        return jsonify(ok=True,lines=result,palm_width_px=width)

    @app.post('/api/keypoints/action')
    @api_error
    def kp_action():
        b=body();n=name(b['project']);path,_=project(n);kind=b['kind']
        if kind not in ('train','pseudo','export-yolo'):raise ValueError('Tác vụ không hợp lệ.')
        args=[kind,'--project',str(path)];out=path/'runs'/(kind+'_'+uuid.uuid4().hex[:10]);args+=['--out',str(out)]
        if kind=='train':
            audited_rows(path);epochs=int(b.get('epochs',60));batch=int(b.get('batch',8))
            if not 1<=epochs<=500 or not 1<=batch<=32:raise ValueError('Epoch 1..500; batch 1..32.')
            args+=['--epochs',str(epochs),'--batch-size',str(batch)]
        elif kind=='export-yolo':audited_rows(path)
        else:
            m=model_for(n)
            limit=int(b.get('limit',0))
            if not 0<=limit<=1000000:raise ValueError('Số ảnh pseudo từ 0 đến 1 triệu; 0 là tất cả.')
            args+=['--checkpoint',m['checkpoint'],'--sha256',m['sha256'],'--limit',str(limit)]
        job_id=begin(kind,args,out,n);return jsonify(ok=True,job_id=job_id),202

    @app.post('/api/keypoints/predict')
    @api_error
    def kp_predict():
        if request.content_length is None or request.content_length>21*1024*1024:raise ValueError('Ảnh upload tối đa 20 MiB.')
        n=name(request.form.get('project',''));m=model_for(n)
        upload=request.files.get('image')
        if upload is None:raise ValueError('Chưa chọn ảnh.')
        data=upload.stream.read(20*1024*1024+1)
        if len(data)>20*1024*1024:raise ValueError('Ảnh vượt 20 MiB.')
        with Image.open(io.BytesIO(data)) as im:
            if im.width*im.height>16000000 or max(im.size)>8192:raise ValueError('Ảnh vượt 16 megapixels / 8192 pixels mỗi chiều.')
            rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
        result=infer(PoseCNN.load(m['checkpoint'],m['sha256']),rgb,request.form.get('mirrored','unknown'))
        return jsonify(ok=True,prediction=result)

    @app.get('/api/keypoints/report/<project_name>')
    @api_error
    def kp_report(project_name):
        path,_=project(project_name);runs=[]
        for report in sorted((path/'runs').glob('*/summary.json')):
            runs.append({'name':report.parent.name,'summary':read_json(report)})
        return jsonify(ok=True,runs=runs)


    @app.post('/api/keypoints/review-pseudo')
    @api_error
    def kp_review_pseudo():
        b=body();n=name(b['project']);path,_=project(n)
        run=name(b['run']);report=safe_path(path/'runs',run)/'manual_review.jsonl'
        output_name=name(b.get('name',n[:50]+'_review_'+uuid.uuid4().hex[:8]));out=safe_path(base,output_name)
        if out.exists() or not report.is_file():raise ValueError('Chọn báo cáo pseudo có thật và project review mới.')
        offset=int(b.get('offset',0))
        if not 0<=offset<=1000000:raise ValueError('Offset review từ 0 đến 1 triệu.')
        job_id=begin('review-pseudo',['review-pseudo','--report',str(report),'--out',str(out),'--offset',str(offset)],out,output_name)
        return jsonify(ok=True,job_id=job_id),202

    @app.post('/api/keypoints/metadata/<project_name>')
    @api_error
    def kp_metadata(project_name):
        path,doc=project(project_name)
        if not request.content_length or request.content_length>9*1024*1024:raise ValueError('CSV metadata tối đa 8 MiB.')
        upload=request.files.get('file')
        if upload is None:raise ValueError('Chưa chọn CSV metadata.')
        raw=upload.stream.read(8*1024*1024+1)
        if len(raw)>8*1024*1024:raise ValueError('CSV metadata vượt 8 MiB.')
        rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        if not rows or len(rows)>1000000:raise ValueError('CSV rỗng hoặc quá nhiều dòng.')
        with lock:
            if state['status']=='running':raise ValueError('Đợi tác vụ hiện tại hoàn thành trước khi sửa metadata.')
            remaining=read_json(path/'remaining_sources.json');source_root=Path(remaining.get('source_root',path)).resolve()
            allowed={str(Path(v).resolve()) for v in remaining.get('paths',[])}|{str(Path(v['source_path']).resolve()) for v in doc['images']}
            updates={}
            for row in rows:
                source=str((source_root/row['source_path']).resolve());subject=str(row.get('subject_id','')).strip()
                mirror=row.get('mirrored','unknown').strip() or 'unknown';hand=row.get('handedness','unknown').strip() or 'unknown'
                if source not in allowed or not subject or len(subject)>200 or mirror not in ('unknown','yes','no') or hand not in ('unknown','left','right'):raise ValueError('Metadata cần source_path thuộc dataset, subject_id thật, mirrored unknown/yes/no; handedness unknown/left/right.')
                if source in updates:raise ValueError('CSV lặp source_path; hợp nhất metadata trước khi nhập.')
                updates[source]={'subject_id':subject,'source_group':row.get('source_group','').strip()[:200],'mirrored':mirror,'handedness':hand}
            remaining.setdefault('metadata',{}).update(updates);write_json(path/'remaining_sources.json',remaining)
            changed=0
            for item in doc['images']:
                meta=updates.get(str(Path(item['source_path']).resolve()))
                if not meta:continue
                a=read_json(annotation_path(path,item['image_id']));a['subject_id']=meta['subject_id'];a['source_group']=meta['source_group']
                for field in ('mirrored','handedness'):
                    if meta[field]!='unknown' and meta[field]!=a[field]:
                        a[field]=meta[field]
                        if a['status']=='approved':a['status']='pending'
                a['revision']+=1;write_json(annotation_path(path,item['image_id']),a);changed+=1
        return jsonify(ok=True,metadata_rows=len(updates),queue_updated=changed)

    @app.get('/api/keypoints/annotations-export/<project_name>')
    @api_error
    def kp_annotations_export(project_name):
        path,doc=project(project_name)
        data={'schema':doc['schema'],'source_report':doc.get('source_report'),'review_only':doc.get('review_only',False),'images':[{**item,'annotation':read_json(annotation_path(path,item['image_id']))} for item in doc['images']]}
        return app.response_class(json.dumps(data,ensure_ascii=False,allow_nan=False),mimetype='application/json',headers={'Content-Disposition':'attachment; filename="annotations.json"'})
