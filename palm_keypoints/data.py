"""Versioned JSON annotations and subject-aware splits; originals are never modified."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
from PIL import Image
from prototype_common import read_rgb,sha256,perceptual_hash,group_rows,write_json
from . import SCHEMA,LEGACY_SCHEMA,LINES,N_POINTS,TOTAL_POINTS
from .spline import trace_line,width_from_points


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_path(root,relative):
    root=Path(root).resolve();p=(root/str(relative)).resolve()
    if not p.is_relative_to(root):raise ValueError('Đường dẫn nằm ngoài project.')
    return p


def annotation_path(project,image_id):
    if not re.fullmatch('[a-f0-9]{20}',str(image_id)):raise ValueError('Image ID không hợp lệ.')
    return Path(project)/'annotations'/f'{image_id}.json'


def empty_annotation(image_id):
    return {'schema':SCHEMA,'image_id':image_id,'revision':0,'status':'pending','provenance':'human_annotation_required',
            'handedness':'unknown','mirrored':'unknown','subject_id':'','source_group':'','palm_width_points':[],
            'notes':'','lines':{name:{'status':'unreviewed','points':[]} for name in LINES}}


def normalize_annotation(annotation):
    """Read v1 without discarding its fourth line or fabricating missing labels."""
    a=json.loads(json.dumps(annotation,allow_nan=False))
    if not isinstance(a,dict):raise ValueError('Annotation phải là JSON object.')
    if a.get('schema')==LEGACY_SCHEMA:
        if not isinstance(a.get('lines'),dict) or set(a['lines'])!=set(LINES)|{'fate_line'}:raise ValueError('Schema v1 không hợp lệ.')
        a.setdefault('legacy_lines',{})['fate_line']=a['lines'].pop('fate_line')
        a['legacy_schema']=LEGACY_SCHEMA;a['schema']=SCHEMA
    for key,value in empty_annotation(a.get('image_id','')).items():
        if key not in ('schema','lines'):a.setdefault(key,value)
    return a


def validate_annotation(annotation,image_size,approve=False):
    a=normalize_annotation(annotation)
    if a.get('schema')!=SCHEMA or not isinstance(a.get('lines'),dict) or set(a['lines'])!=set(LINES):raise ValueError('Schema phải là Tâm đạo / Trí đạo / Sinh đạo, mỗi đường tối đa 6 điểm.')
    if a.get('status') not in ('pending','approved','rejected'):raise ValueError('Trạng thái annotation không hợp lệ.')
    for name in LINES:
        line=a['lines'][name]
        if not isinstance(line,dict):raise ValueError('Mỗi đường phải là JSON object.')
        state=line.get('status');p=np.asarray(line.get('points',[]),float)
        if state not in ('unreviewed','present','absent'):raise ValueError('Trạng thái đường không hợp lệ.')
        if p.shape!=(0,) and (p.ndim!=2 or p.shape[1]!=2 or len(p)>N_POINTS or not np.isfinite(p).all() or np.any((p<0)|(p>1))):raise ValueError(f'{name}: điểm phải trong ảnh, tối đa 6 điểm.')
        if state=='absent' and p.size:raise ValueError(f'{name}: đường absent không có tọa độ.')
    if a.get('handedness') not in ('unknown','left','right') or a.get('mirrored') not in ('unknown','yes','no'):raise ValueError('Bên tay/lật gương không hợp lệ.')
    ref=np.asarray(a.get('palm_width_points',[]),float)
    if ref.shape!=(0,) and (ref.ndim!=2 or ref.shape[1]!=2 or len(ref)>2 or not np.isfinite(ref).all() or np.any((ref<0)|(ref>1))):raise ValueError('Điểm bề rộng không hợp lệ.')
    if approve or a['status']=='approved':
        # Missing fields remain unknown even after a swipe. Only supplied labels are checked.
        width=width_from_points(a['palm_width_points'],image_size) if len(ref)==2 else max(image_size)
        for name in LINES:
            line=a['lines'][name]
            if line['status']=='present':
                result=trace_line(line['points'],image_size,width,min_confidence=0)
                if not result['valid']:raise ValueError(f'{name}: '+', '.join(result['reasons']))
    a['notes']=str(a.get('notes',''))[:2000]
    for key in ('subject_id','source_group'):a[key]=str(a.get(key,'')).strip()[:200]
    return a


def create_from_review(run,project,count=100):
    if not 1<=count<=400:raise ValueError('Queue thủ công từ 1 đến 400 ảnh.')
    run,project=Path(run).resolve(),Path(project).resolve()
    if project.exists():raise FileExistsError('Project đã tồn tại; chọn tên mới.')
    with (run/'review/review.csv').open(encoding='utf-8-sig',newline='') as f:all_rows=list(csv.DictReader(f))
    if any(not row.get('source_path') for row in all_rows):
        manifest=run/'01_preprocessed_dataset/labels_pseudo_strict.csv'
        if not manifest.is_file():raise ValueError('Queue thiếu source_path và manifest nguồn; cần khôi phục labels_pseudo_strict.csv.')
        with manifest.open(encoding='utf-8-sig',newline='') as f:by_image={r['image_path']:r for r in csv.DictReader(f)}
        all_rows=[{**by_image.get(row.get('image_path'),{}),**row} for row in all_rows]
        if any(not row.get('source_path') for row in all_rows):raise ValueError('Không nối được ảnh review với ảnh gốc; không dùng ảnh crop/mask làm fallback nhãn.')
    rows=all_rows[:count]
    if not rows:raise ValueError('Queue gốc không có ảnh.')
    project.mkdir(parents=True);entries=[];seen=set()
    try:
        for row in rows:
            source=Path(row['source_path']).resolve()
            if not source.is_file():raise FileNotFoundError(f'Ảnh gốc không còn ở {source}; khôi phục đường dẫn trước khi tạo queue.')
            digest=sha256(source)
            if digest in seen:continue
            seen.add(digest);image_id=digest[:20];rgb=read_rgb(source);image=Image.fromarray(rgb);image.thumbnail((1024,1024))
            image_path=f'images/{image_id}.jpg';target=project/image_path;target.parent.mkdir(exist_ok=True)
            image.save(target,quality=95)
            item={'image_id':image_id,'image_path':image_path,'width':image.width,'height':image.height,'source_path':str(source),
                  'source_sha256':digest,'sha256':sha256(target),'phash':perceptual_hash(np.asarray(image)),
                  'source_group':str(source.parent),'subject_id':''}
            entries.append(item);write_json(annotation_path(project,image_id),empty_annotation(image_id))
            if len(entries)%25==0:print(f'Keypoint queue: {len(entries)}/{len(rows)} ảnh',flush=True)
        remaining_file=run/'01_preprocessed_dataset/remaining_sources.json'
        remaining=read_json(remaining_file) if remaining_file.exists() else {'paths':[]}
        remaining['paths']=list(dict.fromkeys(remaining.get('paths',[])+[r['source_path'] for r in all_rows[count:]]))
        write_json(project/'remaining_sources.json',remaining)
        write_json(project/'project.json',{'schema':SCHEMA,'n_keypoints':N_POINTS,'lines':list(LINES),'source_run':str(run),
                                          'status':'ready','images':entries,'remaining_count':len(remaining.get('paths',[]))})
        return {'ok':True,'project':str(project),'images':len(entries),'remaining':len(remaining.get('paths',[]))}
    except Exception as exc:
        write_json(project/'creation_error.json',{'error':str(exc),'fallback':'Ảnh gốc và queue mask vẫn giữ nguyên; sửa lỗi rồi dùng tên project mới.'});raise


def load_project(project):
    data=read_json(Path(project)/'project.json')
    if data.get('schema') not in (SCHEMA,LEGACY_SCHEMA):raise ValueError('Project không phải keypoint schema hiện tại.')
    if data['schema']==LEGACY_SCHEMA:data={**data,'schema':SCHEMA,'legacy_schema':LEGACY_SCHEMA,'lines':list(LINES)}
    return data


def migrate_project(project):
    """Back up every original annotation before persisting the three-line schema."""
    import uuid
    project=Path(project);original=read_json(project/'project.json');doc=load_project(project)
    paths=list((project/'annotations').glob('*.json'))
    changed=[(p,read_json(p)) for p in paths if read_json(p).get('schema')==LEGACY_SCHEMA]
    if original['schema']==SCHEMA and not changed:return {'ok':True,'changed':0}
    backup=project/'backups'/('schema_v1_'+uuid.uuid4().hex)
    write_json(backup/'project.json',original)
    for path,a in changed:write_json(backup/'annotations'/path.name,a)
    for path,a in changed:
        updated=normalize_annotation(a);updated['revision']+=1;write_json(path,updated)
    write_json(project/'project.json',doc)
    return {'ok':True,'changed':len(changed),'backup':str(backup)}


def limit_review_queue(project,count=100):
    """Reduce a local queue with a recovery snapshot; keep every reviewed image."""
    if not 1<=count<=400:raise ValueError('Review count must be 1..400.')
    project=Path(project);doc=load_project(project)
    if len(doc['images'])<=count:return {'ok':True,'images':len(doc['images']),'deferred':0}
    reviewed=[];pending=[]
    for item in doc['images']:
        a=read_json(annotation_path(project,item['image_id']))
        (pending if a['status']=='pending' else reviewed).append(item)
    if len(reviewed)>count:raise ValueError('Target is smaller than the number of reviewed images; no changes made.')
    chosen={r['image_id'] for r in (reviewed+pending)[:count]}
    deferred=[r for r in doc['images'] if r['image_id'] not in chosen]
    remaining=read_json(project/'remaining_sources.json')
    import uuid
    backup=project/'backups'/('queue_'+uuid.uuid4().hex+'.json')
    write_json(backup,{'project':doc,'remaining_sources':remaining})
    remaining['paths']=list(dict.fromkeys(remaining.get('paths',[])+[r['source_path'] for r in deferred]))
    # Write remaining first: interruption can leave duplicate references, never lost sources.
    write_json(project/'remaining_sources.json',remaining)
    doc['images']=[r for r in doc['images'] if r['image_id'] in chosen]
    doc['remaining_count']=len(remaining['paths']);doc['review_target']=count
    write_json(project/'project.json',doc)
    return {'ok':True,'images':len(doc['images']),'deferred':len(deferred),'backup':str(backup)}


def audited_rows(project,seed=42):
    project=Path(project);rows=[]
    if load_project(project).get('tutorial_only'):raise ValueError('Đây là mẫu hướng dẫn, không dùng để train hoặc export tập huấn luyện.')
    if load_project(project).get('review_only'):raise ValueError('Queue pseudo này chỉ để review; không chia lại held-out. Xuất nhãn đã duyệt để hợp nhất có audit vào tập train gốc.')
    for item in load_project(project)['images']:
        a=read_json(annotation_path(project,item['image_id']))
        if a.get('status')!='approved':continue
        a=validate_annotation(a,(item['width'],item['height']),approve=True)
        image_path=safe_path(project,item['image_path']);image=read_rgb(image_path)
        if sha256(image_path)!=item['sha256']:raise ValueError('Ảnh annotation đã đổi; cần kiểm tra lại keypoint và tạo project mới.')
        rows.append({**item,'annotation':a,'subject_id':a['subject_id'],
                     'source_group':a['source_group'] or ('subject:'+a['subject_id'] if a['subject_id'] else item['source_group']),'sha256':sha256(image_path),'phash':perceptual_hash(image)})
    if len(rows)<3:raise ValueError(f'Mới có {len(rows)} ảnh keypoint approved; cần nhãn thật và ít nhất 3 nhóm độc lập để train/val/test.')
    try:return group_rows(rows,seed=seed)
    except ValueError as exc:raise ValueError('Không chia được tập độc lập. Bổ sung metadata người/nhóm nguồn có thật; không chia ngẫu nhiên các ảnh cùng session để ép train. '+str(exc)) from exc


def export_yolo(project,out,seed=42):
    """One palm object: 18 line points + 2 optional reference points."""
    out=Path(out)
    if out.exists():raise FileExistsError('Thư mục export phải mới.')
    rows=audited_rows(project,seed)
    out.mkdir(parents=True)
    for row in rows:
        a=row['annotation'];points=[]
        for name in LINES:
            points.extend([[float(x),float(y),2] for x,y in a['lines'][name]['points']] if a['lines'][name]['status']=='present' else [[0,0,0]]*N_POINTS)
        points.extend([[float(x),float(y),2] for x,y in a['palm_width_points']] if len(a['palm_width_points'])==2 else [[0,0,0]]*2)
        # Full normalized image box is intentional: previews contain one hand, no detector-derived box.
        values=[0,.5,.5,1,1]+[v for point in points for v in point]
        image_dir=out/'images'/row['split'];label_dir=out/'labels'/row['split'];image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(safe_path(project,row['image_path']),image_dir/(row['image_id']+'.jpg'))
        (label_dir/(row['image_id']+'.txt')).write_text(' '.join(map(str,values))+'\n',encoding='utf-8')
    config=f'path: {json.dumps(out.resolve().as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nkpt_shape: [{TOTAL_POINTS}, 3]\nflip_idx: {list(range(TOTAL_POINTS))}\nnames:\n  0: palm\n'
    (out/'data.yaml').write_text(config,encoding='utf-8');write_json(out/'schema.json',{'schema':SCHEMA,'order':[f'{name}_{i}' for name in LINES for i in range(6)]+['palm_width_0','palm_width_1'],'note':'Missing/unreviewed landmarks have visibility 0. Original statuses are preserved in annotations.json. Fate is archived, not mapped to another line.'})
    write_json(out/'split.json',[{k:r[k] for k in ('image_id','subject_id','source_group','split','group_id')} for r in rows])
    write_json(out/'annotations.json',{'schema':SCHEMA,'images':[{**{k:r[k] for k in ('image_id','width','height','split','annotation')},'image_path':f"images/{r['split']}/{r['image_id']}.jpg"} for r in rows]})
    return {'ok':True,'count':len(rows),'out':str(out)}


def create_from_pseudo(report,project,count=100,offset=0):
    """Materialize proposals into an editable, pending queue; never auto-approve."""
    report=Path(report).resolve();project=Path(project).resolve()
    if project.exists():raise FileExistsError('Dùng tên project review mới.')
    if not 1<=count<=400:raise ValueError('Mỗi batch review tối đa 400 ảnh.')
    if offset<0:raise ValueError('Offset phải >=0.')
    if not report.is_file():raise ValueError('Chưa có danh sách pseudo cần kiểm duyệt.')
    project.mkdir(parents=True);entries=[];errors=[];seen=set()
    with report.open(encoding='utf-8-sig') as f:
        for index,line in enumerate(f):
            if index<offset:continue
            if index>=offset+count:break
            try:
                record=json.loads(line);source=Path(record['source_path']).resolve();digest=sha256(source)
                if record.get('source_sha256') and record['source_sha256']!=digest:raise ValueError('Ảnh nguồn đã đổi sau pseudo-label.')
                if digest in seen:continue
                seen.add(digest);image_id=digest[:20];image=Image.fromarray(read_rgb(source));image.thumbnail((1024,1024))
                rel=f'images/{image_id}.jpg';target=project/rel;target.parent.mkdir(exist_ok=True);image.save(target,quality=95)
                prediction=record.get('prediction',{});a=empty_annotation(image_id);a['provenance']='pseudo_proposal_requires_human_review'
                a['subject_id']=record.get('subject_id','');a['notes']=str(record.get('reason','Kiểm tra toàn bộ điểm do model đề xuất.'))[:2000]
                a['handedness']=prediction.get('handedness','unknown');a['mirrored']=prediction.get('mirrored','unknown')
                ref=np.asarray(prediction.get('palm_width_points',[]),float)
                if ref.shape==(2,2) and np.isfinite(ref).all() and np.all((ref>=0)&(ref<=1)):a['palm_width_points']=ref.tolist()
                for name in LINES:
                    proposed=prediction.get('lines',{}).get(name,{});points=np.asarray(proposed.get('points',[]),float)
                    if proposed.get('status')=='absent':a['lines'][name]={'status':'absent','points':[]}
                    elif points.shape==(6,2) and np.isfinite(points).all() and np.all((points>=0)&(points<=1)):a['lines'][name]={'status':'present','points':points.tolist()}
                a=validate_annotation(a,image.size);write_json(annotation_path(project,image_id),a)
                entries.append({'image_id':image_id,'image_path':rel,'width':image.width,'height':image.height,'source_path':str(source),'source_sha256':digest,'sha256':sha256(target),'phash':perceptual_hash(np.asarray(image)),'source_group':str(source.parent),'subject_id':a['subject_id']})
            except (ValueError,OSError,KeyError,TypeError) as exc:errors.append({'error':str(exc),'record':line[:500]})
    write_json(project/'project.json',{'schema':SCHEMA,'n_keypoints':6,'lines':list(LINES),'images':entries,'source_report':str(report),'remaining_count':0,'status':'ready','review_only':True})
    write_json(project/'remaining_sources.json',{'paths':[]});write_json(project/'unreadable_sources.json',errors)
    return {'ok':True,'project':str(project),'images':len(entries),'unreadable':len(errors),'review_only':True}
