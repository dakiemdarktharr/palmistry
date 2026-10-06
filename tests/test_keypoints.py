"""Regression tests for keypoints, numerical geometry, gradients and isolation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from flask import Flask
from prototype_common import sha256,perceptual_hash
from palm_keypoints import LINES,SCHEMA,TOTAL_POINTS,COORDS,LINE_POINTS,LEGACY_SCHEMA
from palm_keypoints.data import empty_annotation,write_json,annotation_path,validate_annotation,audited_rows,export_yolo,create_from_review,read_json,limit_review_queue,migrate_project
from palm_keypoints.spline import trace_line,width_from_points
from palm_keypoints.model import PoseCNN,prepare_image,to_model,from_model,targets
from palm_keypoints.workflow import train,infer,pseudo
from palm_keypoints.web import register_keypoint_ui


def annotation(image_id):
    a=empty_annotation(image_id);a.update(status='approved',handedness='right',mirrored='no',palm_width_points=[[.15,.8],[.85,.8]])
    for i,name in enumerate(LINES):a['lines'][name]={'status':'present','points':[[float(x),.25+i*.12] for x in np.linspace(.2,.8,6)]}
    return a


def fixture(root,n=12):
    root=Path(root);rows=[];rng=np.random.default_rng(876)
    for i in range(n):
        image=rng.integers(0,255,(64,64,3),dtype=np.uint8);image_id=f'{i:020x}';rel=f'images/{image_id}.jpg';path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(image).save(path)
        a=annotation(image_id);a['subject_id']=f'subject_{i}';write_json(annotation_path(root,image_id),a)
        rows.append({'image_id':image_id,'image_path':rel,'source_path':str(path),'source_sha256':sha256(path),'width':64,'height':64,'sha256':sha256(path),'phash':perceptual_hash(image),'source_group':'one_session','subject_id':''})
    write_json(root/'project.json',{'schema':SCHEMA,'images':rows});write_json(root/'remaining_sources.json',{'paths':[],'source_root':str(root)})
    return rows


class SplineTests(unittest.TestCase):
    def test_straight_exact_and_scale(self):
        p=np.c_[np.linspace(.2,.8,6),np.full(6,.4)]
        a=trace_line(p,(501,401),300);b=trace_line(p,(1001,801),600)
        self.assertTrue(a['valid']);self.assertAlmostEqual(a['length_px'],300,places=5)
        self.assertAlmostEqual(a['length_to_palm_ratio'],b['length_to_palm_ratio'],places=8)
        self.assertEqual(a['connected_paths'],1);self.assertEqual(a['branch_count'],0)
    def test_arc(self):
        t=np.linspace(0,np.pi/2,6);p=np.c_[.4+.3*np.cos(t),.4+.3*np.sin(t)]
        r=trace_line(p,(1001,1001),600);self.assertTrue(r['valid']);self.assertLess(abs(r['length_px']-300*np.pi/2),.6)
    def test_missing_duplicate_bounds_confidence(self):
        p=np.c_[np.linspace(.2,.8,6),np.full(6,.4)]
        for points in (p[:5],np.tile(p[0],(6,1)),p+2,np.full((6,2),np.nan)):
            self.assertFalse(trace_line(points,(501,401),300)['valid'])
        r=trace_line(p,(501,401),300,confidence=[.8]*6);self.assertFalse(r['valid']);self.assertIn('low_keypoint_confidence',r['reasons'])
    def test_turn_crossing_and_width(self):
        p=[[.1,.2],[.8,.8],[.2,.8],[.8,.2],[.4,.1],[.2,.1]]
        self.assertFalse(trace_line(p,(501,401),300)['valid'])
        with self.assertRaises(ValueError):width_from_points([[.2,.2],[.2,.2]],(500,500))


class ModelTests(unittest.TestCase):
    def test_partial_annotation_masks_unknown_coordinates_and_presence(self):
        a=empty_annotation('0'*20);a['handedness']='left'
        a['lines']['heart_line']={'status':'unreviewed','points':[[.2,.3],[.4,.3]]}
        _,meta=prepare_image(np.zeros((64,64,3),np.uint8));y,m=targets(a,meta)
        self.assertEqual(len(y),44);self.assertEqual(m[:-1].sum(),0);self.assertEqual(m[-1],1)
        self.assertEqual(y[-1],0)
        a['lines']['head_line']={'status':'absent','points':[]};_,m=targets(a,meta)
        self.assertEqual(m[COORDS+1],1);self.assertEqual(m[COORDS],0)

    def test_unknown_side_is_not_a_training_label(self):
        a=annotation('0'*20);a.update(handedness='unknown',mirrored='unknown')
        x,meta=prepare_image(np.zeros((64,64,3),np.uint8));y,m=targets(a,meta)
        self.assertEqual(m[-1],0);self.assertTrue(np.all(m[COORDS:COORDS+len(LINES)]==1))
        model=PoseCNN();loss,grad=model.loss_grad(x[None],y[None],m[None],dropout=0)
        changed=y.copy();changed[-1]=1-y[-1]
        other,other_grad=model.loss_grad(x[None],changed[None],m[None],dropout=0)
        self.assertEqual(loss,other);self.assertEqual(grad['b4'][-1],0)
        for key in grad:np.testing.assert_array_equal(grad[key],other_grad[key])

    def test_coordinate_letterbox_roundtrip(self):
        _,meta=prepare_image(np.zeros((120,300,3),np.uint8));p=np.array([[0,0],[1,1],[.2,.7]])
        np.testing.assert_allclose(from_model(to_model(p,meta),meta),p,atol=1e-7)
    def test_gradients_and_learning(self):
        model=PoseCNN(2);x=np.random.default_rng(5).normal(0,.3,(2,64,64,3)).astype(np.float32);y=np.full((2,COORDS+len(LINES)+1),.7,np.float32);mask=np.ones_like(y)
        first,grads=model.loss_grad(x,y,mask,dropout=0)
        for key in ('w1','w2','w3','w4','b4'):
            index=np.unravel_index(np.argmax(np.abs(grads[key])),grads[key].shape);value=model.p[key][index];eps=.002
            model.p[key][index]=value+eps;plus=model.loss_grad(x,y,mask,dropout=0)[0];model.p[key][index]=value-eps;minus=model.loss_grad(x,y,mask,dropout=0)[0];model.p[key][index]=value
            numerical=(plus-minus)/(2*eps);self.assertAlmostEqual(float(grads[key][index]),numerical,delta=max(3e-5,abs(numerical)*.08),msg=key)
        for _ in range(25):loss,g=model.loss_grad(x,y,mask,dropout=0);model.update(g)
        self.assertLess(loss,first*.55)
    def test_absence_mask_mirror_and_checkpoint(self):
        a=annotation('0'*20);a['lines']['life_line']={'status':'absent','points':[]};a['mirrored']='yes'
        x,meta=prepare_image(np.zeros((64,64,3),np.uint8));y,m=targets(a,meta);self.assertEqual(m[24:36].sum(),0);self.assertEqual(y[-1],1)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'m.npz';model=PoseCNN();model.save(path,{})
            loaded=PoseCNN.load(path,sha256(path));np.testing.assert_array_equal(model.predict_tensor(x[None]),loaded.predict_tensor(x[None]))
            with self.assertRaises(ValueError):PoseCNN.load(path,'0'*64)
            self.assertTrue(infer(loaded,np.zeros((64,64,3),np.uint8),'no')['needs_review'])


class DataTests(unittest.TestCase):
    def test_v1_migration_keeps_original_fourth_line_and_backup(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=fixture(root,3);doc=read_json(root/'project.json');doc['schema']=LEGACY_SCHEMA;write_json(root/'project.json',doc)
            for row in rows:
                path=annotation_path(root,row['image_id']);a=read_json(path);a['schema']=LEGACY_SCHEMA;a['lines']['fate_line']={'status':'unreviewed','points':[[.4,.4]]};write_json(path,a)
            result=migrate_project(root);self.assertEqual(result['changed'],3)
            migrated=read_json(annotation_path(root,rows[0]['image_id']))
            self.assertEqual(migrated['schema'],SCHEMA);self.assertEqual(set(migrated['lines']),set(LINES))
            self.assertEqual(migrated['legacy_lines']['fate_line']['points'],[[.4,.4]])
            backup=read_json(Path(result['backup'])/'annotations'/(rows[0]['image_id']+'.json'))
            self.assertEqual(backup['schema'],LEGACY_SCHEMA);self.assertIn('fate_line',backup['lines'])
            self.assertEqual(migrate_project(root)['changed'],0)

    def test_approved_partial_annotation_keeps_blanks_and_exports_fixed_shape(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=fixture(root,6)
            for row in rows:
                a=empty_annotation(row['image_id']);a.update(status='approved',handedness='left',subject_id=row['image_id'])
                a['lines']['heart_line']['points']=[[.1,.2]]
                self.assertEqual(validate_annotation(a,(64,64))['lines']['heart_line']['status'],'unreviewed')
                write_json(annotation_path(root,row['image_id']),a)
            out=root/'export';export_yolo(root,out)
            values=next((out/'labels').rglob('*.txt')).read_text().split()
            self.assertEqual(len(values),65);self.assertTrue(all(float(v)==0 for v in values[5:]))

    def test_approval_without_optional_hand_metadata(self):
        a=annotation('0'*20);a.update(handedness='unknown',mirrored='unknown')
        self.assertEqual(validate_annotation(a,(500,500))['status'],'approved')
        a['lines']['fate_line']={'status':'unreviewed','points':[]}
        with self.assertRaises(ValueError):validate_annotation(a,(500,500))

    def test_reduce_queue_preserves_reviewed_labels_and_sources(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=fixture(root,6)
            for row in rows[:-1]:write_json(annotation_path(root,row['image_id']),empty_annotation(row['image_id']))
            hashes={r['image_id']:sha256(annotation_path(root,r['image_id'])) for r in rows}
            result=limit_review_queue(root,3);doc=read_json(root/'project.json')
            self.assertEqual(result['images'],3);self.assertEqual(result['deferred'],3)
            self.assertIn(rows[-1]['image_id'],[r['image_id'] for r in doc['images']])
            self.assertEqual(len(read_json(result['backup'])['project']['images']),6)
            self.assertEqual(len(read_json(root/'remaining_sources.json')['paths']),3)
            for row in rows:self.assertEqual(hashes[row['image_id']],sha256(annotation_path(root,row['image_id'])))
            self.assertEqual(limit_review_queue(root,3)['deferred'],0)
            with self.assertRaises(ValueError):limit_review_queue(root,0)

    def test_reduce_queue_refuses_to_hide_reviewed_images(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root,4);before=sha256(root/'project.json')
            with self.assertRaises(ValueError):limit_review_queue(root,3)
            self.assertEqual(before,sha256(root/'project.json'))

    def test_schema_and_approval_contract(self):
        a=annotation('0'*20);validate_annotation(a,(500,500));a['lines']['simian_line']=a['lines'].pop('life_line')
        with self.assertRaises(ValueError):validate_annotation(a,(500,500))
        for change in ('missing','absent_points','bad_type'):
            a=annotation('0'*20)
            if change=='missing':a['lines']['head_line']['points'].pop()
            if change=='absent_points':a['lines']['head_line']['status']='absent'
            if change=='bad_type':a['lines']['head_line']=7
            with self.assertRaises(ValueError):validate_annotation(a,(500,500))
    def test_subject_split_and_export(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'p';fixture(root);rows=audited_rows(root)
            self.assertEqual({r['split'] for r in rows},{'train','val','test'})
            self.assertTrue(all(r['source_group'].startswith('subject:') for r in rows))
            out=Path(t)/'export';r=export_yolo(root,out);self.assertEqual(r['count'],12)
            self.assertEqual(len(next((out/'labels').rglob('*.txt')).read_text().split()),65)
            self.assertNotIn('fate_line', (out/'schema.json').read_text())
            self.assertTrue(all((out/r['image_path']).is_file() for r in read_json(out/'annotations.json')['images']))
    def test_legacy_review_manifest_join(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=fixture(root/'source',3);run=root/'run';(run/'review').mkdir(parents=True);(run/'01_preprocessed_dataset').mkdir()
            with (run/'review/review.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['image_path','review_status']);writer.writeheader();writer.writerows({'image_path':r['image_path'],'review_status':'pending'} for r in rows)
            with (run/'01_preprocessed_dataset/labels_pseudo_strict.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['image_path','source_path']);writer.writeheader();writer.writerows({k:r[k] for k in ('image_path','source_path')} for r in rows)
            result=create_from_review(run,root/'new',2);self.assertEqual(result['images'],2)

    def test_pending_and_duplicate_cannot_force_split(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=fixture(root,3)
            for row in rows:
                a=read_json(annotation_path(root,row['image_id']));a['subject_id']='same_person';write_json(annotation_path(root,row['image_id']),a)
            with self.assertRaises(ValueError):audited_rows(root)
            for row in rows:write_json(annotation_path(root,row['image_id']),empty_annotation(row['image_id']))
            with self.assertRaises(ValueError):audited_rows(root)
    def test_new_queue_preserves_sources_and_remaining(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'source';rows=fixture(source,3);run=root/'run';(run/'review').mkdir(parents=True)
            with (run/'review/review.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['source_path']);writer.writeheader();writer.writerows({'source_path':r['source_path']} for r in rows)
            before=[sha256(r['source_path']) for r in rows];out=root/'new';create_from_review(run,out,2)
            self.assertEqual(len(read_json(out/'remaining_sources.json')['paths']),1)
            self.assertEqual(before,[sha256(r['source_path']) for r in rows])
            self.assertTrue(all(read_json(p)['status']=='pending' for p in (out/'annotations').glob('*.json')))


class WorkflowTests(unittest.TestCase):
    def test_train_with_only_swiped_hand_labels(self):
        with tempfile.TemporaryDirectory() as t:
            project=Path(t);rows=fixture(project,6)
            for row in rows:
                a=empty_annotation(row['image_id']);a.update(status='approved',handedness='right',subject_id=row['image_id'])
                write_json(annotation_path(project,row['image_id']),a)
            result=train(project,project/'runs/train',epochs=1,batch_size=3)
            self.assertEqual(result['schema'],SCHEMA);self.assertEqual(result['validation']['reference_samples'],0)
            self.assertIsNone(result['validation']['reference_error_palm_width'])
            for line in result['validation']['lines'].values():
                self.assertEqual(line['present'],0);self.assertEqual(line['absent'],0)
                self.assertFalse(line['auto_label_supported'])

    def test_train_with_unknown_side_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            project=Path(t);rows=fixture(project,6)
            for row in rows:
                a=read_json(annotation_path(project,row['image_id']));a.update(handedness='unknown',mirrored='unknown')
                write_json(annotation_path(project,row['image_id']),a)
            result=train(project,project/'runs/train',epochs=1,batch_size=3)
            self.assertIsNone(result['validation']['hand_accuracy'])
            self.assertEqual(result['validation']['hand_samples'],0)
            self.assertTrue(np.isfinite(result['validation']['reference_error_palm_width']))

    def test_train_and_pseudo_review_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            project=Path(t)/'p';fixture(project);out=project/'runs/train';result=train(project,out,epochs=2,batch_size=6)
            self.assertTrue(result['ok']);self.assertIn('test',result)
            source=project/'unlabeled.jpg';Image.fromarray(np.random.default_rng(99).integers(0,255,(64,64,3),dtype=np.uint8)).save(source)
            write_json(project/'remaining_sources.json',{'source_root':str(project),'paths':[str(source)]})
            output=project/'pseudo'
            with patch('palm_keypoints.workflow.read_rgb',side_effect=AssertionError('Do not decode GB of images when identity metadata is missing')):
                r=pseudo(project,result['checkpoint'],result['sha256'],output)
            self.assertEqual(r['accepted'],0);self.assertEqual(r['review'],1)
            self.assertIn('subject_identity_required', (output/'manual_review.jsonl').read_text())
            split=read_json(out/'split.json');held=next(r for r in split if r['split']=='test')
            write_json(project/'remaining_sources.json',{'source_root':str(project),'paths':[str(source)],'metadata':{str(source):{'subject_id':held['subject_id'],'mirrored':'no'}}})
            r=pseudo(project,result['checkpoint'],result['sha256'],project/'pseudo2');self.assertEqual(r['accepted'],0)
            self.assertIn('held_out_subject', (project/'pseudo2/manual_review.jsonl').read_text())
    def test_validation_and_geometry_gates(self):
        model=PoseCNN();a=annotation('0'*20);points=np.array([p for name in LINES for p in a['lines'][name]['points']]+a['palm_width_points'])
        prediction={'points':points,'std':np.zeros((TOTAL_POINTS,2)),'presence':np.ones(len(LINES))*.999,'right_hand_score':.999}
        model.metadata={'positive_counts':dict.fromkeys(LINES,10),'validation':{'samples':10,'reference_error_palm_width':.01,'hand_accuracy':1,'lines':{k:{'auto_label_supported':True} for k in LINES}}}
        with patch.object(model,'predict',return_value=prediction):
            result=infer(model,np.zeros((500,500,3),np.uint8),'no');self.assertFalse(result['needs_review'])
            model.metadata['validation']['hand_accuracy']=None
            result=infer(model,np.zeros((500,500,3),np.uint8))
            self.assertFalse(result['needs_review']);self.assertEqual(result['handedness'],'unknown')
            prediction['points'][0]=prediction['points'][1];self.assertTrue(infer(model,np.zeros((500,500,3),np.uint8),'no')['needs_review'])


class WebTests(unittest.TestCase):
    def test_swipe_saves_hand_and_partial_data_without_filling_missing_fields(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);project=root/'artifacts/keypoints/test';rows=fixture(project,3)
            app=Flask(__name__);register_keypoint_ui(app,root,'token','nonce');c=app.test_client()
            image_id=rows[0]['image_id'];a=empty_annotation(image_id)
            a['lines']['heart_line']['points']=[[.2,.3]]
            url=f'/api/keypoints/swipe/test/{image_id}'
            result=c.post(url,json={'handedness':'left','annotation':a});self.assertEqual(result.status_code,200)
            stored=read_json(annotation_path(project,image_id))
            self.assertEqual(stored['handedness'],'left');self.assertEqual(stored['status'],'approved')
            self.assertEqual(stored['palm_width_points'],[]);self.assertEqual(stored['lines']['life_line']['status'],'unreviewed')
            self.assertEqual(stored['lines']['heart_line']['points'],[[.2,.3]])
            self.assertEqual(c.post(url,json={'handedness':'right','annotation':a}).status_code,409)
            b=result.json['annotation'];result=c.post(url,json={'handedness':'right','annotation':b})
            self.assertEqual(result.status_code,200);self.assertEqual(result.json['annotation']['handedness'],'right')

    def test_metadata_and_review_queue(self):
        import io
        from palm_keypoints.data import create_from_pseudo
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);project=root/'artifacts/keypoints/test';rows=fixture(project,3)
            app=Flask(__name__);register_keypoint_ui(app,root,'token','nonce');c=app.test_client()
            csv_text='source_path,subject_id,mirrored,handedness\n'+rows[0]['source_path']+',actual_subject,yes,left\n'
            r=c.post('/api/keypoints/metadata/test',data={'file':(io.BytesIO(csv_text.encode()),'metadata.csv')});self.assertEqual(r.status_code,200)
            a=read_json(annotation_path(project,rows[0]['image_id']));self.assertEqual(a['status'],'pending');self.assertEqual(a['mirrored'],'yes');self.assertEqual(a['subject_id'],'actual_subject')
            report=project/'manual_review.jsonl';report.write_text(json.dumps({'source_path':rows[0]['source_path'],'source_sha256':rows[0]['source_sha256'],'reason':'confidence_low'})+'\n',encoding='utf-8')
            out=root/'review';result=create_from_pseudo(report,out);self.assertEqual(result['images'],1)
            with self.assertRaises(ValueError):audited_rows(out)
            self.assertEqual(c.get('/api/keypoints/annotations-export/test').status_code,200)

    def test_draft_revision_validation_and_membership(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=fixture(root/'artifacts/keypoints/test',3);app=Flask(__name__);register_keypoint_ui(app,root,'token','nonce');c=app.test_client();image_id=rows[0]['image_id'];url=f'/api/keypoints/annotation/test/{image_id}'
            a=c.get(url).json['annotation'];a['status']='pending';r=c.post(url,json=a);self.assertEqual(r.status_code,200)
            self.assertEqual(c.post(url,json=a).status_code,409)
            a=r.json['annotation'];a['lines']['heart_line']=7;self.assertEqual(c.post(url,json=a).status_code,400)
            self.assertEqual(c.get('/keypoints/assets/test/'+'f'*20).status_code,400)
            self.assertEqual(c.post('/api/keypoints/action',json={'kind':'oops','project':'test'}).status_code,400)

if __name__=='__main__':unittest.main()
