import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import palm_geometry as geometry
import palmistry_strict_auto_onefile as core
import prototype_common as common
import tien_xu_ly as converter


def open_hand():
    mask=np.zeros((280,240),np.uint8)
    cv2.rectangle(mask,(80,105),(160,215),1,-1)
    cv2.rectangle(mask,(100,210),(141,265),1,-1)
    for x,top in [(80,44),(103,20),(126,28),(149,53)]:
        cv2.rectangle(mask,(x,top),(x+11,120),1,-1)
    cv2.fillPoly(mask,[np.array([[85,120],[35,138],[24,153],[41,169],[83,177]],np.int32)],1)
    return mask


class PalmGeometryTests(unittest.TestCase):
    def test_horizontal_diagonal_lengths_are_arc_distances(self):
        horizontal=np.zeros((80,80),np.uint8);horizontal[20,10:51]=1
        diagonal=np.zeros_like(horizontal)
        for i in range(10,51):diagonal[i,i]=1
        self.assertAlmostEqual(geometry.principal_trace(horizontal)['length_px'],40)
        self.assertAlmostEqual(geometry.principal_trace(diagonal)['length_px'],40*math.sqrt(2))

    def test_branch_is_not_summed_into_main_trace(self):
        sk=np.zeros((100,140),np.uint8);sk[50,10:111]=1;sk[40:51,60]=1
        result=geometry.principal_trace(sk)
        self.assertAlmostEqual(result['length_px'],100)
        self.assertLess(result['length_px'],float(sk.sum()))

    def test_disconnected_lines_do_not_add_lengths(self):
        sk=np.zeros((80,100),np.uint8);sk[20,10:40]=1;sk[40,60:90]=1
        self.assertFalse(geometry.principal_trace(sk)['detected'])

    def test_length_ratio_is_padding_and_scale_invariant(self):
        sem=np.zeros((120,120),np.uint8);cv2.circle(sem,(60,60),40,1,-1);sem[60,30:91]=3
        a=core.line_feature_from_mask(sem,3)
        b=core.line_feature_from_mask(np.pad(sem,30),3)
        c=core.line_feature_from_mask(cv2.resize(sem,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST),3)
        self.assertTrue(a['detected']);self.assertAlmostEqual(a['length_norm'],b['length_norm'],delta=1e-6)
        self.assertAlmostEqual(a['length_norm'],c['length_norm'],delta=.04)

    def test_handedness_and_mirror_metadata(self):
        mask=open_hand()
        self.assertEqual(geometry.infer_handedness(mask,False)['side'],'right')
        self.assertEqual(geometry.infer_handedness(np.fliplr(mask),False)['side'],'left')
        self.assertEqual(geometry.infer_handedness(np.fliplr(mask),True)['side'],'right')
        self.assertEqual(geometry.infer_handedness(mask,None)['side'],'unknown')

    def test_unclear_pose_and_multiple_regions_abstain(self):
        self.assertEqual(geometry.infer_handedness(np.rot90(open_hand()),False)['side'],'unknown')
        two=np.concatenate([open_hand(),open_hand()],axis=1)
        self.assertEqual(geometry.infer_handedness(two,False)['reason'],'multiple_regions')
        self.assertEqual(geometry.infer_handedness(np.ones((50,50)),False)['side'],'unknown')

    def test_simian_is_separate_from_unknown_and_conflicts_abstain(self):
        self.assertEqual(common.CLASS_MAP[5],'minor_or_unknown_line')
        self.assertEqual(common.CLASS_MAP[6],'simian_line')
        self.assertEqual(converter.class_id_to_mask_class(0,{0:'simian_line'}),6)
        result=geometry.resolve_transverse_lines({'head_line':{'detected':True},'simian_line':{'detected':True},'life_line':{'detected':True}})
        self.assertFalse(result['head_line']['detected']);self.assertFalse(result['simian_line']['detected'])
        self.assertTrue(result['life_line']['detected'])
        self.assertTrue(geometry.resolve_transverse_lines({'simian_line':{'detected':True}})['simian_line']['detected'])

    def test_unlearned_simian_cannot_be_predicted_or_appear_after_reload(self):
        n=len(common.CLASS_MAP);features=len(core.MODEL_FEATURE_NAMES)
        model=core.NumpyPixelModel(np.zeros((n,features)),np.ones((n,features)),np.ones(n),32,{'class_sample_counts':[10,10,10,10,10,10,0]})
        image=np.zeros((32,32,3),np.uint8)
        self.assertTrue(np.all(model.predict_proba(image)[6]==0))
        with tempfile.TemporaryDirectory() as folder:
            ckpt=Path(folder)/'m.npz';core._save_numpy_checkpoint(ckpt,model,{'input_size':32})
            loaded,_,_=core.load_model_for_predict(ckpt,expected_sha256=core.sha256_file(ckpt))
            self.assertTrue(np.all(loaded.predict_proba(image)[6]==0))

    def test_legacy_class_map_rejected_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'class_map.json';p.write_bytes(Path('class_map.v1.json').read_bytes());before=p.read_bytes()
            with self.assertRaisesRegex(ValueError,'V2 adds simian'):
                common.check_class_map(p)
            self.assertEqual(p.read_bytes(),before)


    def test_simian_training_and_confusion_matrix_include_class_six(self):
        import prototype_fixture
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);prototype_fixture.generate_fixture(root)
            rows=common.read_manifest(root,root/'labels.csv')
            row=next(r for r in rows if r['split']=='train')
            mask_path=root/row['mask_path'];mask=cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED)
            mask[mask==3]=6;mask[mask==4]=1;cv2.imwrite(str(mask_path),mask)
            model=core._fit_numpy_model(root,[row],32,42)
            self.assertGreater(model.metadata['class_sample_counts'][6],0)
            self.assertTrue(model.learned_classes[6])
            scores=model.predict_proba(core.imread_rgb(root/row['image_path']))
            self.assertTrue(np.any(scores[6]>0))
            cm=common.confusion_matrix(np.array([[6]],np.uint8),np.array([[6]],np.uint8))
            self.assertEqual(cm.shape,(7,7));self.assertEqual(cm[6,6],1)

if __name__=='__main__':unittest.main()
