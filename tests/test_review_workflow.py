import argparse
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bootstrap_review as boot
import pipeline
import prototype_common as common
import prototype_fixture
import giao_dien_ui as ui


class ReviewWorkflowContracts(unittest.TestCase):
    def test_local_dataset_over_archive_limit_dedupes_without_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); src = root / 'images'; src.mkdir()
            first = src / 'one.png'; second = src / 'duplicate.png'
            first.write_bytes(b'same-content'); second.write_bytes(first.read_bytes())
            with mock.patch.object(pipeline.core, 'list_images', return_value=[first] * 20000 + [second]):
                paths, report = pipeline.collect_directory(src, root / 'raw', None)
            self.assertEqual(paths, [first])
            self.assertEqual(report['duplicates'], 20000)
            self.assertEqual(report['source_files'], 20001)
            self.assertEqual(list((root / 'raw').glob('*.png')), [])
            self.assertTrue(second.exists())

    def test_manual_review_ignores_auto_accept_gates_and_does_not_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fixture = root / 'fixture'
            prototype_fixture.generate_fixture(fixture)
            paths = list(fixture.glob('images/*/input.png'))
            args = argparse.Namespace(seed=42, review_count=3, out_size=128, min_short_side=2048,
                                      strict_line_threshold=0.999, input_dir=str(fixture), run_dir=str(root))
            report = pipeline.prepare_manual_candidates(paths, root / 'data', args)
            queue = pipeline.prepare_review_queue(root / 'data', root / 'review', 3)
            self.assertEqual(queue['queue_count'], 3)
            self.assertEqual(report['accepted'], 0)
            rows = boot.read_csv(root / 'data/labels_pseudo_strict.csv')
            self.assertTrue(all(r['split'] == 'unassigned' and r['accepted'] == '0' for r in rows))
            self.assertEqual(len(json.loads((root / 'data/remaining_sources.json').read_text())['paths']), 3)
            self.assertTrue(all(r['review_status'] == 'pending' for r in boot.read_csv(root / 'review/review.csv')))

    def test_large_group_audit_has_no_10000_cap(self):
        rng = random.Random(12)
        rows = [{'image_path': str(i), 'source_group': str(i), 'sha256': str(i),
                 'phash': f'{rng.getrandbits(63):016x}'} for i in range(10001)]
        result = common.group_rows(rows)
        self.assertEqual(len(result), 10001)
        self.assertEqual({row['split'] for row in result}, {'train','val','test'})

    def test_hash_bands_keep_transitive_near_duplicates_together(self):
        rng = random.Random(27)
        values = [rng.getrandbits(63) for _ in range(80)]
        values += [values[0] ^ ((1 << 6) - 1), values[0] ^ ((1 << 12) - 1)]
        rows = [{'image_path': str(i), 'source_group': str(i), 'sha256': str(i),
                 'phash': f'{v:016x}'} for i,v in enumerate(values)]
        grouped = {int(r['image_path']): r['group_id'] for r in common.group_rows(rows)}
        self.assertEqual(grouped[0], grouped[80]); self.assertEqual(grouped[80], grouped[81])
        for i, a in enumerate(values):
            for j,b in enumerate(values):
                if (a ^ b).bit_count() <= 6:
                    self.assertEqual(grouped[i],grouped[j])

    def test_rejected_and_holdout_sources_never_become_pseudo_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = [{'image_path':'rejected.jpg','source_group':'train-src'},
                    {'image_path':'test-neighbour.jpg','source_group':'test-src'}]
            reviewed = [{'image_path':'test.jpg','source_group':'test-src','split':'test',
                         'phash':'0123456789abcdef','sha256':'held'}]
            with mock.patch.object(boot, 'confidence_prediction', side_effect=AssertionError('Must not predict')):
                generated, skipped, stats = boot.prepare_combined(base,reviewed,root,root/'out',None,32,None,0.9,0.005,None,
                                                                  excluded_images={'rejected.jpg'})
            self.assertFalse(generated)
            self.assertEqual(stats['candidate_count'],1)
            self.assertEqual(skipped[0]['reason'],'held_out_source')

    def test_bootstrap_train_full_synthetic_review_freezes_holdouts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'run/01_preprocessed_dataset'
            prototype_fixture.generate_fixture(data)
            rows=boot.read_csv(data/'labels.csv')
            boot.write_csv(data/'labels_pseudo_strict.csv', rows)
            reviewed=[dict(row,review_id=f'r{i}',review_status='approved',review_notes='TEST FIXTURE ONLY') for i,row in enumerate(rows)]
            boot.write_csv(root/'run/review/review.csv',reviewed)
            args=boot.make_parser().parse_args(['--run_dir',str(root/'run'),'--out_dir',str(root/'boot'),
                                               '--input_size','32','--epochs','1','--train_final'])
            args.max_remaining=None
            result=boot.run(args)
            self.assertTrue(result['ok'])
            seed={r['image_path']:r['split'] for r in boot.read_csv(root/'boot/seed_labels.csv')}
            combined={r['image_path']:r['split'] for r in boot.read_csv(root/'boot/labels_bootstrapped.csv')}
            self.assertEqual(seed,combined)
            self.assertTrue(Path(result['final_checkpoint']).is_file())

    def test_rar_unsafe_paths_blocked_before_backend(self):
        entry=type('Entry',(),{'filename':'../evil.png','file_size':100})()
        with mock.patch.object(ui,'_run_rar_batch_extract') as extract:
            with self.assertRaisesRegex(ValueError,'không an toàn'):
                ui._import_rar_batch('fake.rar',[entry],'test')
            extract.assert_not_called()


    def test_review_pagination_and_assets_do_not_exhaust_ui_rate_budget(self):
        from collections import deque
        rows=[{'review_id':f'r{i}', 'image_path':'images/a.png', 'mask_path':'masks/a.png',
               'overlay_path':'overlays/a.jpg', 'review_status':'pending'} for i in range(81)]
        with mock.patch.object(ui, '_requests', deque()), mock.patch.object(ui, '_read_review_rows', return_value=(Path('.'),Path('review.csv'),rows)):
            client=ui.app.test_client()
            page=client.get('/pipeline_review/test_run?page=2')
            self.assertEqual(page.status_code,200)
            self.assertEqual(page.text.count('<tr data-review-id='),40)
            self.assertIn("data-review-id='r40'",page.text)
            self.assertNotIn("data-review-id='r39'",page.text)
            self.assertIn("loading='lazy'",page.text)
            self.assertIn('trang 2/3',page.text)
            with tempfile.TemporaryDirectory() as folder, mock.patch.object(ui,'ARTIFACTS_DIR',Path(folder)):
                asset=Path(folder)/'test_run/asset.txt'; asset.parent.mkdir(); asset.write_text('test')
                for _ in range(310):
                    response=client.get('/pipeline_artifacts/test_run/asset.txt')
                    self.assertEqual(response.status_code,200)
                    response.close()
                self.assertEqual(client.get('/api/pipeline/status').status_code,200)

    def test_retry_allocates_new_run_without_overwriting_existing(self):
        from collections import deque
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(ui,'ARTIFACTS_DIR',Path(folder)), mock.patch.object(ui,'_requests',deque()), mock.patch.object(ui,'dataset_image_count',return_value=24000), mock.patch.object(ui,'_start_pipeline_command',return_value={'status':'running'}) as start:
            old=Path(folder)/'existing'; old.mkdir(); marker=old/'keep.txt'; marker.write_text('keep')
            response=ui.app.test_client().post('/api/pipeline/prepare-review',json={'run_name':'existing','review_count':400},headers={'X-Palm-CSRF':ui.CSRF_TOKEN})
            self.assertEqual(response.status_code,200)
            self.assertTrue(start.call_args.args[2].startswith('existing_'))
            self.assertEqual(marker.read_text(),'keep')

    def test_bootstrap_preflight_explains_no_approvals_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            boot.write_csv(root/'review/review.csv',[{'review_status':'pending'}])
            with self.assertRaisesRegex(ValueError,'0 mask approved'):
                boot.preflight(root)

    def test_bootstrap_honours_custom_review_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); custom=root/'custom.csv'
            boot.write_csv(custom,[{'review_status':'pending'}])
            with self.assertRaisesRegex(ValueError,'0 mask approved'):
                boot.preflight(root,custom)


    def test_mask_color_preview_preserves_semantic_file(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(ui,'ARTIFACTS_DIR',Path(folder)):
            path=Path(folder)/'test_run/mask.png';path.parent.mkdir()
            mask=np.tile(np.arange(6,dtype=np.uint8),(6,1));cv2.imwrite(str(path),mask)
            original=path.read_bytes()
            response=ui.app.test_client().get('/pipeline_artifacts/test_run/mask.png?view=mask')
            self.assertEqual(response.status_code,200)
            preview=cv2.imdecode(np.frombuffer(response.data,dtype=np.uint8),cv2.IMREAD_COLOR)
            self.assertGreater(int(preview.max()),100)
            self.assertEqual(path.read_bytes(),original)

if __name__ == '__main__':
    unittest.main()
