import csv
import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import giao_dien_ui as ui
import pipeline
import prototype_fixture
import palm_photo_ui


class PhotoAndMigrationTests(unittest.TestCase):
    def setUp(self):
        self.client=ui.app.test_client();ui._requests.clear()
        self.headers={'X-Palm-CSRF':ui.CSRF_TOKEN}

    def test_photo_screen_and_invalid_requests(self):
        self.assertEqual(self.client.get('/analyze').status_code,200)
        self.assertEqual(self.client.post('/api/photo/analyze').status_code,403)
        response=self.client.post('/api/photo/analyze',headers=self.headers)
        self.assertEqual(response.status_code,400)
        response=self.client.post('/api/photo/analyze',headers=self.headers,data={'image':(io.BytesIO(b'bad'),'bad.png'),'mode':'invalid'})
        self.assertEqual(response.status_code,400)

    def test_photo_real_subprocess_returns_retake_for_small_blank_image(self):
        ok,encoded=cv2.imencode('.png',np.zeros((64,64,3),np.uint8));self.assertTrue(ok)
        response=self.client.post('/api/photo/analyze',headers=self.headers,data={'image':(io.BytesIO(encoded.tobytes()),'blank.png'),'mode':'classical','mirrored':'no'})
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.json['ok'])
        self.assertEqual(response.json['result']['status'],'need_retake')
        self.assertFalse(response.json['result'].get('accepted_lines'))
        self.assertNotIn('provenance',response.json['result'])

    def test_photo_timeout_has_fallback(self):
        with mock.patch.object(palm_photo_ui.subprocess,'run',side_effect=palm_photo_ui.subprocess.TimeoutExpired('test',60)):
            response=self.client.post('/api/photo/analyze',headers=self.headers,data={'image':(io.BytesIO(b'fake'),'fake.png'),'mode':'classical'})
        self.assertEqual(response.status_code,504)
        self.assertIn('nhỏ hơn',response.json['error'])

    def test_review_migration_is_copy_only_and_all_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'old';data=old/'01_preprocessed_dataset'
            prototype_fixture.generate_fixture(data)
            legacy=Path('class_map.v1.json').read_bytes();(data/'class_map.json').write_bytes(legacy)
            with (data/'labels.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
            pipeline.core.write_csv(data/'labels_pseudo_strict.csv',rows)
            pipeline.prepare_review_queue(data,old/'review',6)
            review=old/'review/review.csv';before=review.read_bytes()
            result=pipeline.migrate_review_v2(old,root/'new')
            self.assertTrue(result['ok']);self.assertEqual(review.read_bytes(),before)
            self.assertEqual((data/'class_map.json').read_bytes(),legacy)
            with (root/'new/review/review.csv').open(encoding='utf-8') as f: migrated=list(csv.DictReader(f))
            self.assertEqual(len(migrated),6)
            self.assertTrue(all(r['review_status']=='pending' for r in migrated))
            self.assertEqual(json.loads((root/'new/01_preprocessed_dataset/class_map.json').read_text())['classes']['6'],'simian_line')
            with self.assertRaises(ValueError):pipeline.migrate_review_v2(old,root/'new')

if __name__=='__main__':unittest.main()
