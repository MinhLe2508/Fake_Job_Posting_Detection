"""Kiểm thử hành vi có ý nghĩa: chống leakage, input thiếu/lạ, parity và khóa test."""
import json, unittest
import numpy as np
import pandas as pd
from fakejobs.experiment import ROOT,load_data,load_json,metrics,sha
from fakejobs.inference import Detector
from fakejobs.features import make_features

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts,cls.groups,cls.folds=load_data();cls.detector=Detector()
    def test_group_disjoint_and_minority_every_fold(self):
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse(set(self.groups[a])&set(self.groups[b]))
        for tr,va in self.folds:
            self.assertEqual(int(self.parts['train'].fraudulent.iloc[va].sum()),106)
            self.assertFalse(set(self.groups['train'].iloc[tr])&set(self.groups['train'].iloc[va]))
    def test_serialized_prediction_parity(self):
        predictions=pd.read_csv(ROOT/'results/validation_predictions.csv').set_index('job_id')
        for row in self.parts['validation'].head(3).to_dict('records'):
            expected=predictions.loc[row.pop('job_id')];row.pop('fraudulent')
            fields={k:None if pd.isna(v) else v for k,v in row.items()}
            actual=self.detector.predict(fields)
            self.assertAlmostEqual(actual['score'],expected.score,places=10)
            self.assertEqual(actual['flagged'],bool(expected.predicted))
    def test_missing_and_unknown_category_supported(self):
        result=self.detector.predict({'title':'Data analyst','industry':'UNSEEN_CATEGORY_999'})
        self.assertTrue(np.isfinite(result['score']))
    def test_invalid_input_rejected(self):
        for bad in [{},{'fraudulent':1,'title':'Job'},{'title':'Job','has_company_logo':2},{'title':['Job']},{'title':'a'*100001}]:
            with self.assertRaises(ValueError):self.detector.predict(bad)
    def test_transform_does_not_change_model(self):
        before=sha(ROOT/'models/fake_job_detector.joblib')
        self.detector.predict({'description':'Unknown words zxqvbnm 3450123'})
        self.assertEqual(before,sha(ROOT/'models/fake_job_detector.joblib'))
        vocab=self.detector.bundle['pipeline'].named_steps['preprocess'].named_steps['columns'].named_transformers_['text'].vocabulary_
        self.assertNotIn('zxqvbnm',vocab)
    def test_metrics_recomputed_from_frozen_test_predictions(self):
        p=pd.read_csv(ROOT/'results/test_predictions.csv');saved=load_json(ROOT/'results/test_metrics.json')
        actual=metrics(p.fraudulent,p.score,saved['threshold'])
        for key,value in actual.items():self.assertAlmostEqual(value,saved[key],places=10)
    def test_protocol_and_model_hashes_match_lock(self):
        lock=load_json(ROOT/'models/lock.json')
        self.assertEqual(sha(ROOT/'results/protocol.json'),lock['protocol_sha256'])
        self.assertEqual(sha(ROOT/'models/fake_job_detector.joblib'),lock['model_sha256'])
        self.assertFalse(lock['test_used_for_selection'])
    def test_features_independent_of_labels(self):
        frame=self.parts['train'].head(5).copy();a=make_features(frame)
        frame['fraudulent']=1-frame['fraudulent'];frame['job_id']=-1
        pd.testing.assert_frame_equal(a,make_features(frame))

if __name__=='__main__':unittest.main(verbosity=2)
