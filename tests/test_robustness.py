"""Kiểm thử ranh giới nhóm cho phân tích bổ sung, không chạy lại tuning."""
import unittest
import numpy as np
import pandas as pd
from fakejobs.robustness import split_audit

class RobustnessTests(unittest.TestCase):
    def test_rejects_group_overlap_even_when_both_classes_present(self):
        y=pd.Series([0,1,0,1]);g=pd.Series(['a','b','a','c'])
        with self.assertRaises(AssertionError):split_audit(y,g,np.array([0,1]),np.array([2,3]),'overlap')
    def test_rejects_fold_without_positive_class(self):
        y=pd.Series([0,1,0,0]);g=pd.Series(['a','b','c','d'])
        with self.assertRaises(AssertionError):split_audit(y,g,np.array([0,1]),np.array([2,3]),'no-positive')
    def test_preserves_nonconsecutive_training_indices(self):
        y=pd.Series([0,1,0,1],index=[2,4,8,16]);g=pd.Series(['a','b','c','d'],index=y.index)
        audit=split_audit(y,g,np.array([0,1]),np.array([2,3]),'nested-subset')
        self.assertEqual(audit['eval_fake'],1);self.assertEqual(audit['group_overlap'],0)

if __name__=='__main__':unittest.main()
