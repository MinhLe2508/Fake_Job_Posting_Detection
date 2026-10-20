"""Kiểm tra phân rã AP theo nhóm, đặc biệt khi các điểm dự đoán bằng nhau."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from analyze_profile_folds import ranking_details

class FoldDiagnosticsTests(unittest.TestCase):
    def test_ties_use_threshold_endpoint(self):
        rank,p,neg,c=ranking_details([1,0,1,0],[.8,.8,.2,.1])
        np.testing.assert_array_equal(rank,[2,2,3,4])
        np.testing.assert_array_equal(neg,[1,1,1,2])
        self.assertAlmostEqual(c.sum(),7/12)
        self.assertAlmostEqual(p[0],.5)

    def test_group_contributions_add_to_global_ap(self):
        _,_,_,c=ranking_details([1,1,0,0],[.9,.8,.5,.2])
        self.assertAlmostEqual(c[[0,2]].sum()+c[[1,3]].sum(),1.)
        self.assertTrue((c[2:]==0).all())

    def test_invalid_labels_and_nonfinite_scores_rejected(self):
        for y,s in [([1,1],[.2,.3]),([0,2],[.1,.2]),([0,1],[0,float('nan')])]:
            with self.assertRaises(ValueError):ranking_details(y,s)

if __name__=='__main__':unittest.main()
