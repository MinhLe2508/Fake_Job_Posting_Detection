"""Kiểm thử phần mở rộng sau khóa: kiểm định thống kê và việc không chạm vào test."""
import inspect, unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fakejobs import extension as E

class ExtensionTests(unittest.TestCase):
    def test_nadeau_bengio_matches_hand_calculation(self):
        d = np.array([.01, .03, .02, .02, .02])  # mean .02, var 0.00005
        t, p = E.nadeau_bengio(d, .25)
        self.assertAlmostEqual(t, .02 / np.sqrt(.45 * .00005), places=9)
        self.assertGreater(p, 0.0); self.assertLess(p, 1.0)
        # hiệu chỉnh luôn làm p lớn hơn t-test ghép cặp thường
        from scipy import stats
        self.assertGreater(p, stats.ttest_1samp(d, 0).pvalue)
    def test_nadeau_bengio_zero_variance_is_nan(self):
        t, p = E.nadeau_bengio(np.array([.01] * 5), .25)
        self.assertTrue(np.isnan(t) and np.isnan(p))
    def test_holm_is_monotone_and_capped(self):
        adj = E.holm([.04, .01, .30])
        self.assertEqual(adj, [.08, .03, .30])
        self.assertTrue(all(0 <= a <= 1 for a in adj))
    def test_extension_never_loads_test_split(self):
        parts, groups, folds = E.load_train_validation()
        self.assertEqual(set(parts), {'train', 'validation'})
        self.assertFalse(set(groups['train']) & set(groups['validation']))
        self.assertEqual(len(folds), 5)
        source = inspect.getsource(E)
        self.assertNotIn("'test'", source.replace("'test_policy'", "").replace("'test_over_train'", "").split('def write_protocol')[0])
    def test_added_budget_matches_main_protocol(self):
        configs = E.extension_configurations()
        self.assertEqual(len(configs), 16)
        self.assertEqual({c['model'] for c in configs}, {'NaiveBayes', 'HistGradientBoosting'})
    def test_validation_timing_excludes_prediction(self):
        # Đồng hồ giả kiểm chứng lỗi từng gặp: fit=2 giây, dự đoán=3 giây.
        # Không fit mô hình thật và không ghi kết quả vào bộ nộp.
        frame=pd.DataFrame({'job_id':[1,2], 'fraudulent':[0,1], 'title':['a','b']})
        parts={'train':frame, 'validation':frame}
        class FakeModel:
            def fit(self, X, y): return self
        cfg={'config_id':'LinearSVM_03','model':'LinearSVM','params':{}}
        with patch.object(E,'load_train_validation',return_value=(parts,{'validation':pd.Series(['a','b'])},[])), \
             patch.object(E,'configurations',return_value=[cfg]), \
             patch.object(E,'feature_views',return_value={'sparse':(np.zeros((2,1)),np.zeros((2,1)))}), \
             patch.object(E,'make_estimator',return_value=FakeModel()), \
             patch.object(E,'scores',return_value=np.array([0.,1.])), \
             patch.object(E,'load_json',return_value={'AP':1.}), \
             patch.object(E.time,'perf_counter',side_effect=[10.,12.,20.,23.]), \
             patch.object(E,'dump') as saved:
            E.run_validation_compare({'LinearSVM':'LinearSVM_03'})
        result=saved.call_args.args[1]
        self.assertEqual(result['models']['LinearSVM']['fit_seconds'],2.)
        self.assertEqual(result['models']['LinearSVM']['predict_seconds'],3.)
        self.assertTrue(result['timing_schema'].startswith('v2'))

if __name__ == '__main__': unittest.main()
