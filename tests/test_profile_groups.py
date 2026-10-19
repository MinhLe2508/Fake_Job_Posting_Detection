"""Kiểm tra nhóm nối bắc cầu và xử lý profile thiếu trước đánh giá."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_generalization import connected_groups

class ProfileGroupsTest(unittest.TestCase):
    def test_transitive_content_or_profile(self):
        g,p=connected_groups(['a','a','b','c'],['First','Second',' second ','Other'])
        self.assertEqual(g[0],g[2])
        self.assertNotEqual(g[0],g[3])
        self.assertEqual(p[1],p[2])

    def test_missing_profiles_do_not_merge_unrelated_content(self):
        g,p=connected_groups(['a','b','c','c'],[None,'','  ',None])
        self.assertEqual(g.nunique(),3)
        self.assertTrue(p.eq('').all())

    def test_normalization_and_disjoint_profiles(self):
        g,p=connected_groups(['a','b','c'],[' ACME\n Ltd ','acme ltd','Elsewhere'])
        self.assertEqual(g[0],g[1])
        self.assertNotEqual(g[1],g[2])

if __name__=='__main__':unittest.main()
