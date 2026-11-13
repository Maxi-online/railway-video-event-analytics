import unittest
from src.utils.zone_events import feet_in_zone
from src.utils.geometry_utils import intersection_over_union
from offline_demo import run

class ZoneTests(unittest.TestCase):
    def test_replay(self): self.assertEqual(run()["frame_zone_states"],[False,True,True,False])
    def test_polygon(self): self.assertTrue(feet_in_zone([2,2,4,4],[[0,0],[8,0],[8,8],[0,8]]))
    def test_degenerate_detection_is_rejected(self): self.assertFalse(feet_in_zone([1,1,1,1],[0,0,8,8]))
    def test_threshold_is_strict(self): self.assertFalse(feet_in_zone([0,0,10,10],[0,0,5,10],0.5))
    def test_invalid_threshold(self):
        with self.assertRaises(ValueError): feet_in_zone([0,0,1,1],[0,0,2,2],2)
    def test_identity_iou(self): self.assertEqual(intersection_over_union([0,0,4,4],[0,0,4,4]),1)
