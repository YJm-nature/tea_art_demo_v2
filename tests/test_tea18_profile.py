import unittest

from src.item_matcher import ItemMatcher
from src.model_config import (
    _FALLBACK_CANDIDATES,
    load_profiles,
    profile_override_for_model,
)
from src.tea_detector import DetectedItem


class Tea18ProfileTests(unittest.TestCase):
    def test_current_detector_is_the_first_default_candidate(self):
        self.assertIn(
            "front_detect_selected_holdout_stage1-2",
            _FALLBACK_CANDIDATES[0],
        )
        self.assertEqual(
            profile_override_for_model(_FALLBACK_CANDIDATES[0]),
            "tea18_warm_clean",
        )

    def test_profile_and_alias_preserve_new_classes(self):
        profile = load_profiles()["tea18"]
        self.assertEqual(len(profile.class_names), 18)
        self.assertEqual(profile.active_class_ids, [0, 1, 2, 3, 4, 5, 6, 7, 10, 14])
        self.assertNotIn("电子秤", profile.active_class_names)
        self.assertIn("盖碗", profile.supported_items)
        matcher = ItemMatcher(
            supported_items=profile.scoring_items,
            item_aliases=profile.item_aliases,
        )
        body = DetectedItem(
            bbox=(10, 10, 100, 100), confidence=0.9,
            item_name="盖碗碗身", source="yolo",
        )
        lid = DetectedItem(
            bbox=(20, 20, 80, 30), confidence=0.9,
            item_name="盖碗碗盖", source="yolo",
        )
        matched = matcher.match([body, lid], (720, 1280))
        self.assertEqual([item.item_name for item in matched], ["盖碗", "盖碗碗盖"])
        checklist = matcher.get_checklist(matched)
        self.assertTrue(checklist["盖碗"]["detected"])

    def test_warm_clean_profile_adds_kettle_without_changing_class_ids(self):
        profiles = load_profiles()
        base = profiles["tea18"]
        warm_clean = profiles["tea18_warm_clean"]

        self.assertEqual(warm_clean.class_names, base.class_names)
        self.assertEqual(
            warm_clean.active_class_ids,
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 14],
        )
        self.assertEqual(warm_clean.class_names[9], "烧水壶")

    def test_serve_layout_profile_adds_tray_without_enabling_kettle(self):
        profiles = load_profiles()
        base = profiles["tea18"]
        serve_layout = profiles["tea18_serve_layout"]

        self.assertEqual(serve_layout.class_names, base.class_names)
        self.assertEqual(
            serve_layout.active_class_ids,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14],
        )
        self.assertIn("茶盘", serve_layout.active_class_names)
        self.assertNotIn("烧水壶", serve_layout.active_class_names)


if __name__ == "__main__":
    unittest.main()
