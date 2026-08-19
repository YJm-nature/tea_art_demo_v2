from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from src.draw_utils import (
    _format_observation_snapshot,
    _measure_text,
    _resolve_font_path,
    draw_controls,
    draw_detections,
    draw_sop_panel,
    draw_step_details_panel,
)
from src.observation_runtime import ObservationSnapshot, ObservationState


class DrawUtilsTests(unittest.TestCase):
    @staticmethod
    def _sop_state():
        step = {
            "step_id": "setup",
            "name": "茶具齐全且人员端坐",
            "observation_id": "action_setup_ready",
            "business_step": "step01_setup",
            "business_step_name": "备具布席",
            "action_flow": ["检测全部茶具", "检测人员端坐", "完成备具"],
            "requirements": ["茶具全部检出", "人员在桌前端坐"],
        }
        return {
            "mode": "strict",
            "status": "running",
            "current_step_id": "setup",
            "steps": [step],
            "runtime": {"setup": {"status": "active"}},
        }

    def test_sop_and_action_panels_use_opposite_top_corners(self):
        frame = np.full((720, 1280, 3), 210, dtype=np.uint8)
        state = self._sop_state()
        snapshot = ObservationSnapshot(
            observation_id="action_setup_ready",
            name="茶具齐全且人员端坐",
            sop_step=1,
            state=ObservationState.IDLE,
            reason="缺少：品茗杯、烧水壶",
        )
        sop_only = draw_sop_panel(frame, state, {snapshot.observation_id: snapshot})
        self.assertFalse(np.array_equal(sop_only[15:175, 15:615], frame[15:175, 15:615]))
        self.assertTrue(np.array_equal(sop_only[15:175, 655:1265], frame[15:175, 655:1265]))

        details_only = draw_step_details_panel(
            frame, state, {snapshot.observation_id: snapshot}
        )
        self.assertTrue(np.array_equal(details_only[15:175, 15:615], frame[15:175, 15:615]))
        self.assertFalse(np.array_equal(details_only[15:175, 655:1265], frame[15:175, 655:1265]))

    def test_observation_panel_text_includes_layout_value(self):
        snapshot = ObservationSnapshot(
            observation_id="result_cup_layout",
            name="品茗杯布局",
            sop_step=6,
            state=ObservationState.COMPLETED,
            confidence=0.96,
            value="品字形",
        )
        text = _format_observation_snapshot(snapshot)
        self.assertIn("品茗杯布局", text)
        self.assertIn("已完成", text)
        self.assertIn("品字形", text)

    def test_observation_panel_text_shows_irregular_layout(self):
        snapshot = ObservationSnapshot(
            observation_id="result_cup_layout",
            name="品茗杯布局",
            sop_step=6,
            state=ObservationState.IDLE,
            confidence=0.88,
            value="其他布局",
        )
        text = _format_observation_snapshot(snapshot)
        self.assertIn("品茗杯布局", text)
        self.assertIn("不规则", text)
        self.assertNotIn("一字形", text)
        self.assertNotIn("品字形", text)

    def test_windows_cjk_font_is_available(self):
        font_path = _resolve_font_path()
        self.assertIsNotNone(font_path)
        self.assertTrue(Path(font_path).is_file())
        self.assertGreater(_measure_text("盖碗碗身", 0.45)[0], 0)

    def test_chinese_detection_and_control_labels_render(self):
        frame = np.full((240, 480, 3), 235, dtype=np.uint8)
        item = SimpleNamespace(
            bbox=(40, 70, 120, 100),
            item_name="盖碗碗身",
            confidence=0.91,
            track_id=12,
        )
        rendered = draw_detections(frame, [item])
        rendered = draw_controls(rendered, conf=0.15)
        self.assertFalse(np.array_equal(rendered, frame))
        self.assertGreater(np.count_nonzero(rendered[-34:] != frame[-34:]), 100)


if __name__ == "__main__":
    unittest.main()
