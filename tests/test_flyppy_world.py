from __future__ import annotations

from pathlib import Path
import sys
import unittest


EMBODIMENT_DIR = Path(__file__).resolve().parents[1] / "scripts/embodiment"
sys.path.insert(0, str(EMBODIMENT_DIR))

from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld


class FlyppyWorldTests(unittest.TestCase):
    def test_v6_preserves_open_lateral_history(self) -> None:
        world = FlyppyWorld(FlyppyCourse(seed=7, gate_count=6, environment_version="v6"))
        names = {geom.name for geom in world.obstacle_geoms}
        self.assertNotIn("flyppy_side_left", names)
        self.assertNotIn("flyppy_side_right", names)

    def test_v7_closes_lateral_corridor(self) -> None:
        course = FlyppyCourse(seed=7, gate_count=6, environment_version="v7")
        world = FlyppyWorld(course)
        side_geoms = {
            geom.name: geom
            for geom in world.obstacle_geoms
            if geom.name.startswith("flyppy_side_")
        }
        self.assertEqual(set(side_geoms), {"flyppy_side_left", "flyppy_side_right"})

        left = side_geoms["flyppy_side_left"]
        right = side_geoms["flyppy_side_right"]
        half_thickness = world.wall_half_thickness_mm
        self.assertAlmostEqual(float(left.pos[1]) + half_thickness, -course.lateral_half_width_mm)
        self.assertAlmostEqual(float(right.pos[1]) - half_thickness, course.lateral_half_width_mm)

    def test_side_contact_is_terminal_course_collision(self) -> None:
        course = FlyppyCourse(seed=7, gate_count=1, environment_version="v7")
        event = course.update(
            0.0,
            10.0,
            physical_collision_reason="side",
            analytic_body_collision=False,
        )
        self.assertTrue(event.collision)
        self.assertEqual(event.collision_reason, "side")
        self.assertFalse(event.passed_gate)


if __name__ == "__main__":
    unittest.main()
