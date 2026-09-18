from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from virtual_fly.reproducibility import (
    HALTERE_FULL_KIND,
    HALTERE_TIMING_KIND,
    MALECNS_RUNTIME_SEMANTICS,
    MODULATOR_ROLE_DEFINITION,
    validate_haltere_map,
    validate_production_snapshot,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReproducibilityTests(unittest.TestCase):
    def test_production_snapshot_fails_closed_without_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps({"dataset": "male-cns:v1.0", "source_sha256": {"x": "y"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "runtime semantics"):
                validate_production_snapshot(root)

    def test_production_snapshot_accepts_hashed_class_dan_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = root / "modulator_roles.u8"
            role.write_bytes(b"\x01\x00")
            generated = {"modulator_roles.u8": _sha(role)}
            manifest = {
                "format_version": 1,
                "dataset": "male-cns:v1.0",
                "runtime_semantics": MALECNS_RUNTIME_SEMANTICS,
                "generator_provenance": {"generator_git": {"git_sha": "abc"}},
                "modulator_roles_file": "modulator_roles.u8",
                "modulator_role_definition": MODULATOR_ROLE_DEFINITION,
                "source_sha256": {"annotations": "abc"},
                "generated_sha256": generated,
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_production_snapshot(root)
            self.assertEqual(result["runtime_semantics"], MALECNS_RUNTIME_SEMANTICS)

    def test_haltere_map_kinds_are_not_interchangeable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": HALTERE_FULL_KIND,
                        "count": 195,
                        "body_ids_by_side": {"left": [1], "right": [2]},
                        "provenance": {
                            "generator_git": {"git_sha": "x"},
                            "annotations_sha256": "a",
                        },
                    }
                ),
                encoding="utf-8",
            )
            validate_haltere_map(path, HALTERE_FULL_KIND)
            with self.assertRaisesRegex(RuntimeError, "kind mismatch"):
                validate_haltere_map(path, HALTERE_TIMING_KIND)


if __name__ == "__main__":
    unittest.main()
