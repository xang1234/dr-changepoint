import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import schema_gate  # noqa: E402


class SchemaGateTests(unittest.TestCase):
    def test_validate_repo_passes_for_workspace(self):
        errors = schema_gate.validate_repo(schema_gate.REPO_ROOT)
        self.assertEqual(errors, [])

    def test_config_fixture_requires_schema_version_marker(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            schema_gate.validate_config_fixture(
                {
                    "kind": "pipeline_spec",
                    "payload": {},
                }
            )

    def test_config_fixture_rejects_wrong_schema_version(self):
        with self.assertRaisesRegex(ValueError, "schema_version must be 0"):
            schema_gate.validate_config_fixture(
                {
                    "schema_version": 1,
                    "kind": "pipeline_spec",
                    "payload": {},
                }
            )

    def test_checkpoint_fixture_rejects_bad_crc(self):
        with self.assertRaisesRegex(ValueError, "payload_crc32"):
            schema_gate.validate_checkpoint_fixture(
                {
                    "schema_version": 0,
                    "detector_id": "bocpd",
                    "engine_version": "0.1.0",
                    "created_at_ns": 1,
                    "payload_codec": "json",
                    "payload_crc32": "DEADBEEF",
                    "payload": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
