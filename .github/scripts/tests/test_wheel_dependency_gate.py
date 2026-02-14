import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wheel_dependency_gate  # noqa: E402


class WheelDependencyGateTests(unittest.TestCase):
    def test_parse_libraries_from_auditwheel_output(self):
        report = """
        cpd_rs-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl is consistent with the
        following platform tag: "manylinux_2_28_x86_64".
        The wheel references external versioned symbols in these
        system-provided shared libraries: libgcc_s.so.1, libc.so.6, libm.so.6
        """
        libraries = wheel_dependency_gate.parse_libraries(report)
        self.assertIn("libgcc_s.so.1", libraries)
        self.assertIn("libc.so.6", libraries)
        self.assertIn("libm.so.6", libraries)
        self.assertNotIn("manylinux_2_28_x86_64", libraries)

    def test_parse_libraries_from_delocate_output(self):
        report = """
        /tmp/wheelhouse/cpd_rs.whl:
            @rpath/libSystem.B.dylib
            @rpath/libobjc.A.dylib
        """
        libraries = wheel_dependency_gate.parse_libraries(report)
        self.assertIn("libsystem.b.dylib", libraries)
        self.assertIn("libobjc.a.dylib", libraries)

    def test_parse_libraries_from_framework_style_output(self):
        report = """
        /tmp/wheelhouse/cpd_rs.whl:
            /System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate
            /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
        """
        libraries = wheel_dependency_gate.parse_libraries(report)
        self.assertIn("accelerate.framework", libraries)
        self.assertIn("corefoundation.framework", libraries)

    def test_parse_libraries_from_delvewheel_output(self):
        report = """
        cpd_rs-0.1.0-cp312-cp312-win_amd64.whl
            dependencies:
              VCRUNTIME140.dll
              api-ms-win-crt-runtime-l1-1-0.dll
        """
        libraries = wheel_dependency_gate.parse_libraries(report)
        self.assertIn("vcruntime140.dll", libraries)
        self.assertIn("api-ms-win-crt-runtime-l1-1-0.dll", libraries)

    def test_find_blocked_libraries_detects_openblas(self):
        libraries = {"libc.so.6", "libopenblas.so.0", "libgcc_s.so.1"}
        blocked = wheel_dependency_gate.find_blocked_libraries(libraries)
        self.assertEqual(blocked, ["libopenblas.so.0"])

    def test_find_blocked_libraries_detects_lapack(self):
        libraries = {"mylapack.dll", "vcruntime140.dll"}
        blocked = wheel_dependency_gate.find_blocked_libraries(libraries)
        self.assertEqual(blocked, ["mylapack.dll"])

    def test_find_blocked_libraries_detects_accelerate_framework(self):
        libraries = {"accelerate.framework", "corefoundation.framework"}
        blocked = wheel_dependency_gate.find_blocked_libraries(libraries)
        self.assertEqual(blocked, ["accelerate.framework"])

    def test_contains_manylinux_tag(self):
        self.assertTrue(
            wheel_dependency_gate.contains_manylinux_tag(
                "platform tag: manylinux_2_28_x86_64"
            )
        )
        self.assertFalse(
            wheel_dependency_gate.contains_manylinux_tag(
                "platform tag: linux_x86_64"
            )
        )


if __name__ == "__main__":
    unittest.main()
