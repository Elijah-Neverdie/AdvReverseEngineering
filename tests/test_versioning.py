"""版本解析与比较单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from AdvReverseEngineering.utils.versioning import (
    compare_versions,
    format_version,
    parse_bl_info_version,
    read_version_file,
)


class VersioningTests(unittest.TestCase):
    def test_parse_and_format(self) -> None:
        source = 'bl_info = {"version": (0, 6, 18), "name": "x"}'
        self.assertEqual(parse_bl_info_version(source), (0, 6, 18))
        self.assertEqual(format_version((0, 6, 18)), "0.6.18")

    def test_compare_versions(self) -> None:
        self.assertEqual(compare_versions((0, 6, 16), (0, 6, 18)), "AVAILABLE")
        self.assertEqual(compare_versions((0, 6, 18), (0, 6, 18)), "CURRENT")
        self.assertEqual(compare_versions((0, 6, 18), (0, 6, 16)), "AHEAD")

    def test_read_version_file(self) -> None:
        text = (
            "bl_info = {\n"
            '    "name": "AdvReverseEngineering",\n'
            '    "version": (1, 2, 3),\n'
            "}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "__init__.py"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(read_version_file(path), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
