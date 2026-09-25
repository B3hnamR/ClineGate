"""The app version, the exe file properties and the release tag must agree.

The updater compares GitHub release tags with `cline_gateway.__version__`, but
the exe's Details tab used to be stamped from a hand-kept file that said
`1.3.2.0` while the app said `0.3.2` — a mismatch that produces either
perpetual update prompts or releases that install the same code. This test
pins the two sources together; the release workflow additionally refuses a tag
that does not match `__version__`.
"""

from __future__ import annotations

import re
from pathlib import Path

from cline_gateway import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


def _expected_quad() -> tuple[int, ...]:
    parts = tuple(int(p) for p in __version__.split("."))
    return parts + (0,) * (4 - len(parts))


def test_exe_version_file_matches_the_app_version():
    text = (REPO_ROOT / "tools" / "exe_version.txt").read_text(encoding="utf-8")
    match = re.search(r"filevers=\(([^)]+)\)", text)
    assert match, "exe_version.txt has no filevers tuple"
    nums = tuple(int(p.strip()) for p in match.group(1).split(","))
    assert nums == _expected_quad(), (
        f"exe_version.txt filevers {nums} != __version__ {__version__} "
        f"({_expected_quad()})")


def test_exe_version_strings_match_the_app_version():
    text = (REPO_ROOT / "tools" / "exe_version.txt").read_text(encoding="utf-8")
    expected = ".".join(str(n) for n in _expected_quad())
    for field in ("FileVersion", "ProductVersion"):
        assert f"StringStruct('{field}', '{expected}')" in text, (
            f"exe_version.txt {field} is not {expected}")
