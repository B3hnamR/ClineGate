"""Build the tool's icon from the official Cline app assets.

Source of truth: %LOCALAPPDATA%/Cline/icons/app/*.png — the icon files shipped
by the official Cline installer. classic.png is the standard logo. The results
are committed under gateway/assets/, so rebuilding never requires Cline to be
installed.

    python tools/make_icon.py            # write assets/cline.png + clie.ico
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

GATEWAY = Path(__file__).resolve().parent.parent
ASSETS = GATEWAY / "assets"

CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Cline" / "icons" / "app" / "classic.png",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Cline" / "icons" / "app" / "chip.png",
]

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
             (128, 128), (256, 256)]


def main() -> int:
    src = next((p for p in CANDIDATES if p.is_file()), None)
    if src is None:
        print("official Cline icon not found — install Cline or drop a PNG "
              "into assets/", file=sys.stderr)
        return 1

    im = Image.open(src).convert("RGBA")
    print(f"source: {src}  {im.size}")

    ASSETS.mkdir(exist_ok=True)

    png = ASSETS / "cline.png"
    im.save(png)
    print(f"wrote {png}")

    # multi-resolution .ico; 148px source downscales crisply to 128 and below
    im.save(ASSETS / "cline.ico", format="ICO", sizes=ICO_SIZES)
    print(f"wrote {ASSETS / 'cline.ico'}  sizes={ICO_SIZES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
