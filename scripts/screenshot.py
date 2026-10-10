"""Take screenshots of a running CardioTwin dashboard (light, dark, mobile).

    cardiotwin serve --port 8000 &
    python scripts/screenshot.py --url http://127.0.0.1:8000 --out docs/img
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="screenshots")
    ap.add_argument("--demo", default=None, help="demo id to click before capturing")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exe = os.environ.get("CHROMIUM_PATH")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        shots = [("light", "light", {"width": 1360, "height": 900}),
                 ("dark", "dark", {"width": 1360, "height": 900}),
                 ("mobile", "light", {"width": 390, "height": 844})]
        for name, scheme, viewport in shots:
            page = browser.new_page(viewport=viewport, color_scheme=scheme, reduced_motion="reduce",
                                    device_scale_factor=2 if name == "mobile" else 1)
            errors: list[str] = []
            page.on("console", lambda m, errors=errors: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.goto(args.url)
            page.wait_for_selector("#results:not([hidden])", timeout=30000)
            if args.demo:
                page.click(f".demo-btn[data-id='{args.demo}']")
                page.wait_for_timeout(1500)
            page.wait_for_timeout(600)
            page.screenshot(path=str(out / f"dashboard-{name}.png"), full_page=True)
            print(f"{name}: saved; console errors: {errors or 'none'}")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
