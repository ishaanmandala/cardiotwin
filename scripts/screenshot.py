"""Take screenshots of a running CardioTwin dashboard: every tab in light mode, plus dark and mobile.

    cardiotwin serve --no-browser &
    python scripts/screenshot.py --url http://127.0.0.1:8000 --out docs/img
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

TABS = ["overview", "heart", "ecg", "future", "about"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="screenshots")
    ap.add_argument("--demo", default=None, help="demo id to click before capturing")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exe = os.environ.get("CHROMIUM_PATH")
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()

        def page_for(scheme: str, viewport: dict, scale: int = 1):
            page = browser.new_page(viewport=viewport, color_scheme=scheme, reduced_motion="reduce",
                                    device_scale_factor=scale)
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(args.url)
            page.wait_for_selector("#results:not([hidden])", timeout=30000)
            if args.demo:
                page.click(f".person[data-id='{args.demo}']")
            page.wait_for_timeout(1200)
            return page

        page = page_for("light", {"width": 1360, "height": 900})
        for tab in TABS:
            page.click(f"#tab-{tab}")
            page.wait_for_timeout(700)
            page.screenshot(path=str(out / f"dashboard-{tab}.png"), full_page=True)
            print(f"light/{tab}: saved")
        page.click("#medical-toggle + .switch-track")
        page.click("#tab-heart")
        page.wait_for_timeout(600)
        page.screenshot(path=str(out / "dashboard-heart-medical.png"), full_page=True)
        page.close()

        page = page_for("dark", {"width": 1360, "height": 900})
        page.screenshot(path=str(out / "dashboard-dark.png"), full_page=True)
        page.click("#tab-heart")
        page.wait_for_timeout(600)
        page.screenshot(path=str(out / "dashboard-dark-heart.png"), full_page=True)
        page.close()

        page = page_for("light", {"width": 390, "height": 844}, scale=2)
        page.screenshot(path=str(out / "dashboard-mobile.png"), full_page=True)
        page.close()
        browser.close()
    print("console errors:", errors or "none")


if __name__ == "__main__":
    main()
