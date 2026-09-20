"""Record the browser demo (luce serve → /demo) as a video with Playwright, then convert with ffmpeg.

    python scripts/record_demo.py --url http://localhost:8000/demo --out media/live_triage
Produces media/live_triage.webm (raw), .mp4 and .gif.
"""
import argparse
import asyncio
import os
import shutil
import subprocess

TYPED = [
    "Charged twice for order #4471",
    "Two charges for one order. This is unacceptable, fix it today or I dispute the card.",
]
TYPED_CALM = "Hi, I noticed two charges for one order. Could you take a look when you have a moment? Thanks."


async def main(url: str, out: str, feed_seconds: float) -> None:
    from playwright.async_api import async_playwright
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmpdir = out + "_video"
    shutil.rmtree(tmpdir, ignore_errors=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800}, record_video_dir=tmpdir,
                                        record_video_size={"width": 1280, "height": 800}, device_scale_factor=1)
        page = await ctx.new_page()
        await page.goto(url)
        await page.wait_for_selector("#start")
        await page.wait_for_timeout(1200)
        await page.click("#start")                       # inbox feed
        await page.wait_for_timeout(feed_seconds * 1000)
        await page.click("#start")                       # pause feed, now type
        box = page.locator("#typed")
        await box.click()
        await box.type(TYPED[0] + "\n", delay=45)
        await box.type(TYPED[1], delay=28)
        await page.wait_for_timeout(2500)
        await box.fill("")
        await box.type(TYPED[0] + "\n", delay=30)
        await box.type(TYPED_CALM, delay=28)
        await page.wait_for_timeout(3000)
        await ctx.close()
        await browser.close()
    webm = next(os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".webm"))
    shutil.move(webm, out + ".webm"); shutil.rmtree(tmpdir, ignore_errors=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".webm", "-vf", "fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out + ".mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".mp4", "-vf", "fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", out + ".gif"], check=True)
    for ext in (".webm", ".mp4", ".gif"):
        print(out + ext, os.path.getsize(out + ext) // 1024, "KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/demo")
    ap.add_argument("--out", default="media/live_triage")
    ap.add_argument("--feed-seconds", type=float, default=40)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.out, a.feed_seconds))
