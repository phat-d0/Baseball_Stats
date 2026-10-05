"""Draw the app icons (web/icons/*.png). Run once: python scripts/make_icons.py"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "web" / "icons"
BLUE, WHITE, RED = (42, 120, 214), (255, 255, 255), (227, 73, 72)


def draw(size: int) -> Image.Image:
    s = 4  # supersample for smooth edges
    W = 512 * s
    img = Image.new("RGB", (W, W), BLUE)
    d = ImageDraw.Draw(img)
    c, r = W // 2, 150 * s
    d.ellipse([c - r, c - r, c + r, c + r], fill=WHITE)
    for sign in (-1, 1):
        # Seam: an arc of a large circle, clipped to the ball.
        cx = c + sign * 330 * s
        R = 270 * s
        d.arc([cx - R, c - R, cx + R, c + R], 0, 360, fill=RED, width=12 * s)
        # Stitches across the seam.
        for dy in (-75, -38, 0, 38, 75):
            y = c + dy * s
            x = cx - sign * (R ** 2 - (y - c) ** 2) ** 0.5
            d.line([x - 16 * s, y - 6 * s * sign, x + 16 * s, y + 6 * s * sign], fill=RED, width=9 * s)
    # Clip anything outside the ball back to blue.
    mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mask).ellipse([c - r, c - r, c + r, c + r], fill=255)
    img = Image.composite(img, Image.new("RGB", (W, W), BLUE), mask)
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    for name, size in (("icon-512.png", 512), ("icon-192.png", 192), ("apple-touch-icon.png", 180)):
        draw(size).save(OUT / name)
