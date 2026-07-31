"""German Ortsschilder Zeichen 310 / 311."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def draw_ortsschild(img, text, screen_x, ground_y, *, exit_sign=False):
    """StVO Ortstafel: yellow plate, black text; exit = diagonal strike (311)."""
    if not text or screen_x < -120 or screen_x > img.width + 120:
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w = max(120, tw + 36)
    h = max(42, th + 20)
    cx = int(screen_x)
    post_top = ground_y - 52
    plate_top = post_top - h + 6
    draw.rectangle([cx - 3, int(post_top), cx + 3, int(ground_y)], fill=(90, 90, 90, 255))
    x0, y0 = cx - w // 2, int(plate_top)
    yellow, black = (255, 204, 0, 255), (20, 20, 20, 255)
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=yellow, outline=black, width=3)
    draw.rectangle([x0 + 3, y0 + 3, x0 + w - 3, y0 + h - 3], outline=black, width=1)
    draw.text((cx - tw // 2, y0 + (h - th) // 2 - 1), text, fill=black, font=font)
    if exit_sign:
        draw.line([(x0 + 6, y0 + h - 6), (x0 + w - 6, y0 + 6)], fill=black, width=4)
    return Image.alpha_composite(img.convert("RGBA"), layer)
