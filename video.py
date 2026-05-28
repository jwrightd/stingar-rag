import textwrap
from datetime import datetime
from pathlib import Path
from moviepy import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont

# Instagram Reels — 9:16 portrait
W, H = 1080, 1920
FONT_PATH = "/System/Library/Fonts/Geneva.ttf"
FONT_FALLBACKS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]
TOTAL_SLIDES = 6


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [FONT_PATH] + FONT_FALLBACKS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap_to_fit(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_px: int) -> list[str]:
    """Word-wrap text so no line exceeds max_px pixels wide."""
    words = text.split()
    lines = []
    current = []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_px:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _crop_to_fill(img: Image.Image) -> Image.Image:
    """Crop and resize to exactly W×H without distortion."""
    src_w, src_h = img.size
    target_ratio = W / H
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        # Too wide — crop sides
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        # Too tall — crop top/bottom
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    return img.resize((W, H), Image.LANCZOS)


def _draw_centered_text(draw: ImageDraw.Draw, lines: list[str], font: ImageFont.FreeTypeFont,
                         start_y: int, line_spacing: int, color: tuple) -> int:
    """Draw lines centered horizontally. Returns y after last line."""
    y = start_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = (W - line_w) // 2
        draw.text((x, y), line, font=font, fill=color)
        y += line_spacing
    return y


def _make_title_card_image(paper_title: str, out_path: str):
    """Render title card entirely with PIL — guaranteed no overflow."""
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── "ML PAPER BREAKDOWN" tag ──────────────────────────────────────────
    tag_font = _load_font(30)
    tag_text = "ML PAPER BREAKDOWN"
    tag_bbox = draw.textbbox((0, 0), tag_text, font=tag_font)
    tag_w = tag_bbox[2] - tag_bbox[0]
    draw.text(((W - tag_w) // 2, 240), tag_text, font=tag_font, fill=(136, 136, 136))

    # Thin divider line
    draw.line([(W // 2 - 80, 310), (W // 2 + 80, 310)], fill=(70, 70, 70), width=2)

    # ── Paper title — auto-size so every line fits in W-120 ──────────────
    PADDING = 120   # 60px each side
    max_w = W - PADDING
    MAX_TITLE_HEIGHT = 700  # leave plenty of room above/below

    title_font = None
    title_lines = []
    title_size = 22

    for size in [56, 48, 40, 34, 28, 22]:
        font = _load_font(size)
        lines = _wrap_to_fit(draw, paper_title, font, max_w)
        line_h = int(size * 1.35)
        total_h = len(lines) * line_h
        if total_h <= MAX_TITLE_HEIGHT:
            title_font = font
            title_lines = lines
            title_size = size
            break

    if title_font is None:
        title_size = 20
        title_font = _load_font(title_size)
        title_lines = _wrap_to_fit(draw, paper_title, title_font, max_w)

    line_h = int(title_size * 1.35)
    total_title_h = len(title_lines) * line_h

    # Vertically center the title block between y=340 and y=1500
    available_center = (340 + 1500) // 2
    title_start_y = available_center - total_title_h // 2

    y = _draw_centered_text(draw, title_lines, title_font, title_start_y, line_h, (255, 255, 255))

    # ── "Explained in 60 seconds" subtitle ───────────────────────────────
    sub_font = _load_font(32)
    sub_text = "Explained in 60 seconds"
    sub_y = min(y + 70, H - 130)
    sub_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
    sub_w = sub_bbox[2] - sub_bbox[0]
    draw.text(((W - sub_w) // 2, sub_y), sub_text, font=sub_font, fill=(170, 170, 170))

    img.save(out_path, "PNG")


def _make_slide_image(slide: dict, out_path: str):
    """Crop background to W×H, then overlay bottom bar + counter with PIL."""
    # Load and fill-crop to exact frame size
    img = _crop_to_fill(Image.open(slide["image_path"]).convert("RGB"))

    # ── Semi-transparent bottom bar via alpha composite ───────────────────
    BAR_H = 110
    bar_top = H - BAR_H

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([(0, bar_top), (W, H)], fill=(0, 0, 0, 170))  # ~67% opacity

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ── Slide title in bar — shrink until it fits ─────────────────────────
    MAX_LABEL_W = W - 40   # 20px margin each side
    label_text = slide["title"].upper()

    label_font = _load_font(20)
    for size in [38, 32, 26, 22, 18]:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), label_text, font=font)
        if bbox[2] - bbox[0] <= MAX_LABEL_W:
            label_font = font
            label_size = size
            break
    else:
        label_size = 18

    label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
    label_w = label_bbox[2] - label_bbox[0]
    label_h = label_bbox[3] - label_bbox[1]
    label_x = (W - label_w) // 2
    label_y = bar_top + (BAR_H - label_h) // 2

    # Subtle shadow for depth
    draw.text((label_x + 2, label_y + 2), label_text, font=label_font, fill=(0, 0, 0))
    draw.text((label_x, label_y), label_text, font=label_font, fill=(255, 255, 255))

    # ── Slide counter — top-right safe zone ──────────────────────────────
    counter_font = _load_font(28)
    counter_text = f"{slide['slide_number']} / {TOTAL_SLIDES}"
    counter_bbox = draw.textbbox((0, 0), counter_text, font=counter_font)
    counter_w = counter_bbox[2] - counter_bbox[0]
    counter_h = counter_bbox[3] - counter_bbox[1]
    cx = W - counter_w - 30   # 30px from right edge
    cy = 46

    # Shadow + text
    draw.text((cx + 2, cy + 2), counter_text, font=counter_font, fill=(0, 0, 0))
    draw.text((cx, cy), counter_text, font=counter_font, fill=(255, 255, 255))

    img.save(out_path, "PNG")


def _make_title_card(paper_title: str, out_path: str, duration: float = 2.5) -> ImageClip:
    _make_title_card_image(paper_title, out_path)
    return ImageClip(out_path, duration=duration)


def _make_slide_clip(slide: dict, composite_path: str) -> ImageClip:
    duration = slide["duration_seconds"] + 0.5
    _make_slide_image(slide, composite_path)
    image_clip = ImageClip(composite_path, duration=duration)
    audio_clip = AudioFileClip(slide["audio_path"])
    return image_clip.with_audio(audio_clip)


def assemble_video(slides: list[dict], paper_metadata: dict, arxiv_id: str) -> str:
    out_dir = Path("output") / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(out_dir / f"final_video_{timestamp}.mp4")

    title_img_path = str(out_dir / "title_card.png")
    print("  Building title card...")
    title_card = _make_title_card(paper_metadata["title"], title_img_path)

    print("  Building slide clips...")
    slide_clips = []
    for s in slides:
        composite_path = str(out_dir / f"slide_{s['slide_number']}_composite.png")
        slide_clips.append(_make_slide_clip(s, composite_path))

    print("  Concatenating and exporting...")
    final = concatenate_videoclips([title_card] + slide_clips, method="compose")
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        logger=None,
    )

    total = sum(s["duration_seconds"] for s in slides) + 2.5
    print(f"  Total duration: {total:.1f}s")
    return output_path
