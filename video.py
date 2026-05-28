import textwrap
from datetime import datetime
from pathlib import Path
from moviepy import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_audioclips,
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
LOFI_PATH = Path(__file__).parent / "assets" / "lofi_bg.mp3"


# ── Font helper ───────────────────────────────────────────────────────────────

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


# ── Image helpers ─────────────────────────────────────────────────────────────

def _wrap_to_fit(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_px: int) -> list[str]:
    words = text.split()
    lines, current = [], []
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
    src_w, src_h = img.size
    target_ratio = W / H
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    return img.resize((W, H), Image.LANCZOS)


def _fit_to_frame(img: Image.Image, padding: int = 60) -> Image.Image:
    src_w, src_h = img.size
    max_w, max_h = W - padding * 2, H - padding * 2
    scale = min(max_w / src_w, max_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    frame = Image.new("RGB", (W, H), (15, 15, 15))
    frame.paste(img, ((W - new_w) // 2, (H - new_h) // 2))
    return frame


# ── Ken Burns ─────────────────────────────────────────────────────────────────

def _apply_ken_burns(clip: ImageClip, zoom_in: bool = True) -> ImageClip:
    """Slow zoom in or out over the clip duration. Crops to keep frame at W×H."""
    duration = clip.duration
    ZOOM = 1.15

    def scale_fn(t):
        progress = t / duration
        return 1.0 + (ZOOM - 1.0) * progress if zoom_in else ZOOM - (ZOOM - 1.0) * progress

    return (
        clip
        .resized(scale_fn)
        .cropped(x_center=W / 2, y_center=H / 2, width=W, height=H)
    )


# ── Zoom punch ────────────────────────────────────────────────────────────────

def _apply_zoom_punch(clip: ImageClip, zoom_in: bool = True) -> ImageClip:
    """
    150ms punch at clip start (1.0→1.08), then Ken Burns for the rest.
    zoom_in controls the Ken Burns direction after the punch.
    """
    duration = clip.duration
    PUNCH = 1.08
    ZOOM = 1.15
    PUNCH_DUR = 0.15

    def scale_fn(t):
        if t < PUNCH_DUR:
            # Punch: ease out from 1.0 to PUNCH
            progress = t / PUNCH_DUR
            return 1.0 + (PUNCH - 1.0) * progress
        else:
            # Ken Burns: continues from PUNCH toward ZOOM (or back)
            kb_t = t - PUNCH_DUR
            kb_dur = duration - PUNCH_DUR
            if kb_dur <= 0:
                return PUNCH
            progress = kb_t / kb_dur
            if zoom_in:
                return PUNCH + (ZOOM - PUNCH) * progress
            else:
                return PUNCH - (PUNCH - 1.0) * progress

    return (
        clip
        .resized(scale_fn)
        .cropped(x_center=W / 2, y_center=H / 2, width=W, height=H)
    )


# ── Title card ────────────────────────────────────────────────────────────────

def _make_title_card_image(paper_title: str, hook_text: str, bg_image_path: str | None, out_path: str):
    """
    Hook-first title card:
    - Background: first slide's image (darkened) or pure black
    - Hook text large and bold at top-center
    - Paper title smaller below
    - "ML Paper Breakdown" label at bottom
    """
    # Background
    if bg_image_path:
        img = _crop_to_fill(Image.open(bg_image_path).convert("RGB"))
        # Dark gradient overlay
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        for y in range(H):
            alpha = int(180 + 60 * (y / H))  # 180→240 top to bottom
            ov_draw.line([(0, y), (W, y)], fill=(0, 0, 0, alpha))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    else:
        img = Image.new("RGB", (W, H), (0, 0, 0))

    draw = ImageDraw.Draw(img)
    PADDING = 80

    # ── Hook text (large, bold, centered) ────────────────────────────────────
    hook_text = hook_text.strip().rstrip(".")
    for size in [72, 64, 56, 48, 40]:
        font = _load_font(size)
        lines = _wrap_to_fit(draw, hook_text, font, W - PADDING * 2)
        line_h = int(size * 1.3)
        if len(lines) <= 3:
            hook_font, hook_lines, hook_lh = font, lines, line_h
            hook_size = size
            break
    else:
        hook_size = 40
        hook_font = _load_font(hook_size)
        hook_lines = _wrap_to_fit(draw, hook_text, hook_font, W - PADDING * 2)
        hook_lh = int(hook_size * 1.3)

    hook_total_h = len(hook_lines) * hook_lh
    hook_y = H // 2 - hook_total_h // 2 - 80

    for line in hook_lines:
        bbox = draw.textbbox((0, 0), line, font=hook_font)
        lw = bbox[2] - bbox[0]
        x = (W - lw) // 2
        # Shadow
        draw.text((x + 3, hook_y + 3), line, font=hook_font, fill=(0, 0, 0))
        draw.text((x, hook_y), line, font=hook_font, fill=(255, 255, 255))
        hook_y += hook_lh

    # ── Paper title (smaller, below hook) ────────────────────────────────────
    title_y = hook_y + 40
    for size in [36, 30, 26, 22]:
        font = _load_font(size)
        lines = _wrap_to_fit(draw, paper_title, font, W - PADDING * 2)
        line_h = int(size * 1.3)
        if len(lines) <= 3:
            title_font, title_lines, title_lh = font, lines, line_h
            break
    else:
        title_font = _load_font(22)
        title_lines = _wrap_to_fit(draw, paper_title, title_font, W - PADDING * 2)
        title_lh = int(22 * 1.3)

    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = bbox[2] - bbox[0]
        draw.text(((W - lw) // 2, title_y), line, font=title_font, fill=(200, 200, 200))
        title_y += title_lh

    # ── Bottom label ──────────────────────────────────────────────────────────
    label_font = _load_font(26)
    label = "ML PAPER BREAKDOWN"
    lb = draw.textbbox((0, 0), label, font=label_font)
    draw.text(((W - (lb[2] - lb[0])) // 2, H - 100), label, font=label_font, fill=(140, 140, 140))

    img.save(out_path, "PNG")


# ── Slide compositing ─────────────────────────────────────────────────────────

def _make_slide_image(slide: dict, out_path: str):
    """Crop/fit background, overlay bottom bar + counter with PIL."""
    src = Image.open(slide["image_path"]).convert("RGB")
    img = _fit_to_frame(src) if slide.get("role") == "architecture" else _crop_to_fill(src)

    # Semi-transparent bottom bar
    BAR_H = 110
    bar_top = H - BAR_H
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([(0, bar_top), (W, H)], fill=(0, 0, 0, 170))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Slide title
    MAX_LABEL_W = W - 40
    label_text = slide["title"].upper()
    label_font = _load_font(20)
    for size in [38, 32, 26, 22, 18]:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), label_text, font=font)
        if bbox[2] - bbox[0] <= MAX_LABEL_W:
            label_font = font
            break

    label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
    label_w, label_h = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
    label_x = (W - label_w) // 2
    label_y = bar_top + (BAR_H - label_h) // 2
    draw.text((label_x + 2, label_y + 2), label_text, font=label_font, fill=(0, 0, 0))
    draw.text((label_x, label_y), label_text, font=label_font, fill=(255, 255, 255))

    # Slide counter top-right
    counter_font = _load_font(28)
    counter_text = f"{slide['slide_number']} / {TOTAL_SLIDES}"
    cb = draw.textbbox((0, 0), counter_text, font=counter_font)
    cx = W - (cb[2] - cb[0]) - 30
    draw.text((cx + 2, 48), counter_text, font=counter_font, fill=(0, 0, 0))
    draw.text((cx, 46), counter_text, font=counter_font, fill=(255, 255, 255))

    img.save(out_path, "PNG")


# ── Clip builders ─────────────────────────────────────────────────────────────

def _make_title_card(slides: list[dict], out_path: str, duration: float = 2.5) -> ImageClip:
    first_slide = slides[0]
    hook_text = first_slide.get("hook_text", first_slide["narration"][:60])
    bg_path = first_slide.get("image_path")  # may be None if called before media gen
    _make_title_card_image(first_slide.get("title", ""), hook_text, bg_path, out_path)
    clip = ImageClip(out_path, duration=duration)
    return _apply_ken_burns(clip, zoom_in=True)


def _make_slide_clip(slide: dict, composite_path: str) -> CompositeVideoClip:
    duration = slide["duration_seconds"] + 0.5
    _make_slide_image(slide, composite_path)
    base = ImageClip(composite_path, duration=duration)

    is_arch = slide.get("role") == "architecture"

    # Motion: zoom punch + Ken Burns (skip for architecture)
    if not is_arch:
        zoom_in = slide["slide_number"] % 2 == 0
        base = _apply_zoom_punch(base, zoom_in=zoom_in)

    # Captions
    caption_clips = []
    if slide.get("audio_path"):
        try:
            print(f"    🎬 Transcribing slide {slide['slide_number']} for captions...")
            from captions import make_caption_clips, transcribe_slide
            words = transcribe_slide(slide["audio_path"])
            caption_clips = make_caption_clips(words, duration)
        except Exception as e:
            print(f"    ⚠️  Caption generation failed: {e}")

    audio_clip = AudioFileClip(slide["audio_path"])
    layers = [base] + caption_clips
    composite = CompositeVideoClip(layers, size=(W, H))
    return composite.with_audio(audio_clip)


# ── Assembly ──────────────────────────────────────────────────────────────────

def assemble_video(slides: list[dict], paper_metadata: dict, arxiv_id: str) -> str:
    out_dir = Path("output") / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(out_dir / f"final_video_{timestamp}.mp4")

    title_img_path = str(out_dir / "title_card.png")
    print("  Building title card...")
    title_card = _make_title_card(slides, title_img_path)

    print("  Building slide clips...")
    slide_clips = []
    for s in slides:
        composite_path = str(out_dir / f"slide_{s['slide_number']}_composite.png")
        slide_clips.append(_make_slide_clip(s, composite_path))

    print("  Mixing audio...")
    raw_video = concatenate_videoclips([title_card] + slide_clips, method="compose")

    # Lo-fi background music
    if LOFI_PATH.exists():
        try:
            from moviepy.audio.fx import AudioFadeIn, AudioFadeOut, AudioLoop
            music = (
                AudioFileClip(str(LOFI_PATH))
                .with_effects([AudioLoop(duration=raw_video.duration)])
                .with_volume_scaled(0.15)
                .with_effects([AudioFadeIn(1.0), AudioFadeOut(1.5)])
            )
            mixed = CompositeAudioClip([raw_video.audio, music])
            final = raw_video.with_audio(mixed)
            print("  🎵 Lo-fi music mixed in")
        except Exception as e:
            print(f"  ⚠️  Music mixing failed ({e}), continuing without music")
            final = raw_video
    else:
        print(f"  ⚠️  {LOFI_PATH} not found, skipping music")
        final = raw_video

    print("  Concatenating and exporting...")
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
