"""
Word-by-word animated captions using Whisper for timestamp alignment.
Renders Hormozi-style: 4-word sliding window, current word highlighted in yellow.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip

# Lazy-load the Whisper model once per process
_whisper_model = None

W, H = 1080, 1920
CAPTION_Y = H - 230        # above the bottom title bar
CAPTION_FONT_SIZE = 56
WINDOW_SIZE = 4             # words visible at once
HIGHLIGHT_COLOR = (255, 220, 0)    # yellow
HIGHLIGHT_TEXT_COLOR = (0, 0, 0)   # black text on yellow
DIM_ALPHA = 180                    # opacity for non-current words (0–255)

FONT_PATH = "/System/Library/Fonts/Geneva.ttf"
FONT_FALLBACKS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


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


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper_timestamped as whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_slide(audio_path: str) -> list[dict]:
    """
    Return word-level timestamps for an audio file.
    Each entry: {"word": str, "start": float, "end": float}
    """
    import whisper_timestamped as whisper
    model = _get_whisper_model()
    audio = whisper.load_audio(audio_path)
    result = whisper.transcribe(model, audio, language="en", verbose=False)
    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            text = w["text"].strip()
            if text:
                words.append({"word": text, "start": w["start"], "end": w["end"]})
    return words


def _render_caption_frame(window: list[str], current_idx: int) -> np.ndarray:
    """
    Render one caption frame as an RGBA numpy array (transparent background).
    window: list of up to WINDOW_SIZE words
    current_idx: index of the highlighted word within window
    """
    font = _load_font(CAPTION_FONT_SIZE)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Measure each word
    PAD_X, PAD_Y = 18, 8
    word_sizes = []
    for word in window:
        bbox = draw.textbbox((0, 0), word, font=font)
        word_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))

    WORD_GAP = 14
    total_width = sum(w for w, _ in word_sizes) + WORD_GAP * (len(window) - 1)
    # If too wide, shrink font and re-measure
    if total_width > W - 80:
        font = _load_font(CAPTION_FONT_SIZE - 10)
        word_sizes = []
        for word in window:
            bbox = draw.textbbox((0, 0), word, font=font)
            word_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        total_width = sum(w for w, _ in word_sizes) + WORD_GAP * (len(window) - 1)

    x = (W - total_width) // 2
    line_h = max(h for _, h in word_sizes) if word_sizes else CAPTION_FONT_SIZE

    for i, (word, (ww, wh)) in enumerate(zip(window, word_sizes)):
        is_current = (i == current_idx)

        if is_current:
            # Yellow pill background
            rx0 = x - PAD_X
            ry0 = CAPTION_Y - PAD_Y
            rx1 = x + ww + PAD_X
            ry1 = CAPTION_Y + line_h + PAD_Y
            radius = (ry1 - ry0) // 2
            draw.rounded_rectangle([rx0, ry0, rx1, ry1], radius=radius, fill=HIGHLIGHT_COLOR)
            # Black text on yellow
            draw.text((x, CAPTION_Y), word, font=font, fill=HIGHLIGHT_TEXT_COLOR)
        else:
            # White text with black stroke, dimmed
            alpha = DIM_ALPHA
            stroke_col = (0, 0, 0, alpha)
            text_col = (255, 255, 255, alpha)
            # Stroke
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                draw.text((x + dx, CAPTION_Y + dy), word, font=font, fill=stroke_col)
            draw.text((x, CAPTION_Y), word, font=font, fill=text_col)

        x += ww + WORD_GAP

    return np.array(img)


def make_caption_clips(words: list[dict], slide_duration: float) -> list[ImageClip]:
    """
    Build a list of RGBA ImageClips — one per word — to composite over a slide.
    Each clip shows a 4-word window with the current word highlighted.
    """
    if not words:
        return []

    clips = []
    for i, word in enumerate(words):
        # Build the window: try to keep current word at index 1 (second position)
        win_start = max(0, i - 1)
        win_end = min(len(words), win_start + WINDOW_SIZE)
        win_start = max(0, win_end - WINDOW_SIZE)

        window = [w["word"] for w in words[win_start:win_end]]
        current_idx = i - win_start

        frame = _render_caption_frame(window, current_idx)

        t_start = word["start"]
        t_end = min(word["end"], slide_duration)
        if t_end <= t_start:
            continue

        clip = (
            ImageClip(frame, duration=t_end - t_start)
            .with_start(t_start)
        )
        clips.append(clip)

    return clips
