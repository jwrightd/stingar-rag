"""
Media generation for the dialogue pipeline.
Generates interleaved Stewie/Peter audio per slide, then images same as main pipeline.
"""
import os
import base64
import numpy as np
from pathlib import Path
from mutagen.mp3 import MP3
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from PIL import Image, ImageDraw
from moviepy import AudioFileClip, concatenate_audioclips
from moviepy.audio.AudioClip import AudioArrayClip
from dotenv import load_dotenv

load_dotenv()

oai_client = OpenAI()
eleven_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

VOICES = {
    "stewie": "a9wCVWyWIR4tjtYHFEtD",
    "peter":  "CW34A5g0wxYU7JvOgFQl",
}
ELEVENLABS_MODEL = "eleven_multilingual_v2"
LINE_PAUSE_SECONDS = 0.35   # gap between dialogue lines

IMAGE_PROMPT_PREFIX = (
    "Clean, professional diagram illustration style, white background, "
    "no text, no labels, suitable for a vertical Instagram Reel explainer video. "
    "Portrait orientation, bold colours, simple shapes and arrows. "
)

SLIDE_GRADIENTS = [
    ((15, 32, 39),   (44, 83, 100)),
    ((24, 24, 84),   (56, 28, 93)),
    ((11, 59, 11),   (30, 90, 40)),
    ((80, 30, 20),   (130, 60, 30)),
    ((20, 20, 60),   (50, 40, 80)),
]


def _make_gradient_image(slide: dict, path: str, width: int = 1080, height: int = 1920):
    idx = (slide["slide_number"] - 1) % len(SLIDE_GRADIENTS)
    top_color, bottom_color = SLIDE_GRADIENTS[idx]
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / height
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    img.save(path, "PNG")


def _silence_clip(duration: float, fps: int = 44100) -> AudioArrayClip:
    """Return a silent AudioArrayClip of the given duration."""
    frames = np.zeros((max(1, int(duration * fps)), 2), dtype=np.float32)
    return AudioArrayClip(frames, fps=fps)


def _generate_dialogue_audio(slide: dict, out_dir: Path) -> tuple[str, float]:
    """
    Generate one MP3 per dialogue line, then stitch them together
    with a short pause between each line.
    Returns (combined_audio_path, total_duration_seconds).
    """
    line_paths = []
    for i, line in enumerate(slide["lines"]):
        speaker = line["speaker"].lower()
        voice_id = VOICES.get(speaker, VOICES["stewie"])
        line_path = str(out_dir / f"slide_{slide['slide_number']}_line_{i}.mp3")

        audio_stream = eleven_client.text_to_speech.convert(
            voice_id=voice_id,
            text=line["text"],
            model_id=ELEVENLABS_MODEL,
        )
        with open(line_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        line_paths.append(line_path)
        print(f"    🎙️  {speaker.capitalize()}: \"{line['text'][:50]}...\"" if len(line["text"]) > 50
              else f"    🎙️  {speaker.capitalize()}: \"{line['text']}\"")

    # Stitch: clip, pause, clip, pause, ...
    clips = []
    for i, path in enumerate(line_paths):
        clips.append(AudioFileClip(path))
        if i < len(line_paths) - 1:
            clips.append(_silence_clip(LINE_PAUSE_SECONDS))

    combined = concatenate_audioclips(clips)
    combined_path = str(out_dir / f"slide_{slide['slide_number']}.mp3")
    combined.write_audiofile(combined_path, logger=None)

    # Clean up per-line files
    for path in line_paths:
        try:
            os.unlink(path)
        except Exception:
            pass

    duration = MP3(combined_path).info.length
    return combined_path, duration


def generate_dialogue_media(slides: list[dict], arxiv_id: str, figures: list[str] | None = None) -> list[dict]:
    """
    Generate audio (interleaved Stewie/Peter) and images for each slide.
    Mutates and returns the slides list with audio_path, image_path, duration_seconds filled in.
    """
    out_dir = Path("output") / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for slide in slides:
        n = slide["slide_number"]
        print(f"  Generating media for dialogue slide {n}/6: {slide['title']}...")

        # ── Audio: interleaved Stewie + Peter ────────────────────────────────
        audio_path, duration = _generate_dialogue_audio(slide, out_dir)
        slide["audio_path"] = audio_path
        slide["duration_seconds"] = duration

        # ── Image ────────────────────────────────────────────────────────────
        image_path = str(out_dir / f"slide_{n}.png")

        # Architecture slide: scraped figure preferred, else generate
        if slide.get("role") == "architecture":
            from figures import best_architecture_figure
            import shutil
            fig_path = best_architecture_figure(figures) if figures else None
            if fig_path:
                shutil.copy(fig_path, image_path)
                slide["image_path"] = image_path
                print(f"    ✅ Using scraped paper figure")
                continue
            print(f"    ℹ️  No scraped figure — generating with gpt-image-1")

        try:
            full_prompt = IMAGE_PROMPT_PREFIX + slide["image_prompt"]
            image_response = oai_client.images.generate(
                model="gpt-image-1",
                prompt=full_prompt,
                size="1024x1536",
                quality="medium",
            )
            img_b64 = image_response.data[0].b64_json
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            print(f"    ✅ Image generated")
        except Exception as e:
            print(f"    ⚠️  Image generation failed ({e.__class__.__name__}: {e}), using gradient")
            _make_gradient_image(slide, image_path)

        slide["image_path"] = image_path

    return slides
