import os
import base64
import textwrap
import requests
from pathlib import Path
from mutagen.mp3 import MP3
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

oai_client = OpenAI()
eleven_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

# ElevenLabs voice — Rachel (calm, clear, American English)
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
ELEVENLABS_MODEL = "eleven_multilingual_v2"

IMAGE_PROMPT_PREFIX = (
    "Clean, professional diagram illustration style, white background, "
    "no text, no labels, suitable for a vertical Instagram Reel explainer video. "
    "Portrait orientation, bold colours, simple shapes and arrows. "
)


SLIDE_GRADIENTS = [
    ((15, 32, 39),  (44, 83, 100)),
    ((24, 24, 84),  (56, 28, 93)),
    ((11, 59, 11),  (30, 90, 40)),
    ((80, 30, 20),  (130, 60, 30)),
    ((20, 20, 60),  (50, 40, 80)),
]


def _make_gradient_image(slide: dict, path: str, width: int = 1024, height: int = 1792):
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
    # Just show the slide number — clean minimal look
    draw.text((width // 2, height // 2), str(slide["slide_number"]), fill=(255, 255, 255, 80), anchor="mm")
    img.save(path, "PNG")


def _get_mp3_duration(path: str) -> float:
    audio = MP3(path)
    return audio.info.length


def generate_media(slides: list[dict], arxiv_id: str, figures: list[str] | None = None) -> list[dict]:
    out_dir = Path("output") / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for slide in slides:
        n = slide["slide_number"]
        print(f"  Generating media for slide {n}/6: {slide['title']}...")

        # --- ElevenLabs TTS ---
        audio_path = str(out_dir / f"slide_{n}.mp3")
        audio_stream = eleven_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=slide["narration"],
            model_id=ELEVENLABS_MODEL,
        )
        with open(audio_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        slide["audio_path"] = audio_path
        slide["duration_seconds"] = _get_mp3_duration(audio_path)

        # --- Image ---
        image_path = str(out_dir / f"slide_{n}.png")

        # Architecture slide: ONLY use scraped figure or gradient — never generate via AI
        if slide.get("role") == "architecture":
            if figures:
                from figures import best_architecture_figure
                fig_path = best_architecture_figure(figures)
                if fig_path:
                    import shutil
                    shutil.copy(fig_path, image_path)
                    slide["image_path"] = image_path
                    print(f"    ✅ Using scraped paper figure")
                    continue
            # No scraped figure available — gradient fallback (never AI-generated)
            _make_gradient_image(slide, image_path, width=1080, height=1920)
            slide["image_path"] = image_path
            print(f"    ℹ️  No scraped figure available, using gradient fallback")
            continue

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
            print(f"    ⚠️  Image generation failed ({e.__class__.__name__}: {e}), using gradient fallback")
            _make_gradient_image(slide, image_path, width=1080, height=1920)
        slide["image_path"] = image_path

    return slides
