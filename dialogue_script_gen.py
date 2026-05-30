"""
Dialogue script generator — Stewie Griffin explains the paper to Peter Griffin.
Produces the same 6-slide structure as script_gen.py but with back-and-forth
dialogue lines instead of single-voice narration.
"""
import json
from openai import OpenAI
from rag import retrieve
from script_gen import SLIDE_QUERIES, generate_hook_text
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

DIALOGUE_SYSTEM_PROMPT = """\
You write scripts for a 60-second Instagram Reel where Stewie Griffin explains \
an ML paper to Peter Griffin (Family Guy).

STEWIE GRIFFIN: Diabolical genius baby. Perfectly understands machine learning. \
Condescending, uses real technical terms correctly, frequently exasperated by Peter. \
Speaks in full sentences with a slight British flair. Occasionally sighs or mutters \
"Good God, man" when Peter is especially dense.

PETER GRIFFIN: Loveable idiot. Zero tech knowledge. Compares everything to food, \
TV, beer, or mundane life. Signature reactions: "Oh wow", "hehehe", \
"That's like when I...", "Is that like...", "Wait so...". Sometimes asks a \
surprisingly insightful follow-up by accident.

Rules:
- This is a FAST ~40-second Reel. Keep every exchange SNAPPY.
- Stewie speaks FIRST and LAST on every slide.
- 3 lines is the default (Stewie, Peter, Stewie). Allow a 4th line ONLY when \
  Peter's confusion genuinely adds a laugh — never to pad.
- Keep lines SHORT — one punchy sentence each. No monologues.
- Stewie's lines must be technically accurate to the paper.
- Peter's lines are funny and clean — no swearing, nothing offensive.
- MAX 38 words total across ALL lines on a slide. Cut ruthlessly.
- Return JSON only: {"lines": [{"speaker": "stewie", "text": "..."}, ...]}
"""

DIALOGUE_ROLE_INSTRUCTIONS = {
    "hook": (
        "Stewie opens by dramatically stating the problem the paper solves. "
        "Peter misunderstands it as something completely mundane. "
        "Stewie is exasperated but intrigued."
    ),
    "idea": (
        "Stewie introduces the paper by its FULL TITLE in his first sentence. "
        "Peter asks a hilariously dumb question about it. "
        "Stewie corrects him with the core idea."
    ),
    "method": (
        "Stewie explains the key technical mechanism — name real components. "
        "Peter compares it to something absurd (food, TV, sports). "
        "Stewie grudgingly admits the analogy is sort of correct."
    ),
    "results": (
        "Stewie announces the key benchmark number or result. "
        "Peter reacts with disproportionate excitement for the wrong reason. "
        "Stewie clarifies why it actually matters."
    ),
    "takeaway": (
        "Stewie explains why this work matters and what it unlocks. "
        "Peter asks what this means for him personally. "
        "Stewie delivers a memorable closing line."
    ),
    "architecture": (
        "Stewie names the key architectural components and how they connect. "
        "Peter thinks it sounds like something from his house or car. "
        "Stewie corrects him with a technically precise final line."
    ),
}


def _generate_dialogue_slide(role: str, title: str, context_chunks: list[str], paper_title: str) -> dict:
    """Generate dialogue lines for one slide."""
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context available."
    instructions = DIALOGUE_ROLE_INSTRUCTIONS[role]

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DIALOGUE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Paper: {paper_title}\n\n"
                    f"Context from the paper:\n{context}\n\n"
                    f"Slide topic: {title}\n"
                    f"Instructions: {instructions}\n\n"
                    "Write the dialogue. Remember: Stewie first, Stewie last, max 55 words total."
                ),
            },
        ],
    )

    data = json.loads(response.choices[0].message.content)
    lines = data.get("lines", [])

    # Also generate an image prompt (same as standard pipeline)
    img_response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a DALL-E image prompt for a clean portrait-orientation diagram "
                    "(no text, no equations, shapes/arrows/icons, bold colours, white background). "
                    'Return JSON: {"image_prompt": "..."}'
                ),
            },
            {
                "role": "user",
                "content": f"Paper: {paper_title}\nSlide topic: {title}\nContext: {context[:400]}",
            },
        ],
    )
    image_prompt = json.loads(img_response.choices[0].message.content).get("image_prompt", "")

    # Build a flat narration string for caption generation
    narration = " ".join(f"{l['speaker'].capitalize()}: {l['text']}" for l in lines)

    return {"lines": lines, "narration": narration, "image_prompt": image_prompt}


def generate_dialogue_caption(slides: list[dict], paper_metadata: dict) -> str:
    """Generate an Instagram caption from the dialogue slides."""
    narrations = "\n".join(
        f"Slide {s['slide_number']} ({s['title']}): {s['narration']}"
        for s in slides
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You write punchy Instagram captions for ML paper explainer Reels "
                    "narrated by Stewie and Peter Griffin. "
                    "Format:\n"
                    "Line 1: a bold hook (question or provocative statement)\n"
                    "Blank line\n"
                    "2-4 bullet lines with emojis summarising what viewers will learn\n"
                    "Blank line\n"
                    "Short closing line referencing Stewie & Peter\n"
                    "Blank line\n"
                    "10-15 hashtags on one line\n\n"
                    "Under 200 words. No markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper: {paper_metadata['title']}\n\n"
                    f"Slide narrations:\n{narrations}\n\n"
                    "Write the Instagram caption."
                ),
            },
        ],
    )
    caption = response.choices[0].message.content.strip()
    arxiv_id = paper_metadata.get("arxiv_id", "")
    if arxiv_id:
        caption += f"\n\n📄 Full paper: https://arxiv.org/abs/{arxiv_id}"
    return caption


def generate_dialogue_script(collection, paper_metadata: dict) -> list[dict]:
    """
    Run RAG queries for each slide, generate Stewie/Peter dialogue,
    and return a list of 6 slide dicts with 'lines' instead of 'narration'.
    """
    print("  Generating hook text...")
    hook_text = generate_hook_text(paper_metadata)
    print(f"  🪝 Hook: {hook_text}")

    slides = []
    for i, slide_config in enumerate(SLIDE_QUERIES):
        print(f"  Generating dialogue slide {i + 1}/6: {slide_config['title']}...")

        chunks = retrieve(
            collection,
            question=slide_config["question"],
            section_filter=slide_config["section_filter"],
        )

        content = _generate_dialogue_slide(
            role=slide_config["role"],
            title=slide_config["title"],
            context_chunks=chunks,
            paper_title=paper_metadata["title"],
        )

        slide = {
            "slide_number": i + 1,
            "title": slide_config["title"],
            "role": slide_config["role"],
            "lines": content["lines"],
            "narration": content["narration"],   # flat string for captions/hook
            "image_prompt": content["image_prompt"],
            "duration_seconds": None,
            "audio_path": None,
            "image_path": None,
        }
        if i == 0:
            slide["hook_text"] = hook_text

        slides.append(slide)

    return slides
