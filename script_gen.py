import json
from openai import OpenAI
from rag import retrieve
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SLIDE_QUERIES = [
    {
        "question": "What problem does this paper solve and why does it matter?",
        "section_filter": ["Abstract", "Introduction"],
        "role": "hook",
        "title": "The Problem",
    },
    {
        "question": "What is the key idea or main contribution of this paper?",
        "section_filter": ["Abstract", "Introduction", "Conclusion", "Conclusions"],
        "role": "idea",
        "title": "The Idea",
    },
    {
        "question": "How does the proposed method or approach work technically?",
        "section_filter": ["Method", "Methods", "Methodology", "Approach", "Model", "Background"],
        "role": "method",
        "title": "How It Works",
    },
    {
        "question": "What experiments were run and what were the main results or benchmarks?",
        "section_filter": ["Experiments", "Experimental Setup", "Results", "Evaluation"],
        "role": "results",
        "title": "The Results",
    },
    {
        "question": "What are the limitations, future directions, and why does this work matter?",
        "section_filter": ["Discussion", "Conclusion", "Conclusions", "Limitations", "Future Work"],
        "role": "takeaway",
        "title": "Why It Matters",
    },
    {
        "question": "What is the model architecture, its key components, and how are they structured and connected?",
        "section_filter": ["Method", "Methods", "Methodology", "Approach", "Model", "Architecture", "Background"],
        "role": "architecture",
        "title": "The Architecture",
    },
]

ROLE_INSTRUCTIONS = {
    "hook": (
        "Write an engaging hook that opens the video. Start with the problem the paper solves. "
        "Make the viewer feel why this matters. Do not mention the paper title yet."
    ),
    "idea": (
        "Introduce the paper's main contribution. "
        "You MUST say the full paper title naturally in your first sentence — "
        "e.g. '\"Attention Is All You Need\" shows that...' or 'The paper \"LoRA\" proposes...'. "
        "Then explain the core idea simply in the second sentence."
    ),
    "method": (
        "Explain how the method works at an intuitive level. Use an analogy if helpful. "
        "Avoid equations. Focus on the key mechanism that makes this approach work."
    ),
    "results": (
        "Summarize the key results. Mention specific numbers or comparisons if available. "
        "Explain what the numbers mean in plain English — e.g. 'that's 15% better than the previous best'."
    ),
    "takeaway": (
        "Wrap up with why this work matters, what doors it opens, and any notable limitations. "
        "End with a memorable closing line that makes the viewer think."
    ),
    "architecture": (
        "Describe the model's architecture technically but accessibly. "
        "Name the key components (layers, modules, mechanisms) and explain how they connect. "
        "Be specific — mention actual component names from the paper."
    ),
}


def _generate_slide(role: str, title: str, context_chunks: list[str], paper_title: str) -> dict:
    """Generate narration + DALL-E image prompt for a single slide."""
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No specific context available."
    instructions = ROLE_INSTRUCTIONS[role]

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You write narration scripts for 60-second Instagram Reel ML paper explainers. "
                    "Your audience knows ML basics but hasn't read the paper. "
                    "Avoid jargon. Be punchy, fast-paced, and engaging. "
                    "Each slide gets EXACTLY 2 sentences, max 30 words total — every word must earn its place. "
                    "Return JSON: {\"narration\": \"...\", \"image_prompt\": \"...\"}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper: {paper_title}\n\n"
                    f"Context from the paper:\n{context}\n\n"
                    f"Slide role: {title}\n"
                    f"Instructions: {instructions}\n\n"
                    "Write EXACTLY 2 punchy sentences (max 30 words) of narration. "
                    "Then write a DALL-E image prompt for a clean portrait-orientation diagram "
                    "(no text, no equations, use shapes/arrows/icons, bold colours)."
                ),
            },
        ],
    )

    data = json.loads(response.choices[0].message.content)
    return {
        "narration": data.get("narration", ""),
        "image_prompt": data.get("image_prompt", ""),
    }


def generate_hook_text(paper_metadata: dict) -> str:
    """
    Generate a short punchy hook sentence (max 12 words) for the opening frame.
    Creates a curiosity gap to maximise 3-second retention.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You write viral Instagram Reel hooks for ML paper explainers. "
                    "Write ONE sentence, max 12 words. "
                    "Create a curiosity gap — make the viewer feel they MUST keep watching. "
                    "Use present tense. Be specific, bold, and provocative. "
                    "Do NOT mention the paper title. Do NOT use hashtags. "
                    "Examples: "
                    "'This paper just rewrote how every LLM handles attention.' / "
                    "'What if your model could learn from 10x less data?' / "
                    "'The trick behind GPT that nobody talks about — explained.' "
                    "Return only the sentence, no quotes."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper title: {paper_metadata['title']}\n"
                    f"Abstract: {paper_metadata.get('abstract', '')[:500]}\n\n"
                    "Write the hook sentence."
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def generate_caption(slides: list[dict], paper_metadata: dict) -> str:
    """Generate an Instagram caption for the video using the finished slide narrations."""
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
                    "You write punchy Instagram captions for 60-second ML paper explainer Reels. "
                    "Format:\n"
                    "Line 1: a bold hook (question or provocative statement, no hashtags)\n"
                    "Blank line\n"
                    "2-4 bullet lines using relevant emojis summarising what the viewer will learn\n"
                    "Blank line\n"
                    "A short closing line (e.g. 'Full breakdown in the Reel ↑')\n"
                    "Blank line\n"
                    "10-15 relevant hashtags on one line\n\n"
                    "Keep the whole caption under 200 words. No markdown, no bold/italics."
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

    # Append arXiv link
    arxiv_id = paper_metadata.get("arxiv_id", "")
    if arxiv_id:
        caption += f"\n\n📄 Full paper: https://arxiv.org/abs/{arxiv_id}"

    return caption


def generate_script(collection, paper_metadata: dict) -> list[dict]:
    """
    Run targeted RAG queries for each slide topic, then generate
    narration + image prompt for each slide using GPT-4o.
    Returns a list of 6 slide dicts. The first slide includes a hook_text key.
    """
    print("  Generating hook text...")
    hook_text = generate_hook_text(paper_metadata)
    print(f"  🪝 Hook: {hook_text}")

    slides = []
    for i, slide_config in enumerate(SLIDE_QUERIES):
        print(f"  Generating slide {i + 1}/6: {slide_config['title']}...")

        chunks = retrieve(
            collection,
            question=slide_config["question"],
            section_filter=slide_config["section_filter"],
        )

        content = _generate_slide(
            role=slide_config["role"],
            title=slide_config["title"],
            context_chunks=chunks,
            paper_title=paper_metadata["title"],
        )

        slide = {
            "slide_number": i + 1,
            "title": slide_config["title"],
            "role": slide_config["role"],
            "narration": content["narration"],
            "image_prompt": content["image_prompt"],
            "duration_seconds": None,
            "audio_path": None,
            "image_path": None,
        }
        if i == 0:
            slide["hook_text"] = hook_text
        slides.append(slide)

    return slides
