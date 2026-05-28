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
        "Introduce the paper's main contribution. Explain the core idea simply. "
        "Start with 'This paper proposes...' or similar."
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


def generate_script(collection, paper_metadata: dict) -> list[dict]:
    """
    Run targeted RAG queries for each slide topic, then generate
    narration + image prompt for each slide using GPT-4o.
    Returns a list of 5 slide dicts.
    """
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

        slides.append({
            "slide_number": i + 1,
            "title": slide_config["title"],
            "role": slide_config["role"],
            "narration": content["narration"],
            "image_prompt": content["image_prompt"],
            "duration_seconds": None,
            "audio_path": None,
            "image_path": None,
        })

    return slides
