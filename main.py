import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from discovery import discover_paper
from ingestion import ingest_paper
from script_gen import generate_script, generate_caption
from media import generate_media
from video import assemble_video

Path("output").mkdir(exist_ok=True)

# Known good paper for testing: "Attention Is All You Need" (the Transformer paper)
TEST_PAPER = {
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
    "authors": ["Vaswani et al."],
    "pdf_url": "https://arxiv.org/pdf/1706.03762",
    "selection_reason": "Test mode — hardcoded paper",
}


def run_pipeline(test: bool = False, voice: str = "default"):
    if test:
        print("\n🧪 Test mode — using hardcoded paper (skipping arXiv discovery)")
        paper = TEST_PAPER
    else:
        print("\n🔍 Discovering paper...")
        paper = discover_paper()
    print(f"✅ Selected: {paper['title']}")
    print(f"   Reason: {paper.get('selection_reason', '')}")

    print("\n📄 Ingesting PDF...")
    collection, chunks, figures = ingest_paper(paper["pdf_url"], paper["arxiv_id"])
    print(f"✅ Indexed {len(chunks)} chunks across {len(set(c['metadata']['section'] for c in chunks))} sections")
    print(f"   Found {len(figures)} paper figure(s)")

    print("\n✍️  Generating script...")
    slides = generate_script(collection, paper)
    print(f"✅ Script ready: {len(slides)} slides")

    print("\n🎙️  Generating audio and images...")
    slides = generate_media(slides, paper["arxiv_id"], figures=figures, voice=voice)
    print("✅ Media generated")

    print("\n✍️  Generating caption...")
    caption = generate_caption(slides, paper)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caption_path = Path("output") / paper["arxiv_id"] / f"caption_{timestamp}.txt"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption, encoding="utf-8")
    print(f"✅ Caption saved to: {caption_path}")
    print(f"\n--- CAPTION PREVIEW ---\n{caption}\n-----------------------")

    print("\n🎬 Assembling video...")
    video_path = assemble_video(slides, paper, paper["arxiv_id"])
    print(f"\n✅ Done! Video saved to: {video_path}")

    return video_path


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    voice = "default"
    for arg in sys.argv:
        if arg.startswith("--voice="):
            voice = arg.split("=", 1)[1]
    run_pipeline(test=test_mode, voice=voice)
