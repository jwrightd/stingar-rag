"""
Dialogue pipeline entry point.

Stewie Griffin explains the ML paper to Peter Griffin in a back-and-forth
conversation. Same arXiv discovery and PDF ingestion as the main pipeline,
but script generation and media use two voices and Q&A format.

Usage:
    python3 main_dialogue.py              # discover fresh paper
    python3 main_dialogue.py --test       # use "Attention Is All You Need"
"""
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from discovery import discover_paper
from ingestion import ingest_paper
from dialogue_script_gen import generate_dialogue_script, generate_dialogue_caption
from dialogue_media import generate_dialogue_media
from video import assemble_video

Path("output").mkdir(exist_ok=True)

TEST_PAPER = {
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "abstract": (
        "The dominant sequence transduction models are based on complex recurrent "
        "or convolutional neural networks. We propose a new simple network architecture, "
        "the Transformer, based solely on attention mechanisms."
    ),
    "authors": ["Vaswani et al."],
    "pdf_url": "https://arxiv.org/pdf/1706.03762",
    "selection_reason": "Test mode — hardcoded paper",
}


def run_dialogue_pipeline(test: bool = False):
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
    print(f"✅ Indexed {len(chunks)} chunks, found {len(figures)} figure(s)")

    print("\n✍️  Generating Stewie & Peter dialogue script...")
    slides = generate_dialogue_script(collection, paper)
    print(f"✅ Dialogue script ready: {len(slides)} slides")

    # Print a preview of the dialogue
    for s in slides[:2]:
        print(f"\n  [{s['title']}]")
        for line in s["lines"]:
            print(f"    {line['speaker'].upper()}: {line['text']}")

    print("\n🎙️  Generating audio and images...")
    slides = generate_dialogue_media(slides, paper["arxiv_id"], figures=figures)
    print("✅ Media generated")

    print("\n📝 Generating caption...")
    caption = generate_dialogue_caption(slides, paper)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caption_path = Path("output") / paper["arxiv_id"] / f"caption_dialogue_{timestamp}.txt"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption, encoding="utf-8")
    print(f"✅ Caption saved to: {caption_path}")
    print(f"\n--- CAPTION PREVIEW ---\n{caption}\n-----------------------")

    print("\n🎬 Assembling video...")
    video_path = assemble_video(slides, paper, paper["arxiv_id"])
    print(f"\n✅ Done! Dialogue video saved to: {video_path}")

    return video_path


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    run_dialogue_pipeline(test=test_mode)
