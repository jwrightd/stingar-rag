# ML Paper Video Generator

Automatically turns the latest ML papers into narrated Instagram Reels — no manual steps.

## Pipeline

```
arXiv API
    ↓
Fetch 20 recent ML papers (cs.LG, cs.AI, cs.CV, cs.CL)
    ↓
GPT-4o-mini scores each paper 1-10 (novelty, explainability, interest)
    ↓  filter out already-seen papers via cache
Pick the winner
    ↓
Download PDF → extract text by section → chunk → embed into ChromaDB
Scrape arXiv HTML → download paper figures
    ↓
For each of 6 slides, run a targeted RAG query:
  • query decomposition (GPT-4o-mini breaks question → 3 sub-questions)
  • retrieve candidates from ChromaDB
  • cross-encoder reranker picks best 4 chunks
    ↓
GPT-4o writes 2-sentence narration + image prompt per slide
GPT-4o writes Instagram caption
    ↓
ElevenLabs TTS → MP3 per slide
gpt-image-1 → image per slide (slides 1–5)
Architecture slide (slide 6) → scraped paper figure, fit to frame
    ↓
PIL composites each slide: background + bottom bar + title + counter
moviepy assembles: title card + 6 slide clips → final_video_TIMESTAMP.mp4
Caption saved to caption_TIMESTAMP.txt
```

**6 slides:** The Problem → The Idea → How It Works → The Results → Why It Matters → The Architecture

**Output:** ~60 second 1080×1920 MP4 + Instagram caption, ready to post.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
```

## Usage

```bash
source .venv/bin/activate

# Full pipeline — discovers a fresh paper from arXiv
python3 main.py

# Test mode — uses "Attention Is All You Need", skips arXiv discovery
python3 main.py --test

# Run daily at 08:00 automatically
python3 scheduler.py
```

Output is saved to `output/{arxiv_id}/`:

| File | Description |
|---|---|
| `final_video_TIMESTAMP.mp4` | Finished Reel (1080×1920) |
| `caption_TIMESTAMP.txt` | Instagram caption with hashtags |

## Stack

- **arXiv API** — paper discovery
- **pdfplumber** — PDF text extraction
- **ChromaDB** — vector store for RAG
- **OpenAI** — GPT-4o-mini (scoring, query decomposition), GPT-4o (script + caption), gpt-image-1 (slide images), text-embedding-3-small
- **sentence-transformers** — cross-encoder reranking
- **ElevenLabs** — text-to-speech narration
- **Pillow** — slide compositing
- **moviepy** — video assembly
