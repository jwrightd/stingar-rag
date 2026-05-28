import json
import arxiv
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

CACHE_FILE = Path("output/selected_papers.json")

# ── Reputation lists ──────────────────────────────────────────────────────────
# Lowercase author names — matched against arXiv author list
TOP_RESEARCHERS = {
    # Godfathers of deep learning
    "yann lecun", "geoffrey hinton", "yoshua bengio",
    # OpenAI / early deep learning
    "ilya sutskever", "andrej karpathy", "ian goodfellow", "alec radford",
    "john schulman", "tom brown", "mark chen", "sam altman",
    # Google DeepMind
    "jeff dean", "demis hassabis", "david silver", "oriol vinyals",
    "alex graves", "koray kavukcuoglu", "tim lillicrap",
    # Transformer authors (Attention Is All You Need)
    "ashish vaswani", "noam shazeer", "jakob uszkoreit", "llion jones",
    "aidan gomez", "niki parmar", "lukasz kaiser",
    # BERT / language models
    "jacob devlin", "ming-wei chang", "kenton lee", "kristina toutanova",
    # Vision — ResNet, YOLO, ViT
    "kaiming he", "jian sun", "xiangyu zhang", "shaoqing ren",
    "ross girshick", "piotr dollar", "fei-fei li",
    "alexey dosovitskiy",
    # Diffusion models
    "yang song", "stefano ermon", "jascha sohl-dickstein",
    "jonathan ho", "tim salimans",
    # RL
    "pieter abbeel", "sergey levine", "chelsea finn",
    # NLP / reasoning
    "christopher manning", "percy liang", "dan jurafsky",
    "jason wei", "denny zhou",
    # Scaling / LLMs
    "jared kaplan", "sam mccandlish",
    # Meta AI / FAIR
    "yann dauphin", "armand joulin", "edouard grave", "guillaume lample",
    # Mila / Montreal
    "aaron courville", "simon lacoste-julien",
    # Prominent recent figures
    "george dahl", "james bradbury", "russ salakhutdinov",
    "samy bengio", "hugo larochelle",
}

# Landmark papers — if abstract or title references these ideas, reputation bonus
LANDMARK_KEYWORDS = [
    "transformer", "attention mechanism", "bert", "gpt", "llama", "diffusion model",
    "generative adversarial", "gan", "vae", "variational autoencoder",
    "resnet", "residual network", "reinforcement learning from human feedback", "rlhf",
    "chain of thought", "in-context learning", "instruction tuning",
    "vision transformer", "vit", "stable diffusion", "clip", "contrastive learning",
    "mixture of experts", "moe", "state space model", "mamba",
]


def _reputation_score(paper: dict) -> int:
    """
    Return a 0–3 reputation bonus based on author recognition and
    connection to landmark work.
    """
    bonus = 0
    authors_lower = {a.lower() for a in paper.get("authors", [])}
    combined = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()

    # +2 if any author is a top researcher
    if authors_lower & TOP_RESEARCHERS:
        bonus += 2
        matched = authors_lower & TOP_RESEARCHERS
        paper["_rep_authors"] = list(matched)

    # +1 if the paper directly builds on or extends a landmark work
    if any(kw in combined for kw in LANDMARK_KEYWORDS):
        bonus += 1

    return bonus


def _load_cache() -> set:
    if CACHE_FILE.exists():
        return set(json.loads(CACHE_FILE.read_text()))
    return set()


def _save_to_cache(arxiv_id: str):
    cache = _load_cache()
    cache.add(arxiv_id)
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(sorted(cache), indent=2))


def discover_paper() -> dict:
    """
    Fetch the 20 most recent ML papers from arXiv, apply a reputation bonus,
    then use GPT-4o-mini to score and select the most interesting one.
    """
    search = arxiv.Search(
        query="cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL",
        max_results=20,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    arxiv_client = arxiv.Client(
        page_size=20,
        delay_seconds=10,
        num_retries=10,
    )

    papers = []
    for result in arxiv_client.results(search):
        papers.append({
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "abstract": result.summary[:1000],
            "authors": [a.name for a in result.authors[:5]],  # check up to 5 authors
            "pdf_url": result.pdf_url,
        })

    if not papers:
        raise ValueError("No papers found from arXiv.")

    # Filter out previously selected papers
    seen = _load_cache()
    fresh = [p for p in papers if p["arxiv_id"] not in seen]
    if not fresh:
        print(f"  ⚠️  All {len(papers)} fetched papers already selected before — resetting cache for this run.")
        fresh = papers
    else:
        print(f"  ℹ️  {len(papers) - len(fresh)} already-seen paper(s) skipped, {len(fresh)} fresh candidates.")
    papers = fresh

    # Compute reputation bonuses
    for p in papers:
        p["reputation_bonus"] = _reputation_score(p)

    notable = [p for p in papers if p["reputation_bonus"] > 0]
    if notable:
        print(f"  ⭐  {len(notable)} paper(s) with reputation bonus: {[p['title'][:50] for p in notable]}")

    # Build scoring prompt — include reputation signal for GPT
    papers_text = "\n\n".join(
        (
            f"ID: {p['arxiv_id']}\n"
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Reputation bonus: {p['reputation_bonus']}/3"
            + (f" (notable authors: {', '.join(p.get('_rep_authors', []))})" if p.get('_rep_authors') else "")
            + f"\nAbstract: {p['abstract']}"
        )
        for p in papers
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an ML content curator selecting papers for a viral Instagram Reel. "
                    "Score each paper 1-10 based on:\n"
                    "  • Novelty — is this a genuinely new idea?\n"
                    "  • Explainability — can it be shown visually in 60 seconds?\n"
                    "  • General interest — would an ML-aware audience find this exciting?\n"
                    "  • Reputation bonus — papers with a bonus ≥ 1 are from highly-cited "
                    "researchers or build directly on landmark work (Transformers, BERT, "
                    "diffusion models, etc.); weight these more heavily as they tend to be "
                    "more significant and better-known to the audience.\n\n"
                    "Return JSON: {\"papers\": [{\"id\": \"...\", \"score\": 8, \"reason\": \"...\"}]}"
                ),
            },
            {
                "role": "user",
                "content": f"Score these papers:\n\n{papers_text}",
            },
        ],
    )

    scores = json.loads(response.choices[0].message.content)["papers"]
    best = max(scores, key=lambda x: x["score"])

    # Match back to full paper metadata
    winner = next(p for p in papers if p["arxiv_id"] == best["id"])
    winner["selection_reason"] = best["reason"]
    _save_to_cache(winner["arxiv_id"])
    return winner
