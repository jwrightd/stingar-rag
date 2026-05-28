import json
import random
import arxiv
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

CACHE_FILE = Path("output/selected_papers.json")

# ── Reputation lists ──────────────────────────────────────────────────────────
TOP_RESEARCHERS = {
    # Godfathers of deep learning
    "yann lecun", "geoffrey hinton", "yoshua bengio",
    # OpenAI / early deep learning
    "ilya sutskever", "andrej karpathy", "ian goodfellow", "alec radford",
    "john schulman", "tom brown", "mark chen",
    # Google DeepMind
    "jeff dean", "demis hassabis", "david silver", "oriol vinyals",
    "alex graves", "koray kavukcuoglu", "tim lillicrap",
    # Transformer authors
    "ashish vaswani", "noam shazeer", "jakob uszkoreit", "llion jones",
    "aidan gomez", "niki parmar", "lukasz kaiser",
    # BERT / language models
    "jacob devlin", "ming-wei chang", "kenton lee", "kristina toutanova",
    # Vision — ResNet, ViT
    "kaiming he", "jian sun", "xiangyu zhang", "shaoqing ren",
    "ross girshick", "piotr dollar", "fei-fei li", "alexey dosovitskiy",
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
    # Others
    "george dahl", "james bradbury", "russ salakhutdinov",
    "samy bengio", "hugo larochelle",
}

LANDMARK_KEYWORDS = [
    "transformer", "attention mechanism", "bert", "gpt", "llama", "diffusion model",
    "generative adversarial", "gan", "vae", "variational autoencoder",
    "resnet", "residual network", "reinforcement learning from human feedback", "rlhf",
    "chain of thought", "in-context learning", "instruction tuning",
    "vision transformer", "vit", "stable diffusion", "clip", "contrastive learning",
    "mixture of experts", "moe", "state space model", "mamba",
]

# ── Landmark paper library ────────────────────────────────────────────────────
# arXiv IDs of the most important / well-known ML papers of all time.
# These compete with recent papers every run — once selected they go in the cache.
LANDMARK_PAPERS = [
    # ── Foundational architectures ──────────────────────────────────────────
    "1706.03762",   # Attention Is All You Need (Transformer)
    "1512.03385",   # Deep Residual Learning (ResNet)
    "1406.2661",    # Generative Adversarial Networks (GAN)
    "1312.6114",    # Auto-Encoding Variational Bayes (VAE)
    "1301.3781",    # Word2Vec — Efficient Estimation of Word Representations
    "1409.0473",    # Neural Machine Translation (Bahdanau attention)
    "1502.03167",   # Batch Normalization
    "1412.6980",    # Adam optimizer
    "1207.0580",    # Dropout
    # ── Language models ─────────────────────────────────────────────────────
    "1810.04805",   # BERT
    "2005.14165",   # GPT-3 (Language Models are Few-Shot Learners)
    "2302.13971",   # LLaMA
    "2307.09288",   # LLaMA 2
    "2310.06825",   # Mistral 7B
    "2201.11903",   # Chain-of-Thought Prompting
    "2203.02155",   # InstructGPT / RLHF
    "2212.08073",   # Constitutional AI (Anthropic)
    "2005.00796",   # RAG — Retrieval-Augmented Generation
    "2106.09685",   # LoRA
    "2305.11206",   # QLoRA
    "2001.08361",   # Scaling Laws for Neural Language Models
    # ── Vision ──────────────────────────────────────────────────────────────
    "2010.11929",   # Vision Transformer (ViT)
    "2103.00020",   # CLIP (Learning Transferable Visual Models)
    "2102.12092",   # DALL-E
    "2204.06125",   # DALL-E 2
    "1608.06993",   # DenseNet
    "1409.1556",    # VGGNet (Very Deep CNNs)
    "2103.14030",   # DINO (Self-supervised ViT)
    # ── Diffusion models ────────────────────────────────────────────────────
    "2006.11239",   # DDPM — Denoising Diffusion Probabilistic Models
    "2011.13456",   # Score-based generative models (Song et al.)
    "2112.10752",   # Latent Diffusion Models / Stable Diffusion
    "2207.12598",   # Classifier-Free Diffusion Guidance
    # ── Reinforcement learning ───────────────────────────────────────────────
    "1312.5602",    # DQN — Playing Atari with Deep RL
    "1707.06347",   # PPO — Proximal Policy Optimization
    "1801.01290",   # SAC — Soft Actor-Critic
    "1509.02971",   # DDPG
    # ── Efficiency / systems ────────────────────────────────────────────────
    "2205.14135",   # FlashAttention
    "2307.08691",   # FlashAttention-2
    "1603.05027",   # Deep learning on graphs (Kipf & Welling, GCN)
    # ── Multimodal / agents ─────────────────────────────────────────────────
    "2204.05862",   # Flamingo (visual language model)
    "2301.12503",   # InstructBLIP
    "2303.08774",   # GPT-4 Technical Report
    "2309.10020",   # Phi-1.5
    # ── State space / alternative architectures ──────────────────────────────
    "2312.00752",   # Mamba — Linear-Time Sequence Modeling
    "2305.13048",   # RWKV
]


def _normalize_id(arxiv_id: str) -> str:
    """Strip version suffix — '2605.28820v1' → '2605.28820'."""
    return arxiv_id.split("v")[0]


def _reputation_score(paper: dict) -> int:
    """Return a 0–3 reputation bonus."""
    bonus = 0
    authors_lower = {a.lower() for a in paper.get("authors", [])}
    combined = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()

    if authors_lower & TOP_RESEARCHERS:
        bonus += 2
        paper["_rep_authors"] = list(authors_lower & TOP_RESEARCHERS)

    if any(kw in combined for kw in LANDMARK_KEYWORDS):
        bonus += 1

    return bonus


def _fetch_landmark_candidates(seen: set, arxiv_client: arxiv.Client, n: int = 10) -> list[dict]:
    """
    Randomly sample up to n unseen papers from LANDMARK_PAPERS,
    fetch their metadata from arXiv, and return them.
    """
    unseen_ids = [pid for pid in LANDMARK_PAPERS if _normalize_id(pid) not in seen]
    if not unseen_ids:
        return []

    sample_ids = random.sample(unseen_ids, min(n, len(unseen_ids)))
    results = []
    try:
        search = arxiv.Search(id_list=sample_ids)
        for result in arxiv_client.results(search):
            results.append({
                "arxiv_id": _normalize_id(result.entry_id.split("/")[-1]),
                "title": result.title,
                "abstract": result.summary[:1000],
                "authors": [a.name for a in result.authors[:5]],
                "pdf_url": result.pdf_url,
                "is_landmark": True,
            })
    except Exception as e:
        print(f"  ⚠️  Could not fetch landmark papers: {e}")

    return results


def _load_cache() -> set:
    if CACHE_FILE.exists():
        # Normalize any legacy versioned IDs already in the cache
        return {_normalize_id(i) for i in json.loads(CACHE_FILE.read_text())}
    return set()


def _save_to_cache(arxiv_id: str):
    cache = _load_cache()
    cache.add(_normalize_id(arxiv_id))
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(sorted(cache), indent=2))


def discover_paper() -> dict:
    """
    Fetch 20 recent papers + up to 10 unseen landmark classics,
    apply reputation bonuses, then use GPT-4o-mini to pick the best one.
    """
    arxiv_client = arxiv.Client(
        page_size=20,
        delay_seconds=10,
        num_retries=10,
    )

    # ── Recent papers ─────────────────────────────────────────────────────────
    search = arxiv.Search(
        query="cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL",
        max_results=20,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    recent = []
    for result in arxiv_client.results(search):
        recent.append({
            "arxiv_id": _normalize_id(result.entry_id.split("/")[-1]),
            "title": result.title,
            "abstract": result.summary[:1000],
            "authors": [a.name for a in result.authors[:5]],
            "pdf_url": result.pdf_url,
            "is_landmark": False,
        })

    if not recent:
        raise ValueError("No papers found from arXiv.")

    # ── Filter seen papers ────────────────────────────────────────────────────
    seen = _load_cache()
    fresh_recent = [p for p in recent if p["arxiv_id"] not in seen]
    skipped = len(recent) - len(fresh_recent)
    if skipped:
        print(f"  ℹ️  {skipped} already-seen recent paper(s) skipped.")

    # ── Landmark classics ─────────────────────────────────────────────────────
    classics = _fetch_landmark_candidates(seen, arxiv_client, n=10)
    print(f"  📚 {len(classics)} unseen landmark paper(s) added to candidate pool.")

    # ── Combine pools ─────────────────────────────────────────────────────────
    papers = fresh_recent + classics
    if not papers:
        print("  ⚠️  All candidates already seen — resetting cache for this run.")
        papers = recent

    # ── Reputation bonuses ────────────────────────────────────────────────────
    for p in papers:
        p["reputation_bonus"] = _reputation_score(p)
        # Landmark classics get an extra inherent bonus
        if p.get("is_landmark"):
            p["reputation_bonus"] = min(3, p["reputation_bonus"] + 1)

    notable = [p for p in papers if p["reputation_bonus"] > 0]
    if notable:
        print(f"  ⭐  {len(notable)} paper(s) with reputation bonus.")

    # ── GPT-4o-mini scoring ───────────────────────────────────────────────────
    papers_text = "\n\n".join(
        (
            f"ID: {p['arxiv_id']}\n"
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Type: {'LANDMARK CLASSIC' if p.get('is_landmark') else 'Recent'}\n"
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
                    "  • Novelty / impact — is this a genuinely important idea?\n"
                    "  • Explainability — can it be shown visually in 60 seconds?\n"
                    "  • Audience interest — would ML practitioners find this exciting?\n"
                    "  • Reputation bonus — papers with bonus ≥ 1 are from top researchers "
                    "or directly extend landmark work; weight these more heavily.\n"
                    "  • LANDMARK CLASSIC papers are timeless, highly-cited papers that "
                    "are still widely discussed — these score very high unless the audience "
                    "has almost certainly seen them already.\n\n"
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

    winner = next(p for p in papers if p["arxiv_id"] == best["id"])
    winner["selection_reason"] = best["reason"]
    _save_to_cache(winner["arxiv_id"])
    return winner
