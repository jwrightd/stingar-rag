import requests
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO


def _score_figure(img_tag, base_url: str) -> tuple[int, str] | None:
    """
    Score a figure by likely relevance (prefers architecture/model diagrams).
    Returns (score, absolute_url) or None if invalid.
    """
    src = img_tag.get("src", "")
    if not src or src.startswith("data:"):
        return None

    # Build absolute URL using urljoin so relative paths resolve correctly
    url = urljoin(base_url, src)

    # Skip tiny icons/logos (likely < 5KB)
    alt = (img_tag.get("alt", "") + " " + img_tag.get("title", "")).lower()
    caption = ""
    figure_parent = img_tag.find_parent("figure")
    if figure_parent:
        figcaption = figure_parent.find("figcaption")
        if figcaption:
            caption = figcaption.get_text().lower()

    combined = alt + " " + caption

    # Score higher if caption mentions architecture keywords
    score = 0
    architecture_keywords = [
        "architecture", "model", "overview", "framework", "structure",
        "diagram", "transformer", "network", "layer", "encoder", "decoder",
        "attention", "figure 1", "fig. 1", "fig 1",
    ]
    for kw in architecture_keywords:
        if kw in combined:
            score += 2

    return score, url


def extract_paper_figures(arxiv_id: str, out_dir: Path, max_figures: int = 3) -> list[str]:
    """
    Scrape figures from the arXiv HTML version of a paper.
    Returns list of local file paths to downloaded figures, best first.
    Falls back to empty list if HTML version is unavailable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(html_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            print(f"  ⚠️  arXiv HTML not available for {arxiv_id} (status {resp.status_code})")
            return []
    except Exception as e:
        print(f"  ⚠️  Could not fetch arXiv HTML: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    # Use the final URL after redirects as the base for relative paths
    # Do NOT add trailing slash — urljoin needs the page URL, not a directory URL
    base_url = resp.url

    # Find all img tags inside figure elements
    figures = soup.find_all("figure")
    scored = []
    for fig in figures:
        for img in fig.find_all("img"):
            result = _score_figure(img, base_url)
            if result:
                scored.append(result)

    # Sort by score descending, deduplicate URLs
    seen_urls = set()
    unique = []
    for score, url in sorted(scored, key=lambda x: x[0], reverse=True):
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append((score, url))

    # Download top figures, skip ones that are too small
    saved_paths = []
    for i, (score, url) in enumerate(unique[:max_figures * 2]):
        if len(saved_paths) >= max_figures:
            break
        try:
            img_resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            img_resp.raise_for_status()
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            # Skip tiny images (likely icons)
            if img.width < 200 or img.height < 200:
                continue
            path = str(out_dir / f"figure_{len(saved_paths)}.png")
            img.save(path, "PNG")
            saved_paths.append(path)
            print(f"  📸 Downloaded figure {len(saved_paths)}: {img.width}×{img.height}px")
        except Exception as e:
            print(f"  ⚠️  Could not download figure: {e}")
            continue

    return saved_paths


def search_architecture_figure(paper_title: str, out_dir: Path) -> str | None:
    """
    Web-search for an architecture diagram for this paper using DuckDuckGo images.
    Returns local file path if a usable image is found, else None.
    """
    import time
    from ddgs import DDGS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query = f'"{paper_title}" architecture diagram'
    print(f"  🔍 Searching web for architecture figure: {query}")

    results = []
    for attempt in range(3):
        try:
            results = list(DDGS().images(query, max_results=15))
            break
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt  # 1s, 2s
                print(f"  ⚠️  Search attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ⚠️  Web image search failed after 3 attempts: {e}")
                return None

    for result in results:
        url = result.get("image", "")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            if img.width < 300 or img.height < 200:
                continue
            path = str(out_dir / "figure_web.png")
            img.save(path, "PNG")
            print(f"  ✅ Web figure downloaded: {img.width}×{img.height}px")
            return path
        except Exception:
            continue

    print("  ⚠️  No usable web figure found")
    return None


def best_architecture_figure(figures: list[str]) -> str | None:
    """Return the path to the largest figure (most likely to be a full architecture diagram)."""
    if not figures:
        return None
    # Pick largest by pixel area
    best = max(figures, key=lambda p: Image.open(p).size[0] * Image.open(p).size[1])
    return best
