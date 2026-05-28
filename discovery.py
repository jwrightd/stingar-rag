import json
import arxiv
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def discover_paper() -> dict:
    """
    Fetch the 20 most recent ML papers from arXiv and use GPT-4o-mini
    to score and select the most interesting one for a video.
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
            "abstract": result.summary[:1000],  # trim very long abstracts
            "authors": [a.name for a in result.authors[:3]],
            "pdf_url": result.pdf_url,
        })

    if not papers:
        raise ValueError("No papers found from arXiv.")

    # Build scoring prompt
    papers_text = "\n\n".join(
        f"ID: {p['arxiv_id']}\nTitle: {p['title']}\nAbstract: {p['abstract']}"
        for p in papers
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an ML content curator. Score each paper 1-10 based on: "
                    "novelty (is this a new idea?), explainability (can it be shown visually?), "
                    "and general interest to an ML audience. "
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
    return winner
