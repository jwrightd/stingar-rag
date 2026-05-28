import json
from openai import OpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _decompose_query(question: str) -> list[str]:
    """Use GPT-4o-mini to break a high-level question into 3 specific sub-questions."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You decompose research questions into specific sub-questions "
                    "for retrieving relevant sections of an ML paper. "
                    "Return JSON: {\"sub_questions\": [\"...\", \"...\", \"...\"]}"
                ),
            },
            {
                "role": "user",
                "content": f"Decompose this into 3 specific sub-questions: {question}",
            },
        ],
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("sub_questions", [question])


def retrieve(collection, question: str, section_filter: list[str] | None = None) -> list[str]:
    """
    Retrieve the top 4 most relevant chunks for a question using:
    1. Query decomposition into sub-questions
    2. Section-aware filtering
    3. Cross-encoder reranking
    """
    sub_questions = _decompose_query(question)

    # Build ChromaDB where filter
    where = None
    if section_filter:
        where = {"section": {"$in": section_filter}}

    # Retrieve top 5 chunks per sub-question
    seen_texts = set()
    candidate_chunks = []

    for sub_q in sub_questions:
        try:
            results = collection.query(
                query_texts=[sub_q],
                n_results=5,
                where=where,
                include=["documents", "distances"],
            )
        except Exception:
            # Fall back to unfiltered if section filter yields no results
            results = collection.query(
                query_texts=[sub_q],
                n_results=5,
                include=["documents", "distances"],
            )

        for doc in results["documents"][0]:
            if doc not in seen_texts:
                seen_texts.add(doc)
                candidate_chunks.append(doc)

    if not candidate_chunks:
        return []

    # Rerank all candidates against the original question
    pairs = [(question, chunk) for chunk in candidate_chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:4]]
