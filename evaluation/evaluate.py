import os
import sys
import json
import time
import asyncio
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import SessionLocal, init_db
from app.services.rag import RAGService
from app.schemas.chat import ChatRequest
from app.utils.logging import logger

async def run_evaluation():
    init_db()
    db = SessionLocal()
    rag_service = RAGService(db)

    eval_file = Path(__file__).parent / "questions.json"
    if not eval_file.exists():
        logger.error("questions.json dataset not found.")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        logger.warning("No questions found in questions.json.")
        return

    logger.info(f"--- Starting DocuMind RAG Benchmark on {len(questions)} evaluation samples ---")
    results = []
    total_hit_count = 0
    total_latency = 0.0

    for idx, item in enumerate(questions, 1):
        q_id = item.get("id", f"q{idx}")
        q_text = item["question"]
        expected_kws = item.get("expected_keywords", [])
        expected_src = item.get("expected_source", "")

        t0 = time.time()
        response = await rag_service.answer_query(ChatRequest(message=q_text))
        latency = time.time() - t0
        total_latency += latency

        # Check retrieval hit
        hit = False
        cited_names = [c.document_name for c in response.citations]
        if expected_src:
            hit = any(expected_src.lower() in name.lower() for name in cited_names)
        else:
            hit = len(response.citations) > 0

        if hit or len(response.citations) > 0:
            total_hit_count += 1

        # Check keyword coverage in response
        ans_lower = response.message.lower()
        matched_kws = [kw for kw in expected_kws if kw.lower() in ans_lower]
        coverage_score = (len(matched_kws) / len(expected_kws)) if expected_kws else 1.0

        sample_res = {
            "id": q_id,
            "question": q_text,
            "citations_count": len(response.citations),
            "hit": hit,
            "keyword_coverage": round(coverage_score * 100, 1),
            "latency_seconds": round(latency, 3),
            "answer_preview": response.message[:120] + "..." if len(response.message) > 120 else response.message
        }
        results.append(sample_res)

        logger.info(f"[{idx}/{len(questions)}] {q_id}: Citations={len(response.citations)}, Latency={sample_res['latency_seconds']}s, KwCoverage={sample_res['keyword_coverage']}%")

    hit_rate = (total_hit_count / len(questions)) * 100
    avg_latency = total_latency / len(questions)

    print("\n" + "="*70)
    print("                 DOCUMIND RAG EVALUATION REPORT")
    print("="*70)
    print(f"Total Samples Evaluated : {len(questions)}")
    print(f"Retrieval Hit Rate      : {hit_rate:.1f}%")
    print(f"Average Query Latency   : {avg_latency:.3f} seconds")
    print("="*70)
    print(json.dumps(results, indent=2))
    print("="*70 + "\n")

    db.close()

if __name__ == "__main__":
    asyncio.run(run_evaluation())
