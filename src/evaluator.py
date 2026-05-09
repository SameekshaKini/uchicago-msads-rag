import os
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)

DEFAULT_TEST_SET = [
    {
        "question": "What are the core courses in the MS in Applied Data Science program?",
        "ground_truth": (
            "The core courses include Machine Learning, Data Engineering Platforms, "
            "Statistical Inference, and Applied Data Science."
        ),
    },
    {
        "question": "What are the admission requirements for the MS-ADS program?",
        "ground_truth": (
            "Applicants need a bachelor's degree in a related field with coursework in "
            "programming, statistics, and mathematics, plus a personal statement, "
            "letters of recommendation, and a resume."
        ),
    },
    {
        "question": "Can I study the MS-ADS program online?",
        "ground_truth": (
            "Yes, the MS in Applied Data Science is available in both in-person "
            "and online formats."
        ),
    },
    {
        "question": "What is the capstone project in the MS-ADS program?",
        "ground_truth": (
            "The capstone project is a key component where students work on real-world "
            "data science problems, applying their skills to develop data-driven solutions."
        ),
    },
    {
        "question": "What career outcomes do MS-ADS graduates typically achieve?",
        "ground_truth": (
            "Graduates work as data scientists, machine learning engineers, "
            "data analysts, and AI researchers across various industries."
        ),
    },
    {
        "question": "How long does the MS in Applied Data Science program take?",
        "ground_truth": (
            "The program is typically completed in about one year of full-time study."
        ),
    },
    {
        "question": "What is the tuition for the MS-ADS program?",
        "ground_truth": (
            "Tuition and fee details are listed on the UChicago MS-ADS tuition page. "
            "Financial aid and scholarships may be available."
        ),
    },
    {
        "question": "Who are the instructors in the MS-ADS program?",
        "ground_truth": (
            "The program has faculty with expertise in machine learning, statistics, "
            "data engineering, and applied data science from the University of Chicago."
        ),
    },
    {
        "question": "What is the difference between the online and in-person MS-ADS programs?",
        "ground_truth": (
            "Both programs share the same curriculum and degree. The in-person program "
            "is held in Chicago, while the online program allows remote participation "
            "with the same coursework and faculty."
        ),
    },
    {
        "question": "What are the application deadlines for the MS-ADS program?",
        "ground_truth": (
            "Application deadlines vary by quarter. Check the events and deadlines page "
            "on the MS-ADS website for current dates."
        ),
    },
]


def simple_evaluate(question: str, answer: str, retrieved_docs: list) -> dict:
    """
    Fast heuristic evaluation — no LLM calls needed.
    Returns scores in [0, 1] for each dimension.
    """
    answer_lower = answer.lower()

    # 1. Has a real answer (not a fallback)
    fallback_phrases = [
        "don't have that information",
        "i don't have",
        "not in my knowledge base",
        "please contact",
    ]
    has_answer = (
        len(answer.strip()) > 30
        and not any(p in answer_lower for p in fallback_phrases)
    )

    # 2. Retrieved enough context
    context_score = min(len(retrieved_docs) / 5, 1.0)

    # 3. Keyword overlap between question and answer
    stop_words = {
        "what", "is", "the", "are", "a", "an", "in", "of", "for",
        "how", "can", "i", "do", "does", "tell", "me", "about",
        "program", "ms", "ads", "uchicago",
    }
    q_keywords = set(question.lower().split()) - stop_words
    a_words    = set(answer_lower.split())
    overlap    = q_keywords & a_words
    relevance  = min(len(overlap) / max(len(q_keywords), 1), 1.0)

    # 4. Faithfulness proxy: penalise hedging/uncertainty language
    hedging_phrases = [
        "i think", "i believe", "probably", "might be",
        "not certain", "i'm not sure",
    ]
    has_hedging  = any(p in answer_lower for p in hedging_phrases)
    faithfulness = 0.7 if has_hedging else 1.0

    # 5. Has source citation
    has_sources = "sources:" in answer_lower or "source:" in answer_lower
    citation_score = 1.0 if has_sources else 0.5

    composite = (
        float(has_answer) + context_score + relevance + faithfulness + citation_score
    ) / 5

    return {
        "has_answer":     float(has_answer),
        "context_score":  round(context_score, 4),
        "relevance":      round(relevance, 4),
        "faithfulness":   round(faithfulness, 4),
        "citation_score": round(citation_score, 4),
        "composite":      round(composite, 4),
    }


def run_simple_evaluation(
    rag_chain,
    test_set: list = None,
    output_path: str = "data/eval_results.json",
    verbose: bool = True,
) -> dict:
    """
    Runs heuristic evaluation on the test set.
    No LLM API calls — works offline/free.
    """
    if test_set is None:
        test_set = DEFAULT_TEST_SET

    log.info("Running simple evaluation on %d questions...", len(test_set))

    records = []
    metric_keys = ["has_answer", "context_score", "relevance",
                   "faithfulness", "citation_score", "composite"]
    totals = {k: 0.0 for k in metric_keys}

    for i, item in enumerate(test_set, 1):
        q  = item["question"]
        gt = item.get("ground_truth", "")
        if verbose:
            print(f"  [{i}/{len(test_set)}] {q[:60]}...")

        try:
            result = rag_chain.ask(q)
            ans = result["answer"]
            docs = result["docs"]
            scores = simple_evaluate(q, ans, docs)
        except Exception as exc:
            log.warning("Error on question %r: %s", q, exc)
            ans = ""
            docs = []
            scores = {k: 0.0 for k in metric_keys}

        record = {
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "num_docs_retrieved": len(docs),
            **scores,
        }
        records.append(record)
        for k in metric_keys:
            totals[k] += scores.get(k, 0.0)

    n        = len(test_set)
    averages = {f"avg_{k}": round(v / n, 4) for k, v in totals.items()}

    output = {
        "evaluated_at": datetime.now().isoformat(),
        "num_questions": n,
        "averages": averages,
        "per_question": records,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    log.info("Evaluation complete. Results saved to %s", output_path)
    return output


def run_ragas_evaluation(
    rag_chain,
    test_set: list = None,
    output_path: str = "data/ragas_eval_results.json",
) -> dict:
    """
    Full RAGAS evaluation (LLM-as-judge).
    Requires: pip install ragas datasets
    Uses API credits — run after simple evaluation passes.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "RAGAS not installed. Run: pip install ragas datasets"
        )

    if test_set is None:
        test_set = DEFAULT_TEST_SET

    log.info("Running RAGAS evaluation on %d questions...", len(test_set))

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in test_set:
        q  = item["question"]
        gt = item.get("ground_truth", "")
        try:
            result = rag_chain.ask(q)
            ans = result["answer"]
            ctx_lst = [d.page_content for d in result["docs"]]
        except Exception as exc:
            log.warning("Error on %r: %s", q, exc)
            ans, ctx_lst = "", []

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx_lst)
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result  = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    metrics = dict(result)
    metrics["evaluated_at"] = datetime.now().isoformat()
    metrics["num_questions"] = len(test_set)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    log.info("RAGAS evaluation complete -> %s", output_path)
    return metrics
