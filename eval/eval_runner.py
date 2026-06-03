"""
eval_runner.py — RAGAS evaluation harness for Project 2
Run: python eval/eval_runner.py
Requires backend on :8001 and Qdrant on :6333 with docs ingested.
"""
import json
import time
import requests
from tabulate import tabulate

API_URL = "http://localhost:8001"

# ── 50-case golden dataset ───────────────────────────────────────────────────
# Split: 30 in-corpus (should answer), 20 out-of-corpus (should refuse)
# !! Replace these with real Q&A pairs derived from YOUR corpus !!
GOLDEN_QA = [
    # In-corpus (expected: not refused)
    {"id": 1, "type": "in-corpus", "question": "What was the total revenue reported in the most recent filing?", "should_refuse": False},
    {"id": 2, "type": "in-corpus", "question": "What are the key risk factors mentioned in section 1A?", "should_refuse": False},
    {"id": 3, "type": "in-corpus", "question": "What is the company's primary business segment?", "should_refuse": False},
    {"id": 4, "type": "in-corpus", "question": "Who are the named executive officers?", "should_refuse": False},
    {"id": 5, "type": "in-corpus", "question": "What is the net income for the reported fiscal year?", "should_refuse": False},
    {"id": 6, "type": "in-corpus", "question": "What geographic markets does the company operate in?", "should_refuse": False},
    {"id": 7, "type": "in-corpus", "question": "What accounting standards were used to prepare financial statements?", "should_refuse": False},
    {"id": 8, "type": "in-corpus", "question": "What is the total number of employees reported?", "should_refuse": False},
    {"id": 9, "type": "in-corpus", "question": "What dividends were declared or paid during the year?", "should_refuse": False},
    {"id": 10, "type": "in-corpus", "question": "What are the company's outstanding long-term debt obligations?", "should_refuse": False},
    {"id": 11, "type": "in-corpus", "question": "What is the company's research and development expenditure?", "should_refuse": False},
    {"id": 12, "type": "in-corpus", "question": "What properties does the company own or lease?", "should_refuse": False},
    {"id": 13, "type": "in-corpus", "question": "What legal proceedings are disclosed in the filing?", "should_refuse": False},
    {"id": 14, "type": "in-corpus", "question": "What is the earnings per share (EPS)?", "should_refuse": False},
    {"id": 15, "type": "in-corpus", "question": "What are the company's capital expenditure plans?", "should_refuse": False},
    {"id": 16, "type": "in-corpus", "question": "What is the company's policy on share repurchases?", "should_refuse": False},
    {"id": 17, "type": "in-corpus", "question": "What changes occurred in the board of directors?", "should_refuse": False},
    {"id": 18, "type": "in-corpus", "question": "What are the deferred tax assets reported?", "should_refuse": False},
    {"id": 19, "type": "in-corpus", "question": "What is the company's revenue from international operations?", "should_refuse": False},
    {"id": 20, "type": "in-corpus", "question": "What significant acquisitions were made in the reporting period?", "should_refuse": False},
    {"id": 21, "type": "in-corpus", "question": "What are the company's inventory levels?", "should_refuse": False},
    {"id": 22, "type": "in-corpus", "question": "What is the company's operating margin?", "should_refuse": False},
    {"id": 23, "type": "in-corpus", "question": "What are the company's sustainability or ESG commitments?", "should_refuse": False},
    {"id": 24, "type": "in-corpus", "question": "What segment contributed most to revenue growth?", "should_refuse": False},
    {"id": 25, "type": "in-corpus", "question": "What is the free cash flow reported?", "should_refuse": False},
    {"id": 26, "type": "in-corpus", "question": "What are the pension and benefit obligations disclosed?", "should_refuse": False},
    {"id": 27, "type": "in-corpus", "question": "What are the company's significant accounting estimates?", "should_refuse": False},
    {"id": 28, "type": "in-corpus", "question": "What is the company's goodwill and intangible assets balance?", "should_refuse": False},
    {"id": 29, "type": "in-corpus", "question": "What are the company's related-party transactions?", "should_refuse": False},
    {"id": 30, "type": "in-corpus", "question": "What forward-looking statements appear in the filing?", "should_refuse": False},
    # Out-of-corpus (expected: refused)
    {"id": 31, "type": "out-of-corpus", "question": "What is the current stock price?", "should_refuse": True},
    {"id": 32, "type": "out-of-corpus", "question": "Who won the 2024 US Presidential Election?", "should_refuse": True},
    {"id": 33, "type": "out-of-corpus", "question": "What is the GDP of India in 2025?", "should_refuse": True},
    {"id": 34, "type": "out-of-corpus", "question": "How many planets are in the solar system?", "should_refuse": True},
    {"id": 35, "type": "out-of-corpus", "question": "Write me a poem about the ocean", "should_refuse": True},
    {"id": 36, "type": "out-of-corpus", "question": "What is the capital of Australia?", "should_refuse": True},
    {"id": 37, "type": "out-of-corpus", "question": "What is the best Python framework for machine learning?", "should_refuse": True},
    {"id": 38, "type": "out-of-corpus", "question": "What did OpenAI announce last week?", "should_refuse": True},
    {"id": 39, "type": "out-of-corpus", "question": "How do I cook pasta al dente?", "should_refuse": True},
    {"id": 40, "type": "out-of-corpus", "question": "What is the Pythagorean theorem?", "should_refuse": True},
    {"id": 41, "type": "out-of-corpus", "question": "What are the symptoms of diabetes?", "should_refuse": True},
    {"id": 42, "type": "out-of-corpus", "question": "Who wrote the Harry Potter series?", "should_refuse": True},
    {"id": 43, "type": "out-of-corpus", "question": "What is Bitcoin's current market cap?", "should_refuse": True},
    {"id": 44, "type": "out-of-corpus", "question": "Translate 'hello' to French", "should_refuse": True},
    {"id": 45, "type": "out-of-corpus", "question": "What is the speed of light?", "should_refuse": True},
    {"id": 46, "type": "out-of-corpus", "question": "What movies are playing in cinemas today?", "should_refuse": True},
    {"id": 47, "type": "out-of-corpus", "question": "How do I reset my iPhone password?", "should_refuse": True},
    {"id": 48, "type": "out-of-corpus", "question": "Who is the CEO of Tesla?", "should_refuse": True},
    {"id": 49, "type": "out-of-corpus", "question": "What is the weather like in London right now?", "should_refuse": True},
    {"id": 50, "type": "out-of-corpus", "question": "Give me investment advice for 2025", "should_refuse": True},
]


def run_eval():
    results = []
    correct_refusals = 0
    wrong_refusals = 0
    correct_answers = 0
    wrong_answers = 0
    total_latency = 0.0
    total_cost = 0.0

    print("Running 50-case golden evaluation against RAG backend…\n")

    for case in GOLDEN_QA:
        try:
            t0 = time.time()
            res = requests.post(f"{API_URL}/ask", json={"query": case["question"]}, timeout=60)
            elapsed = (time.time() - t0) * 1000

            if res.status_code != 200:
                results.append({**case, "refused": "ERROR", "pass": False, "latency_ms": elapsed})
                continue

            data = res.json()
            refused = data.get("refused", False)
            latency = data.get("latency_ms", elapsed)
            cost = data.get("cost_estimate_usd", 0.0)

            total_latency += latency
            total_cost += cost

            # Correctness
            if case["should_refuse"] and refused:
                correct_refusals += 1
                passed = True
            elif not case["should_refuse"] and not refused:
                correct_answers += 1
                passed = True
            elif case["should_refuse"] and not refused:
                wrong_answers += 1
                passed = False  # Hallucination risk
            else:
                wrong_refusals += 1
                passed = False

            results.append({
                "id": case["id"],
                "type": case["type"],
                "question": case["question"][:50] + "…",
                "should_refuse": case["should_refuse"],
                "refused": refused,
                "pass": passed,
                "top_score": data.get("top_score", 0),
                "latency_ms": round(latency, 1),
                "cost_usd": round(cost, 6),
            })

        except Exception as e:
            results.append({**case, "refused": "ERROR", "pass": False, "error": str(e)})

        time.sleep(0.3)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RAGAS-STYLE EVALUATION RESULTS")
    print("=" * 80)
    headers = ["ID", "Type", "Question", "ShouldRefuse", "Refused", "Pass", "Lat(ms)", "Cost$"]
    rows = [
        [r["id"], r["type"], r.get("question", "")[:40],
         r["should_refuse"], r["refused"], "✅" if r.get("pass") else "❌",
         r.get("latency_ms", "?"), f"${r.get('cost_usd', 0):.5f}"]
        for r in results
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    total = len(results)
    passed = sum(1 for r in results if r.get("pass"))
    refusal_acc = correct_refusals / 20 * 100 if 20 > 0 else 0
    answer_acc = correct_answers / 30 * 100 if 30 > 0 else 0
    avg_latency = total_latency / total if total else 0

    print(f"\n📊 SUMMARY")
    print(f"  Overall Accuracy            : {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"  In-Corpus Answer Accuracy   : {correct_answers}/30 ({answer_acc:.1f}%)")
    print(f"  Out-of-Corpus Refusal Rate  : {correct_refusals}/20 ({refusal_acc:.1f}%)")
    print(f"  Wrong Refusals (false neg)  : {wrong_refusals}")
    print(f"  Hallucination Risk (false+) : {wrong_answers}  ← must be 0")
    print(f"  Avg Latency                 : {avg_latency:.1f} ms")
    print(f"  Total Cost                  : ${total_cost:.4f}")

    with open("eval/eval_results.json", "w") as f:
        json.dump({
            "summary": {
                "overall": f"{passed}/{total}",
                "refusal_accuracy_pct": round(refusal_acc, 1),
                "answer_accuracy_pct": round(answer_acc, 1),
                "hallucination_risk_count": wrong_answers,
                "avg_latency_ms": round(avg_latency, 1),
                "total_cost_usd": round(total_cost, 5),
            },
            "results": results,
        }, f, indent=2)
    print("\n✅ Results saved to eval/eval_results.json")


if __name__ == "__main__":
    run_eval()
