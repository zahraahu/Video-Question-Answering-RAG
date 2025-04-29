import time
import csv
import numpy as np
from collections import defaultdict
from embed import embed_texts, embed_texts_clip
from utils.utils import load_json, save_json
from retrieve import text_retrieve, image_retrieve
from utils.paths import GOLD_TEST_PATH, EVALUATION_RESULTS_PATH, EVALUATION_SUMMARY_CSV

TOLERANCE = 20  # seconds

def evaluate_retrieval_method(query, query_embedding, expected_timestamp, retrieval_method="faiss", index_type="ivfflat", modality="text", num_runs=5):
    """Perform retrieval runs and collect results."""
    latency = []
    result_timestamp = None

    for i in range(num_runs):
        start_time = time.time()

        if modality == "text":
            _, result_timestamp, _ = text_retrieve(query, query_embedding, retrieval_method, index_type)
        elif modality == "image":
            _, result_timestamp, _ = image_retrieve(query_embedding, retrieval_method, index_type)
        else:
            result_timestamp = None

        latency.append(time.time() - start_time)
    
    print(f"Result Timestamp: {result_timestamp}, Expected Timestamp: {expected_timestamp}")
    return result_timestamp, expected_timestamp, np.mean(latency)


def summarize_results(all_results):
    """Compute summary statistics from detailed results."""
    grouped = defaultdict(list)

    for res in all_results:
        key = (res["retrieval_method"], res["modality"], res.get("index_type", None))
        grouped[key].append(res)

    summary = []

    for (retrieval_method, modality, index_type), results in grouped.items():
        correct = 0
        false_positives = 0
        total_latency = 0
        unanswerable_total = 0

        for r in results:
            exp = r["expected_timestamp"]
            gen = r["generated_timestamp"]
            latency = r["latency"]
            total_latency += latency

            if exp == "" and gen is None:
                correct += 1  # Correctly rejected
                unanswerable_total += 1
            elif exp == "" and gen:
                false_positives += 1
                unanswerable_total += 1 # Incorrectly returned something
            elif exp and gen and abs(float(exp) - float(gen)) <= TOLERANCE:
                correct += 1

        total = len(results)
        accuracy = correct / total
        rejection_quality = (unanswerable_total - false_positives) / unanswerable_total
        avg_latency = total_latency / total

        summary.append({
            "retrieval_method": retrieval_method,
            "modality": modality,
            "index_type": index_type,
            "mean_accuracy": accuracy,
            "mean_rejection_quality": rejection_quality,
            "mean_avg_latency": avg_latency
        })

    return summary


def save_summary_to_csv(summary):
    fields = ["Retrieval Method", "Modality", "Index Type", "Accuracy", "Rejection Quality", "Latency"]
    with open(EVALUATION_SUMMARY_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                "Retrieval Method": row["retrieval_method"],
                "Modality": row["modality"],
                "Index Type": row.get("index_type", ""),
                "Accuracy": row["mean_accuracy"],
                "Rejection Quality": row["mean_rejection_quality"],
                "Latency": row["mean_avg_latency"]
            })
    print(f"Summary saved to {EVALUATION_SUMMARY_CSV}")


def main():
    gold_test = load_json(GOLD_TEST_PATH)
    all_results = []
    queries_grouped_results = []

    for query, expected_timestamp in gold_test.items():
        print(f"\n=== Query: {query} ===")
        query_results = []

        methods = ["faiss", "postgres-ivfflat", "postgres-hnsw", "tfidf", "bm25"]
        modalities = ["text", "image"]

        for modality in modalities:

            if modality == "text":
                query_embedding = embed_texts([query])[0]
            elif modality == "image":
                query_embedding = embed_texts_clip([query])[0]

            for method in methods:
                if modality == "image" and method in ["tfidf", "bm25"]:
                    continue

                if method == "postgres-hnsw":
                    index = "hnsw"
                elif method == "postgres-ivfflat":
                    index = "ivfflat"
                else:
                    index = None

                print(f"\nEvaluating {method} with {modality} modality...")
                method_base = method.split('-')[0]

                result_timestamp, expected_timestamp, latency = evaluate_retrieval_method(
                    query, query_embedding, expected_timestamp,
                    retrieval_method=method_base, index_type=index,
                    modality=modality, num_runs=5
                )

                result = {
                    "retrieval_method": method_base,
                    "modality": modality,
                    "expected_timestamp": expected_timestamp,
                    "generated_timestamp": result_timestamp,
                    "latency": latency
                }
                if index:
                    result["index_type"] = index

                query_results.append(result)
                all_results.append(result)

        queries_grouped_results.append({
            "query": query,
            "results": query_results
        })

    # Save detailed per-query results
    save_json(queries_grouped_results, EVALUATION_RESULTS_PATH)

    # Summarize and save
    summary = summarize_results(all_results)
    save_summary_to_csv(summary)


if __name__ == "__main__":
    main()