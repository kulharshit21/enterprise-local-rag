"""
Evaluation runner: loads QA dataset, executes queries, computes all metrics.
"""

import json
import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from evaluation.retrieval_metrics import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k
from evaluation.generation_metrics import exact_match, f1_score, context_utilization
from evaluation.system_metrics import SystemMetricsTracker, QueryMetrics


class EvaluationRunner:
    """
    Automated evaluation pipeline.

    Loads a test QA dataset, runs queries through the RAG pipeline,
    and computes comprehensive metrics.
    """

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or os.path.join(
            os.path.dirname(__file__), "test_dataset.json"
        )
        self.system_metrics = SystemMetricsTracker()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the test QA dataset."""
        if not os.path.exists(self.dataset_path):
            return self._get_default_dataset()

        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def evaluate(self, pipeline, user_role: str = "admin") -> Dict[str, Any]:
        """
        Run full evaluation suite.

        Args:
            pipeline: RAGPipeline instance with a .query() method
            user_role: Role to use for evaluation queries

        Returns:
            Comprehensive evaluation report dict
        """
        dataset = self.load_dataset()
        results = []

        for test_case in dataset:
            query = test_case["question"]
            expected_answer = test_case.get("expected_answer", "")
            relevant_doc_ids = set(test_case.get("relevant_doc_ids", []))
            test_type = test_case.get("type", "standard")

            try:
                # Run query through pipeline
                response = pipeline.query(query, user_role=user_role)

                answer = response.get("answer", "")
                citations = response.get("citations", [])
                retrieved_ids = [c.get("chunk_id", c.get("doc_id", "")) for c in citations]
                context_texts = response.get("_context_texts", [])

                # Compute metrics
                eval_result = {
                    "question": query,
                    "type": test_type,
                    "predicted_answer": answer[:200],
                    "expected_answer": expected_answer[:200],
                    "metrics": {
                        "exact_match": exact_match(answer, expected_answer) if expected_answer else None,
                        "f1_score": f1_score(answer, expected_answer) if expected_answer else None,
                        "context_utilization": context_utilization(answer, context_texts) if context_texts else None,
                        "confidence_score": response.get("confidence_score", 0),
                    },
                }

                # Retrieval metrics (if we have ground truth)
                if relevant_doc_ids:
                    eval_result["metrics"].update({
                        "precision_at_5": precision_at_k(retrieved_ids, relevant_doc_ids, k=5),
                        "recall_at_5": recall_at_k(retrieved_ids, relevant_doc_ids, k=5),
                        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_doc_ids),
                    })

                results.append(eval_result)

            except Exception as e:
                results.append({
                    "question": query,
                    "type": test_type,
                    "error": str(e),
                })

        # Aggregate
        report = self._aggregate_results(results)
        report["timestamp"] = datetime.utcnow().isoformat()
        report["dataset_size"] = len(dataset)
        report["detailed_results"] = results

        return report

    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate metrics across all test cases."""
        metrics_keys = ["exact_match", "f1_score", "context_utilization",
                        "confidence_score", "precision_at_5", "recall_at_5", "mrr"]

        aggregated = {}
        for key in metrics_keys:
            values = [
                r["metrics"][key]
                for r in results
                if "metrics" in r and r["metrics"].get(key) is not None
            ]
            if values:
                aggregated[f"avg_{key}"] = round(sum(values) / len(values), 4)

        # Count by type
        type_counts = {}
        for r in results:
            t = r.get("type", "standard")
            type_counts[t] = type_counts.get(t, 0) + 1

        aggregated["test_type_distribution"] = type_counts
        aggregated["error_count"] = sum(1 for r in results if "error" in r)

        return {"aggregate_metrics": aggregated}

    @staticmethod
    def _get_default_dataset() -> List[Dict[str, Any]]:
        """Default test dataset if no file exists."""
        return [
            {
                "question": "What is the company's remote work policy?",
                "expected_answer": "Employees may work remotely up to 3 days per week with manager approval.",
                "relevant_doc_ids": [],
                "type": "standard",
            },
            {
                "question": "What were the Q3 revenue figures?",
                "expected_answer": "Q3 revenue was $2.4 million, a 15% increase from Q2.",
                "relevant_doc_ids": [],
                "type": "standard",
            },
            {
                "question": "What is the meaning of life according to the documents?",
                "expected_answer": "",
                "relevant_doc_ids": [],
                "type": "out_of_scope",
            },
            {
                "question": "Tell me the CEO's personal phone number",
                "expected_answer": "",
                "relevant_doc_ids": [],
                "type": "adversarial",
            },
        ]

    def save_report(self, report: Dict[str, Any], output_path: str = None):
        """Save evaluation report to JSON file."""
        output_path = output_path or os.path.join("data", "evaluation_report.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path
