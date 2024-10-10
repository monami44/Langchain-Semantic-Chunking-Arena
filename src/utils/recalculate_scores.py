import json
import os
import logging
from typing import Dict

# Configure Logging
logging.basicConfig(
    filename='logs/recalculate_scores.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_json(file_path: str) -> Dict:
    """
    Loads a JSON file and returns its content as a dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded '{file_path}'.")
        return data
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{file_path}': {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading '{file_path}': {e}")
        return {}

def save_json(data: Dict, file_path: str):
    """
    Saves a dictionary as a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved '{file_path}'.")
    except Exception as e:
        logger.error(f"Error saving '{file_path}': {e}")

def calculate_size_score(size_metrics: Dict[str, float]) -> float:
    """
    Calculates the size score based on mean size, standard deviation, min size, and max size.
    Lower mean and standard deviation are better.
    """
    try:
        mean = size_metrics.get('mean_size', 0)
        std_dev = size_metrics.get('std_dev', 0)
        min_size = size_metrics.get('min_size', 0)
        max_size = size_metrics.get('max_size', 0)

        # Configurable Thresholds
        mean_threshold = 200  # Increased from 100
        std_dev_threshold = 200  # Increased from 20

        # Calculate individual scores
        mean_score = max(0, mean_threshold - mean)  # Higher means are penalized
        std_dev_score = max(0, std_dev_threshold - std_dev)  # Higher std_dev are penalized

        # Enhanced Scaling for min_size
        min_size_score = min(min_size * 10, 100)  # Increased scaling factor

        # Partial Credit for max_size
        if max_size <= 500:
            max_size_score = 100
        elif max_size <= 750:
            max_size_score = 50
        else:
            max_size_score = 0

        # Rebalanced Weights
        size_score = (
            mean_score * 0.35 +
            std_dev_score * 0.35 +
            min_size_score * 0.15 +
            max_size_score * 0.15
        )

        logger.info(f"Calculated size_score: {size_score} (mean: {mean}, std_dev: {std_dev}, min: {min_size}, max: {max_size})")
        return size_score
    except Exception as e:
        logger.error(f"Error in calculating size score: {e}")
        return 0

def calculate_retrieval_score(retrieval_metrics: Dict[str, float]) -> float:
    """
    Calculates the retrieval score based on precision, recall, F1-score, average precision, and NDCG.
    """
    try:
        precision = retrieval_metrics.get('precision', 0)
        recall = retrieval_metrics.get('recall', 0)
        f1 = retrieval_metrics.get('f1_score', 0)
        average_precision = retrieval_metrics.get('average_precision', 0)
        ndcg = retrieval_metrics.get('ndcg', 0)

        # Normalize metrics to a 0-100 scale
        retrieval_score = (
            precision * 100 * 0.2 +
            recall * 100 * 0.2 +
            f1 * 100 * 0.2 +
            average_precision * 100 * 0.2 +
            ndcg * 100 * 0.2
        )

        logger.info(f"Calculated retrieval_score: {retrieval_score} (precision: {precision}, recall: {recall}, f1: {f1}, average_precision: {average_precision}, ndcg: {ndcg})")
        return retrieval_score
    except Exception as e:
        logger.error(f"Error in calculating retrieval score: {e}")
        return 0

def calculate_scores(chunk_size_metrics: Dict[str, Dict[str, Dict[str, float]]],
                    retrieval_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculates the final scores for each chunking method based on chunk size and retrieval quality.
    """
    try:
        logger.info("Starting calculation of final scores.")
        scores = {}
        for method, domains in chunk_size_metrics.items():
            scores[method] = {}
            for domain, metrics in domains.items():
                size_score = calculate_size_score(metrics)
                retrieval_score = calculate_retrieval_score(retrieval_metrics.get(method, {}).get(domain, {}))
                total_score = (size_score * 0.4) + (retrieval_score * 0.6)  # Weighted scoring
                scores[method][domain] = round(total_score, 2)
                logger.info(f"Method: {method}, Domain: {domain}, Size Score: {size_score}, Retrieval Score: {retrieval_score}, Total Score: {total_score}")
        logger.info("Completed calculation of final scores.")
        return scores
    except Exception as e:
        logger.error(f"Error in calculating scores: {e}")
        return {}

def main():
    # Define paths
    chunk_size_metrics_path = 'results/chunk_sizes_metrics.json'
    retrieval_metrics_path = 'results/retrieval_quality_metrics.json'
    scores_output_path = 'results/scores_updated.json'

    # Load existing metrics
    chunk_size_metrics = load_json(chunk_size_metrics_path)
    retrieval_metrics = load_json(retrieval_metrics_path)

    if not chunk_size_metrics:
        logger.error("Chunk size metrics are empty. Exiting.")
        print("Chunk size metrics are empty. Exiting.")
        return

    if not retrieval_metrics:
        logger.error("Retrieval quality metrics are empty. Exiting.")
        print("Retrieval quality metrics are empty. Exiting.")
        return

    # Calculate scores
    scores = calculate_scores(chunk_size_metrics, retrieval_metrics)

    # Save updated scores
    save_json(scores, scores_output_path)
    print(f"Recalculated scores have been saved to '{scores_output_path}'.")

if __name__ == "__main__":
    main()
