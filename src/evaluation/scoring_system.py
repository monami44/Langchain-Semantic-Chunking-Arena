# src/evaluation/scoring_system.py

import logging
import statistics
import os
import json
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

# Configure Logging
logging.basicConfig(
    filename='logs/scoring_system.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create a mapping between lowercase method names and their capitalized versions
method_mapping = {
    "percentile": "Percentile",
    "std_deviation": "StdDeviation",
    "interquartile": "Interquantile",
    "gradient": "Gradient"
}

def calculate_scores(chunk_size_metrics: Dict[str, Dict[str, Dict[str, float]]],
                     retrieval_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    print("Starting calculate_scores function")
    try:
        logger.info("Starting calculation of final scores.")
        scores = {}
        for method in chunk_size_metrics:
            print(f"Processing method: {method}")
            scores[method] = {}
            for domain in chunk_size_metrics[method]:
                print(f"Processing domain: {domain}")
                size_score = calculate_size_score(chunk_size_metrics[method][domain])
                print(f"Size score for {method} in {domain}: {size_score}")
                retrieval_domain = 'medical' if domain == 'pubmed' else 'scientific'
                print(f"Retrieval domain: {retrieval_domain}")
                retrieval_score = calculate_retrieval_score(retrieval_metrics.get(method_mapping.get(method.lower(), method), {}).get(retrieval_domain, {}))
                print(f"Retrieval score for {method} in {retrieval_domain}: {retrieval_score}")
                total_score = (size_score * 0.4) + (retrieval_score * 0.6)  # Weighted scoring
                print(f"Total score for {method} in {domain}: {total_score}")
                scores[method][domain] = round(total_score, 2)
        logger.info("Completed calculation of final scores.")
        print(f"Final scores: {scores}")
        return scores
    except Exception as e:
        logger.error(f"Error in calculating scores: {e}")
        print(f"Error in calculate_scores: {e}")
        return {}

def calculate_size_score(size_metrics: Dict[str, float]) -> float:
    print("Starting calculate_size_score function")
    print(f"Input size_metrics: {size_metrics}")
    try:
        mean = size_metrics.get('mean_size', 0)
        std_dev = size_metrics.get('std_dev', 0)
        min_size = size_metrics.get('min_size', 0)
        max_size = size_metrics.get('max_size', 0)
        print(f"Mean: {mean}, Std Dev: {std_dev}, Min Size: {min_size}, Max Size: {max_size}")

        mean_score = max(0, 200 - mean)
        std_dev_score = max(0, 200 - std_dev)
        min_size_score = min(min_size * 10, 100)
        print(f"Mean Score: {mean_score}, Std Dev Score: {std_dev_score}, Min Size Score: {min_size_score}")

        if max_size <= 500:
            max_size_score = 100
        elif max_size <= 750:
            max_size_score = 50
        else:
            max_size_score = 0
        print(f"Max Size Score: {max_size_score}")

        size_score = (mean_score * 0.35) + (std_dev_score * 0.35) + (min_size_score * 0.15) + (max_size_score * 0.15)
        print(f"Final size score: {size_score}")
        return size_score
    except Exception as e:
        logger.error(f"Error in calculating size score: {e}")
        print(f"Error in calculate_size_score: {e}")
        return 0

def calculate_retrieval_score(retrieval_metrics: Dict[str, float]) -> float:
    print("Starting calculate_retrieval_score function")
    print(f"Input retrieval_metrics: {retrieval_metrics}")
    try:
        precision = retrieval_metrics.get('precision', 0)
        recall = retrieval_metrics.get('recall', 0)
        f1 = retrieval_metrics.get('f1_score', 0)
        average_precision = retrieval_metrics.get('average_precision', 0)
        ndcg = retrieval_metrics.get('ndcg', 0)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, AP: {average_precision}, NDCG: {ndcg}")

        retrieval_score = (
            precision * 100 * 0.2 +
            recall * 100 * 0.2 +
            f1 * 100 * 0.2 +
            average_precision * 100 * 0.2 +
            ndcg * 100 * 0.2
        )
        print(f"Final retrieval score: {retrieval_score}")
        return retrieval_score
    except Exception as e:
        logger.error(f"Error in calculating retrieval score: {e}")
        print(f"Error in calculate_retrieval_score: {e}")
        return 0

def plot_scores(scores_file: str):
    print("Starting plot_scores function")
    try:
        with open(scores_file, 'r') as f:
            scores = json.load(f)

        # Define consistent colors and display names
        domain_colors = {
            'arxiv': '#2ecc71',      # green
            'pubmed': '#3498db',     # blue
            'history': '#e74c3c',    # red
            'legal': '#f1c40f',      # yellow
            'ecommerce': '#9b59b6'   # purple
        }

        domain_display_names = {
            'arxiv': 'Machine Learning',
            'pubmed': 'Medical',
            'history': 'History',
            'legal': 'Legal',
            'ecommerce': 'E-commerce'
        }

        methods = list(scores.keys())
        domains = list(scores[methods[0]].keys())

        # Create larger figure with more spacing
        fig, ax = plt.subplots(figsize=(15, 8))
        bar_width = 0.15  # Reduced bar width for better spacing
        index = np.arange(len(methods))

        # Plot bars for each domain
        for i, domain in enumerate(domains):
            domain_scores = [scores[method][domain] for method in methods]
            ax.bar([x + i*bar_width for x in index], 
                   domain_scores, 
                   bar_width, 
                   label=domain_display_names[domain],
                   color=domain_colors[domain],
                   alpha=0.8)

        # Customize the plot
        ax.set_xlabel('Methods', fontsize=12, labelpad=10)
        ax.set_ylabel('Scores', fontsize=12, labelpad=10)
        ax.set_title('Comparison of Scores Across Methods and Domains', fontsize=14, pad=20)
        
        # Center the x-tick labels
        ax.set_xticks([x + (bar_width * (len(domains)-1))/2 for x in index])
        ax.set_xticklabels(methods, rotation=45)
        
        # Add legend with better positioning
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('results/scores_comparison.png', bbox_inches='tight', dpi=300)
        print("Scores plot saved as 'results/scores_comparison.png'")
    except Exception as e:
        print(f"Error in plotting scores: {e}")
        logger.error(f"Error in plotting scores: {e}")

