# src/evaluation/chunk_size_evaluator.py

import logging
import statistics
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List

# Configure Logging
logging.basicConfig(
    filename='logs/chunk_size_evaluator.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def evaluate_chunk_sizes(output_path: str = 'results/') -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluates and analyzes chunk size distributions.
    Generates histograms for each chunking method and domain.
    Returns a dictionary with metrics.
    
    Args:
        output_path (str): Directory to save histograms and metrics.
        
    Returns:
        dict: Chunk size metrics for each method and domain.
    """
    try:
        logger.info("Starting evaluation of chunk sizes.")
        metrics = {}
        os.makedirs(output_path, exist_ok=True)

        chunking_methods = ['gradient', 'interquartile', 'std_deviation', 'percentile']
        domains = ['arxiv', 'pubmed', 'history', 'legal', 'ecommerce']
        
        for method in chunking_methods:
            metrics[method] = {}
            method_path = f'chunks/{method}'
            
            for domain in domains:
                domain_path = os.path.join(method_path, domain)
                print(f"\nChecking domain path: {domain_path}")
                if not os.path.exists(domain_path):
                    logger.warning(f"Directory not found for method '{method}' in domain '{domain}'. Skipping.")
                    continue
                    
                print(f"Domain path exists. Contents: {os.listdir(domain_path)}")
                sizes = []
                for chunk_dir in os.listdir(domain_path):
                    chunk_dir_path = os.path.join(domain_path, chunk_dir)
                    print(f"Checking chunk directory: {chunk_dir_path}")
                    if os.path.isdir(chunk_dir_path):
                        metadata_file = os.path.join(chunk_dir_path, 'metadata.json')
                        print(f"Looking for metadata file: {metadata_file}")
                        if os.path.exists(metadata_file):
                            print(f"Found metadata file: {metadata_file}")
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                sizes.extend(metadata['chunk_sizes'])
                        else:
                            print(f"Metadata file not found at: {metadata_file}")
                    else:
                        print(f"Not a directory: {chunk_dir_path}")
                
                if not sizes:
                    logger.warning(f"No chunks found for method '{method}' in domain '{domain}'.")
                    continue

                metrics[method][domain] = {
                    'mean_size': statistics.mean(sizes),
                    'median_size': statistics.median(sizes),
                    'std_dev': statistics.stdev(sizes) if len(sizes) > 1 else 0,
                    'min_size': min(sizes),
                    'max_size': max(sizes)
                }

                # Generate histogram
                plt.figure(figsize=(10, 6))
                plt.hist(sizes, bins=20, alpha=0.7, color='blue')
                plt.title(f'Chunk Size Distribution for {method} Method in {domain} Domain')
                plt.xlabel('Number of Tokens')
                plt.ylabel('Frequency')
                plt.grid(True)
                histogram_path = os.path.join(output_path, f"{method}_{domain}_distribution.png")
                plt.savefig(histogram_path)
                plt.close()
                logger.info(f"Generated histogram for method '{method}' in domain '{domain}'. Saved to '{histogram_path}'.")
        
        # Save metrics to JSON file
        metrics_file = os.path.join(output_path, 'chunk_sizes_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved chunk size metrics to '{metrics_file}'.")

        logger.info("Completed evaluation of chunk sizes.")
        return metrics
    except Exception as e:
        logger.error(f"Error in evaluating chunk sizes: {e}")
        return {}



if __name__ == "__main__":
    evaluate_chunk_sizes()