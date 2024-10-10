import json
import os
import logging

def save_results(chunk_size_metrics, retrieval_metrics, scores, output_dir='results/'):
    """
    Saves the evaluation results to JSON files.
    
    Args:
        chunk_size_metrics (dict): Metrics related to chunk sizes.
        retrieval_metrics (dict): Metrics related to retrieval quality.
        scores (dict): Final scores for each chunking method.
        output_dir (str): Directory to save the results.
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Saving results to '{output_dir}'.")
        
        # Define paths
        chunk_size_path = os.path.join(output_dir, 'chunk_sizes_metrics.json')
        retrieval_quality_path = os.path.join(output_dir, 'retrieval_quality_metrics.json')
        scores_path = os.path.join(output_dir, 'scores.json')
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunk size metrics
        with open(chunk_size_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_size_metrics, f, indent=4)
        logger.info(f"Saved chunk size metrics to '{chunk_size_path}'.")
        
        # Save retrieval quality metrics
        with open(retrieval_quality_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_metrics, f, indent=4)
        logger.info(f"Saved retrieval quality metrics to '{retrieval_quality_path}'.")
        
        # Save scores
        with open(scores_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=4)
        logger.info(f"Saved scores to '{scores_path}'.")
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")
