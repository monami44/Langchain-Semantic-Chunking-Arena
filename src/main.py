# src/main.py

import logging
from utils import download_all_datasets, load_datasets, save_results
from utils.preprocessor import process_and_store_datasets
from evaluation.retrieval_quality_evaluator import evaluate_retrieval_quality
from evaluation.chunk_size_evaluator import evaluate_chunk_sizes
from evaluation.scoring_system import calculate_scores
from utils.ground_truth_creator import create_ground_truths
from chunking_methods import PercentileChunker, StdDeviationChunker, InterquartileChunker, GradientChunker
from langchain_huggingface import HuggingFaceEmbeddings
import json
import os

def load_configuration(config_path='config/data_loader_config.json'):
    """
    Loads the configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration JSON file.
    
    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file '{config_path}' not found.")
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    with open(config_path, 'r') as f:
        config = json.load(f)
    logging.info(f"Loaded configuration from '{config_path}'.")
    return config

def load_ground_truths(ground_truth_path='config/ground_truths.json'):
    """
    Loads the ground truths from a JSON file.
    
    Args:
        ground_truth_path (str): Path to the ground truths JSON file.
    
    Returns:
        dict: Ground truths dictionary.
    """
    if not os.path.exists(ground_truth_path):
        logging.error(f"Ground truth file '{ground_truth_path}' not found.")
        raise FileNotFoundError(f"Ground truth file '{ground_truth_path}' not found.")
    
    with open(ground_truth_path, 'r') as f:
        ground_truths = json.load(f)
    logging.info(f"Loaded ground truths from '{ground_truth_path}'.")
    return ground_truths

def chunks_exist(datasets):
    """
    Check if chunks exist for all chunking methods and datasets.
    
    Args:
        datasets (dict): Dictionary of datasets.
    
    Returns:
        bool: True if chunks exist for all methods and datasets, False otherwise.
    """
    chunking_methods = ['gradient', 'interquartile', 'std_deviation', 'percentile']
    for method in chunking_methods:
        for domain in datasets.keys():
            chunk_path = f'data/chunks/{method}/{domain}'
            if not os.path.exists(chunk_path) or not os.listdir(chunk_path):
                return False
    return True

def load_existing_chunks(datasets):
    """
    Load existing chunks for all methods and datasets.
    
    Args:
        datasets (dict): Dictionary of datasets.
    
    Returns:
        dict: Dictionary of loaded chunks.
    """
    results = {}
    chunking_methods = ['Percentile', 'StdDeviation', 'Interquantile', 'Gradient']
    valid_domains = ['medical', 'scientific']
    for method in chunking_methods:
        results[method] = {}
        for domain in valid_domains:
            results[method][domain] = []
            chunk_path = f'data/chunks/{method.lower()}/{domain}'
            if os.path.exists(chunk_path):
                for chunk_file in os.listdir(chunk_path):
                    with open(os.path.join(chunk_path, chunk_file), 'r') as f:
                        chunks = json.load(f)
                    results[method][domain].extend(chunks)
            else:
                print(f"Warning: Chunk path {chunk_path} does not exist.")
    return results

def save_chunks(method_name, domain, doc_id, chunks):
    """
    Save chunks to a file.
    
    Args:
        method_name (str): Name of the chunking method.
        domain (str): Domain of the document.
        doc_id (str): ID of the document.
        chunks (list): List of chunks to save.
    """
    chunk_path = f'data/chunks/{method_name.lower()}/{domain}'
    os.makedirs(chunk_path, exist_ok=True)
    with open(os.path.join(chunk_path, f'{doc_id}_chunks.json'), 'w') as f:
        json.dump(chunks, f)

def main():
    try:
        # Configure logging for main
        logging.basicConfig(
            filename='logs/main.log',
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logger = logging.getLogger(__name__)
        logger.info("Starting Langchain Semantic Chunking Benchmark.")
        print("Starting Langchain Semantic Chunking Benchmark.")

        # Load configuration
        config = load_configuration()
        print(f"Configuration loaded: {config}")

        # Check if raw datasets exist
        raw_data_path = 'data/raw/'
        if not os.path.exists(raw_data_path) or not os.listdir(raw_data_path):
            # Download datasets
            print("Downloading datasets...")
            download_all_datasets(config)
            print("Datasets downloaded successfully.")
        else:
            print("Using existing raw datasets.")

        # Check if processed datasets exist
        if not os.path.exists('data/processed/') or not os.listdir('data/processed/'):
            # Process and store datasets
            print("Processing and storing datasets...")
            process_and_store_datasets()
            print("Datasets processed and stored.")
        else:
            print("Using existing processed datasets.")

        # Load preprocessed datasets
        print("Loading preprocessed datasets...")
        datasets = load_datasets('data/processed/')
        if not datasets:
            logger.warning("No datasets were loaded. Exiting the benchmark.")
            print("No datasets were loaded. Exiting the benchmark.")
            return
        print(f"Loaded datasets for domains: {list(datasets.keys())}")

        # Check if ground truths exist and are not empty
        ground_truth_path = 'config/ground_truths.json'
        if not os.path.exists(ground_truth_path) or os.path.getsize(ground_truth_path) == 0:
            # Create Ground Truths
            print("Generating ground truths using GPT-4...")
            create_ground_truths(datasets)
            print("Ground truths generated successfully.")
        else:
            print("Using existing ground truths.")

        # Load ground truths
        ground_truths = load_ground_truths()
        print("Ground truths loaded.")

        # Initialize HuggingFace Embeddings
        hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("HuggingFace Embeddings initialized.")

        # Check if chunks already exist
        if chunks_exist(datasets):
            print("Chunks already exist for all methods and datasets. Using existing chunks.")
            existing_chunks = load_existing_chunks(datasets)
        else:
            # Initialize chunking methods
            chunking_methods = {
                'Percentile': PercentileChunker(embeddings=hf_embeddings),
                'StdDeviation': StdDeviationChunker(embeddings=hf_embeddings),
                'Interquantile': InterquartileChunker(embeddings=hf_embeddings),
                'Gradient': GradientChunker(embeddings=hf_embeddings)
            }
            print(f"Initialized {len(chunking_methods)} chunking methods.")

            # Apply chunking methods
            existing_chunks = {}
            for method_name, chunker in chunking_methods.items():
                logger.info(f"Applying chunking method: {method_name}")
                print(f"Applying chunking method: {method_name}")
                existing_chunks[method_name] = {}
                for domain, documents in datasets.items():
                    print(f"  Processing domain: {domain}")
                    existing_chunks[method_name][domain] = []
                    for doc_id, text in documents.items():
                        source = 'pubmed' if 'pubmed' in doc_id.lower() else 'arxiv'
                        chunks = chunker.split_text(text, source=source, file_name=doc_id)
                        existing_chunks[method_name][domain].extend(chunks)
                        # Save chunks to file
                        save_chunks(method_name, domain, doc_id, chunks)
                logger.info(f"Completed chunking for method: {method_name}")
                print(f"Completed chunking for method: {method_name}")

        # Evaluate Chunk Sizes
        print("Evaluating chunk sizes...")
        chunk_size_metrics = evaluate_chunk_sizes()
        print("Chunk size evaluation completed.")

        # Evaluate Retrieval Quality
        print("Evaluating retrieval quality...")
        retrieval_metrics = evaluate_retrieval_quality(existing_chunks, ground_truths, embeddings=hf_embeddings)
        print("Retrieval quality evaluation completed.")

        # Calculate Scores
        print("Calculating scores...")
        scores = calculate_scores(chunk_size_metrics, retrieval_metrics)
        print("Score calculation completed.")

        # Save Results
        print("Saving results...")
        save_results(chunk_size_metrics, retrieval_metrics, scores)
        print("Results saved successfully.")

        logger.info("Completed Langchain Semantic Chunking Benchmark successfully.")
        print("Completed Langchain Semantic Chunking Benchmark successfully.")

    except Exception as e:
        logging.error(f"An error occurred during benchmarking: {e}")
        print(f"An error occurred during benchmarking: {e}")
        raise

if __name__ == "__main__":
    main()