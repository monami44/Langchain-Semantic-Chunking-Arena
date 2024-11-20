# src/evaluation/retrieval_quality_evaluator.py

import logging
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import ndcg_score
import numpy as np
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import json

# Configure Logging
logging.basicConfig(
    filename='logs/retrieval_quality_evaluator.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def evaluate_retrieval_quality(results: Dict[str, Dict[str, List[Dict]]],
                               ground_truths: Dict[str, Dict[str, Dict[str, List[str]]]],
                               embeddings: HuggingFaceEmbeddings,
                               top_k: int = 10):
    print("Starting evaluate_retrieval_quality function")
    try:
        logger.info("Starting evaluation of retrieval quality.")
        metrics = {}
        
        # Define all domains
        domains = ['medical', 'scientific', 'legal', 'history', 'ecommerce']
        
        print(f"Results keys: {results.keys()}")
        print(f"Ground truths keys: {ground_truths.keys()}")
        
        for method, method_results in results.items():
            print(f"Processing method: {method}")
            metrics[method] = {}
            
            for domain in domains:
                if domain not in method_results or domain not in ground_truths:
                    print(f"Domain '{domain}' not found in results or ground truths for method '{method}'. Skipping.")
                    logger.warning(f"Skipping domain '{domain}' for method '{method}' - data not found")
                    continue

                chunks = method_results[domain]
                print(f"Processing domain: {domain}")
                logger.info(f"Evaluating method: {method}, domain: {domain}")
                
                # Encode all chunks once for efficiency
                print("Encoding chunks...")
                chunk_texts = []
                chunk_metadata = []
                for chunk in chunks:
                    if isinstance(chunk, dict) and 'text' in chunk:
                        chunk_texts.append(chunk['text'])
                        chunk_metadata.append(chunk.get('metadata', {}))
                    elif isinstance(chunk, str):
                        chunk_texts.append(chunk)
                        chunk_metadata.append({})  # Empty metadata for string chunks
                    else:
                        print(f"Unexpected chunk format: {type(chunk)}")
                        continue

                chunk_embeddings = embeddings.embed_documents(chunk_texts)
                print(f"Chunk embeddings shape: {len(chunk_embeddings)}, {len(chunk_embeddings[0])}")
                
                # Encode ground truth queries
                print("Processing ground truth queries...")
                queries = ground_truths[domain]['queries']
                relevant_chunks = ground_truths[domain]['relevant_chunks']
                
                print(f"Number of queries for {domain}: {len(queries)}")
                print(f"First query for {domain}: {queries[0] if queries else 'No queries'}")
                print(f"Number of relevant chunks for {domain}: {len(relevant_chunks)}")
                
                method_metrics = {
                    'precision': [],
                    'recall': [],
                    'f1_score': [],
                    'average_precision': [],
                    'ndcg': []
                }
                
                for i, query in enumerate(queries):
                    print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")  # Print first 50 chars of query
                    query_embedding = embeddings.embed_query(query)
                    print(f"Query embedding shape: {len(query_embedding)}")
                    
                    # Compute cosine similarity
                    print("Computing cosine similarity...")
                    similarities = cosine_similarity(query_embedding, chunk_embeddings)
                    print(f"Similarities shape: {len(similarities)}")
                    
                    # Get top_k indices
                    top_indices = similarities.argsort()[-top_k:][::-1]
                    retrieved_chunks = [chunks[i] for i in top_indices]
                    print(f"Number of retrieved chunks: {len(retrieved_chunks)}")
                    
                    # Get relevant chunks for this query
                    query_relevant_chunks = relevant_chunks.get(query, {'chunks': [], 'metadata': {'file_name': ''}})
                    query_relevant_file = query_relevant_chunks['metadata']['file_name']

                    # Binary relevance for precision, recall, f1
                    y_true = []
                    for chunk, metadata in zip(retrieved_chunks, [chunk_metadata[i] for i in top_indices]):
                        if isinstance(chunk, dict):
                            chunk_text = chunk.get('text', '')
                            chunk_file = metadata.get('file_name', '')
                        else:  # chunk is a string
                            chunk_text = chunk
                            chunk_file = ''

                        # Updated relevance condition
                        is_relevant = (chunk_file == query_relevant_file) or any(gt_chunk in chunk_text for gt_chunk in query_relevant_chunks['chunks'])
                        y_true.append(1 if is_relevant else 0)

                    y_pred = [1] * len(retrieved_chunks)  # Retrieved chunks are considered as positive predictions
                    
                    print(f"y_true: {y_true}")
                    print(f"y_pred: {y_pred}")
                    
                    # Calculate Precision, Recall, F1
                    print("Calculating Precision, Recall, F1...")
                    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                    
                    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
                    
                    # Average Precision
                    print("Calculating Average Precision...")
                    ap = average_precision_score(y_true, similarities[top_indices])
                    print(f"Average Precision: {ap}")
                    
                    # NDCG
                    print("Calculating NDCG...")
                    relevance = []
                    for chunk, metadata in zip(retrieved_chunks, [chunk_metadata[i] for i in top_indices]):
                        if isinstance(chunk, dict):
                            chunk_text = chunk.get('text', '')
                            chunk_file = metadata.get('file_name', '')
                        else:  # chunk is a string
                            chunk_text = chunk
                            chunk_file = ''

                        is_relevant = (chunk_file == query_relevant_file) or any(gt_chunk in chunk_text for gt_chunk in query_relevant_chunks['chunks'])
                        relevance.append(1 if is_relevant else 0)

                    ndcg = ndcg_score([relevance], [similarities[top_indices]])
                    print(f"NDCG: {ndcg}")
                    
                    method_metrics['precision'].append(precision)
                    method_metrics['recall'].append(recall)
                    method_metrics['f1_score'].append(f1)
                    method_metrics['average_precision'].append(ap)
                    method_metrics['ndcg'].append(ndcg)
                
                # Aggregate metrics
                print("Aggregating metrics...")
                metrics[method][domain] = {
                    'precision': np.mean(method_metrics['precision']),
                    'recall': np.mean(method_metrics['recall']),
                    'f1_score': np.mean(method_metrics['f1_score']),
                    'average_precision': np.mean(method_metrics['average_precision']),
                    'ndcg': np.mean(method_metrics['ndcg'])
                }
                print(f"Aggregated metrics for {method}, {domain}: {metrics[method][domain]}")
                
        logger.info("Completed evaluation of retrieval quality.")
        print("Completed evaluation of retrieval quality.")
        print(f"Final metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error in evaluating retrieval quality: {e}")
        print(f"Error in evaluating retrieval quality: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}

def cosine_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a query embedding and a set of chunk embeddings.
    
    Args:
        query_embedding (np.ndarray): Embedding vector for the query.
        chunk_embeddings (np.ndarray): Embedding vectors for the chunks.
        
    Returns:
        np.ndarray: Cosine similarity scores.
    """
    query_norm = np.linalg.norm(query_embedding)
    chunks_norm = np.linalg.norm(chunk_embeddings, axis=1)
    similarity = np.dot(chunk_embeddings, query_embedding) / (chunks_norm * query_norm + 1e-10)
    return similarity

def plot_retrieval_quality_metrics(retrieval_metrics_file: str):
    print("Starting plot_retrieval_quality_metrics function")
    try:
        with open(retrieval_metrics_file, 'r') as f:
            retrieval_metrics = json.load(f)

        methods = list(retrieval_metrics.keys())
        domains = list(retrieval_metrics[methods[0]].keys())
        metrics = ['precision', 'recall', 'f1_score', 'average_precision', 'ndcg']

        fig, axs = plt.subplots(len(domains), 1, figsize=(12, 6*len(domains)))
        fig.suptitle('Retrieval Quality Metrics Comparison')

        for i, domain in enumerate(domains):
            ax = axs[i] if len(domains) > 1 else axs
            x = range(len(methods))
            width = 0.15
            
            for j, metric in enumerate(metrics):
                values = [retrieval_metrics[method][domain][metric] for method in methods]
                ax.bar([xi + j*width for xi in x], values, width, label=metric)

            ax.set_ylabel('Score')
            ax.set_title(f'{domain.capitalize()} Domain')
            ax.set_xticks([xi + width*2 for xi in x])
            ax.set_xticklabels(methods)
            ax.legend()

        plt.tight_layout()
        plt.savefig('results/retrieval_quality_comparison.png')
        print("Retrieval quality metrics plot saved as 'results/retrieval_quality_comparison.png'")
    except Exception as e:
        print(f"Error in plotting retrieval quality metrics: {e}")
        logger.error(f"Error in plotting retrieval quality metrics: {e}")


