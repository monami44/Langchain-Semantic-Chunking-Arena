# src/utils/preprocessor.py

import re
import logging
import os
from tqdm import tqdm
from typing import Dict

# Configure Logging
logging.basicConfig(
    filename='logs/preprocessor.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses the input text.
    
    Steps:
        - Remove non-alphanumeric characters (except spaces and basic punctuation).
        - Replace multiple spaces with a single space.
        - Remove leading/trailing whitespace.
    
    Args:
        text (str): The raw text to preprocess.
    
    Returns:
        str: The preprocessed text.
    """
    try:
        logger.debug("Starting text preprocessing.")
        # Remove non-alphanumeric characters (except spaces and basic punctuation)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        logger.debug("Completed text preprocessing.")
        return text
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return ""

def process_and_store_datasets(raw_data_path: str = 'data/raw/',
                              processed_data_path: str = 'data/processed/',
                              min_word_count: int = 100):
    """
    Processes raw datasets and stores them in the processed folder.
    Ensures each processed document has at least a minimum number of words.
    
    Args:
        raw_data_path (str): Path to the raw data folder.
        processed_data_path (str): Path to store processed data.
        min_word_count (int): Minimum number of words required in processed text.
    """
    try:
        logger.info(f"Starting processing of datasets from '{raw_data_path}'.")
        os.makedirs(processed_data_path, exist_ok=True)

        for domain in os.listdir(raw_data_path):
            domain_path = os.path.join(raw_data_path, domain)
            if os.path.isdir(domain_path):
                processed_domain_path = os.path.join(processed_data_path, domain)
                os.makedirs(processed_domain_path, exist_ok=True)
                
                for file_name in tqdm(os.listdir(domain_path), desc=f"Processing {domain} dataset"):
                    file_path = os.path.join(domain_path, file_name)
                    if os.path.isfile(file_path) and file_name.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        # Preprocess the text
                        processed_text = preprocess_text(text)
                        
                        # Ensure the processed text has at least min_word_count words
                        if len(processed_text.split()) >= min_word_count:
                            processed_file_path = os.path.join(processed_domain_path, file_name)
                            with open(processed_file_path, 'w', encoding='utf-8') as f:
                                f.write(processed_text)
                        else:
                            logger.warning(f"Skipped '{file_name}' in '{domain}' dataset: insufficient word count ({len(processed_text.split())} words).")
        logger.info("Completed processing of datasets.")
    except Exception as e:
        logger.error(f"Error in process_and_store_datasets: {e}")
