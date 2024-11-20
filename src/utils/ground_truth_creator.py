# src/utils/ground_truth_generator.py

import os
import json
import logging
import re
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAIError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import AzureOpenAI

# Load environment variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

print("Environment variables loaded:")
print(f"AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT}")
print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {AZURE_OPENAI_DEPLOYMENT_NAME}")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
    raise ValueError("One or more required environment variables are missing. Please check your .env file.")

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-03-15-preview"
)

print("AzureOpenAI client initialized successfully")

# Configure Logging
logging.basicConfig(
    filename='logs/ground_truth_generator.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize OpenAI API for Azure

# Define Pydantic Models for Structured Outputs
class QuestionResponse(BaseModel):
    question: str

class ChunksResponse(BaseModel):
    chunks: List[str]

class GroundTruthEntry(BaseModel):
    question: str
    chunks: List[str]

# Retry Configuration for OpenAI API Calls
retry_strategy = retry(
    retry=retry_if_exception_type(OpenAIError),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)

print("Starting ground truth creator script")

print(f"OpenAI API configured with endpoint: {AZURE_OPENAI_ENDPOINT}")

@retry_strategy
def generate_question(text: str) -> str:
    print(f"Generating question for text of length: {len(text)}")
    try:
        prompt = (
            "Read the following text and generate an insightful and relevant question "
            "that covers a key point discussed in the text. The question should be suitable "
            "for assessing comprehension of the material. Return the question in a JSON format "
            "with the key 'question' mapping to the question string.\n\n"
            f"Text:\n{text}\n\n"
            "Please provide the question in the following JSON format:\n"
            "{\n"
            '  "question": "Your question here."\n'
            "}"
        )

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,  # Use the deployment name as the model
            messages=[
                {"role": "system", "content": "You are an expert in generating comprehension questions from academic texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
            n=1,
            stop=None
        )

        response_text = response.choices[0].message.content
        question = parse_structured_output(response_text, key="question", model=QuestionResponse)
        print(f"Generated question: {question}")
        return question
    except OpenAIError as e:
        print(f"OpenAI API error in generate_question: {e}")
        logger.error(f"OpenAI API error in generate_question: {e}")
        raise e
    except Exception as e:
        print(f"Unexpected error in generate_question: {e}")
        logger.error(f"Unexpected error in generate_question: {e}")
        return ""

@retry_strategy
def find_relevant_chunks(question: str, text: str, top_k: int = 3) -> List[str]:
    print(f"Finding relevant chunks for question: {question}")
    try:
        prompt = (
            "Read the following text and answer the question by providing the most relevant excerpts.\n\n"
            f"Text:\n{text}\n\n"
            f"Question: {question}\n\n"
            f"Please provide the top {top_k} relevant excerpts (chunks) from the text that answer the question. "
            "Each excerpt should be no longer than 300 words and should include enough context to understand the answer. "
            "Return the chunks in a JSON format with the key 'chunks' mapping to a list of excerpts.\n\n"
            "Example format:\n"
            "{\n"
            '  "chunks": [\n'
            '    "Excerpt 1...",\n'
            '    "Excerpt 2...",\n'
            '    "Excerpt 3..."\n'
            '  ]\n'
            "}"
        )

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,  # Use the deployment name as the model
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in information retrieval and text analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000,
            n=1,
            stop=None
        )

        response_text = response.choices[0].message.content
        chunks = parse_structured_output(response_text, key="chunks", model=ChunksResponse)
        print(f"Found {len(chunks)} relevant chunks")
        return chunks
    except OpenAIError as e:
        print(f"OpenAI API error in find_relevant_chunks: {e}")
        logger.error(f"OpenAI API error in find_relevant_chunks: {e}")
        raise e
    except Exception as e:
        print(f"Unexpected error in find_relevant_chunks: {e}")
        logger.error(f"Unexpected error in find_relevant_chunks: {e}")
        return []

def parse_structured_output(response_text: str, key: str, model: BaseModel) -> List[str]:
    print(f"Parsing structured output for key: {key}")
    try:
        # Extract JSON content from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            logger.error("No JSON structure found in the response.")
            return []
        json_text = json_match.group()
        data = json.loads(json_text)

        # Parse the JSON using the provided Pydantic model
        parsed = model(**data)
        item = getattr(parsed, key, "")

        # Ensure the output is a list
        if isinstance(item, list):
            return item
        elif isinstance(item, str):
            return [item]
        else:
            logger.warning(f"The key '{key}' does not map to a string or list in the structured output.")
            return []
    except json.JSONDecodeError as e:
        print(f"JSON decoding error for key '{key}': {e}")
        logger.error(f"JSON decoding error for key '{key}': {e}")
        return []
    except Exception as e:
        print(f"Error parsing structured output for key '{key}': {e}")
        logger.error(f"Error parsing structured output for key '{key}': {e}")
        return []

def process_document(domain: str, doc_id: str, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Processing document: {doc_id} in domain: {domain}")
    try:
        question = generate_question(text)
        if not question:
            print(f"No question generated for document '{doc_id}' in domain '{domain}'.")
            logger.warning(f"No question generated for document '{doc_id}' in domain '{domain}'.")
            return {}

        print(f"Generated question: {question}")
        chunks = find_relevant_chunks(question, text, top_k=3)
        if not chunks:
            print(f"No relevant chunks found for question '{question}' in document '{doc_id}' in domain '{domain}'.")
            logger.warning(f"No relevant chunks found for question '{question}' in document '{doc_id}' in domain '{domain}'.")
            return {}

        print(f"Found {len(chunks)} relevant chunks for document {doc_id}")
        return {
            "question": question,
            "chunks": chunks,
            "metadata": metadata
        }
    except OpenAIError as e:
        print(f"OpenAI API error processing document '{doc_id}' in domain '{domain}': {e}")
        logger.error(f"OpenAI API error processing document '{doc_id}' in domain '{domain}': {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error processing document '{doc_id}' in domain '{domain}': {e}")
        logger.error(f"Unexpected error processing document '{doc_id}' in domain '{domain}': {e}")
        return {}

def create_ground_truths(datasets: Dict[str, Dict[str, str]],
                         output_path: str = 'config/ground_truths.json'):
    print("Starting ground truth creation process")
    ground_truths = {}
    try:
        # Updated domains list to include all domains
        domains = ['medical', 'scientific', 'history', 'legal', 'ecommerce']
        for domain in domains:
            if domain not in datasets:
                print(f"Domain '{domain}' not found in datasets. Skipping.")
                logger.warning(f"Domain '{domain}' not found in datasets. Skipping.")
                continue
            
            ground_truths[domain] = {
                "queries": [],
                "relevant_chunks": {}
            }
            print(f"Generating ground truths for domain: {domain}")
            logger.info(f"Generating ground truths for domain: {domain}")
            
            successful_docs = 0
            total_docs = len(datasets[domain])
            
            for doc_id, text in tqdm(datasets[domain].items(), desc=f"Processing {domain} documents"):
                try:
                    print(f"Processing document {doc_id} in {domain} domain")
                    # Updated source detection
                    source = 'openalex' if any(x in doc_id.lower() for x in ['history', 'legal', 'ecommerce']) else \
                            'pubmed' if 'pubmed' in doc_id.lower() else 'arxiv'
                    metadata = {"source": source, "file_name": doc_id}
                    
                    result = process_document(domain, doc_id, text, metadata)
                    if not result:
                        print(f"No result for document {doc_id}. Skipping.")
                        continue
                        
                    question = result["question"]
                    chunks = result["chunks"]
                    metadata = result["metadata"]
                    
                    # Ensure question is a string, not a list
                    if isinstance(question, list):
                        question = question[0] if question else ""
                        
                    ground_truths[domain]["queries"].append(question)
                    ground_truths[domain]["relevant_chunks"][question] = {
                        "chunks": chunks,
                        "metadata": metadata
                    }
                    print(f"Added question, chunks, and metadata for document {doc_id}")
                    successful_docs += 1
                    
                    # Save progress after each successful document
                    if successful_docs % 5 == 0:  # Save every 5 successful documents
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(ground_truths, f, indent=4)
                        print(f"Progress saved: {successful_docs}/{total_docs} documents processed in {domain}")
                        
                except Exception as e:
                    print(f"Error processing document {doc_id} in {domain}: {e}")
                    logger.error(f"Error processing document {doc_id} in {domain}: {e}")
                    continue
            
            print(f"Completed {domain} domain: {successful_docs}/{total_docs} documents processed successfully")
            logger.info(f"Completed {domain} domain: {successful_docs}/{total_docs} documents processed successfully")
            
            # Save after completing each domain
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ground_truths, f, indent=4)
            print(f"Ground truths saved after completing {domain} domain")
            
    except Exception as e:
        print(f"Error in create_ground_truths: {e}")
        logger.error(f"Error in create_ground_truths: {e}")
        # Save whatever progress we have even if there's an error
        if ground_truths:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ground_truths, f, indent=4)
            print(f"Partial ground truths saved due to error")
    
    print("Ground truth creator script completed")
    return ground_truths

print("Ground truth creator script completed")