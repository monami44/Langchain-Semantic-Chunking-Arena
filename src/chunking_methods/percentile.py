# src/chunking_methods/percentile.py

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import List, Dict
import os
import json
from pathlib import Path
import uuid

class PercentileChunker:
    """
    Splits text into chunks based on the specified percentile of sentence lengths.
    """
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        print(f"Initializing PercentileChunker with embeddings: {embeddings}")
        self.chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
        )
        print(f"SemanticChunker initialized: {self.chunker}")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        print(f"Logger initialized: {self.logger}")
        
        # Create percentile directory and subdirectories if they don't exist
        self.percentile_dir = Path("chunks/percentile")
        self.pubmed_dir = self.percentile_dir / "pubmed"
        self.arxiv_dir = self.percentile_dir / "arxiv"
        self.percentile_dir.mkdir(parents=True, exist_ok=True)
        self.pubmed_dir.mkdir(exist_ok=True)
        self.arxiv_dir.mkdir(exist_ok=True)
        print(f"Percentile directory structure created/verified: {self.percentile_dir}")
        
        self.process_counter = 0
    
    def split_text(self, text: str, source: str, file_name: str) -> List[str]:
        if source not in ['pubmed', 'arxiv']:
            raise ValueError("Source must be either 'pubmed' or 'arxiv'")
        
        self.process_counter += 1
        unique_id = self.generate_unique_id()
        print(f"Processing {source} document {self.process_counter}: {file_name}")
        
        try:
            self.logger.info(f"Starting PercentileChunker.split_text for {source} document: {unique_id}")
            print("Attempting to create documents...")
            chunks = self.chunker.create_documents([text])
            print(f"Documents created. Number of chunks: {len(chunks)}")
            
            chunk_texts = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_texts.append(chunk.page_content)
                print(f"Chunk {i+1} length: {len(chunk.page_content)}")
            
            self.logger.info(f"PercentileChunker created {len(chunk_texts)} chunks for {source} document: {unique_id}")
            print(f"Final number of chunks: {len(chunk_texts)}")
            
            print("Chunk lengths:")
            for i, chunk in enumerate(chunk_texts):
                print(f"Chunk {i+1}: {len(chunk)} characters")
            
            # Store chunks
            self.store_chunks(chunk_texts, source, unique_id, file_name)
            
            return chunk_texts
        except Exception as e:
            self.logger.error(f"Error in PercentileChunker.split_text for {source} document {self.process_counter} ({file_name}): {e}")
            print(f"An error occurred: {e}")
            print("Returning original text as a single chunk.")
            return [text]

    def generate_unique_id(self) -> str:
        return str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID

    def store_chunks(self, chunks: List[str], source: str, unique_id: str, file_name: str):
        print(f"Storing chunks for {source} document {self.process_counter}: {file_name}")
        source_dir = self.pubmed_dir if source == 'pubmed' else self.arxiv_dir
        document_dir = source_dir / f"{self.process_counter:04d}_{unique_id}"
        document_dir.mkdir(exist_ok=True)
        print(f"Document directory created: {document_dir}")
        
        for i, chunk in enumerate(chunks, 1):
            chunk_file = document_dir / f"chunk_{i:02d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            print(f"Chunk {i} stored in {chunk_file}")
        
        # Store metadata
        metadata = {
            "source": source,
            "unique_id": unique_id,
            "file_name": file_name,
            "process_order": self.process_counter,
            "num_chunks": len(chunks),
            "chunk_sizes": [len(chunk) for chunk in chunks]
        }
        metadata_file = document_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata stored in {metadata_file}")

# Add a print statement at the end of the file
print("PercentileChunker class defined successfully with support for PubMed and arXiv sources")
