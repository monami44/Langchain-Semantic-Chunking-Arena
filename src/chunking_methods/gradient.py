# src/chunking_methods/gradient.py

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import List, Dict
import os
import json
from pathlib import Path
import uuid

class GradientChunker:
    """
    Splits text into chunks based on gradient-based similarity thresholds.
    """
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        print(f"Initializing GradientChunker with embeddings: {embeddings}")
        self.chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="gradient",
        )
        print(f"SemanticChunker initialized: {self.chunker}")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        print(f"Logger initialized: {self.logger}")
        
        # Domain mapping to handle aliases
        self.domain_mapping = {
            'medical': 'pubmed',
            'financial': 'ecommerce',
            'social_media': 'history',
            'scientific': 'arxiv',
            'education': 'legal',
            'news': 'ecommerce',
            # Direct mappings
            'pubmed': 'pubmed',
            'arxiv': 'arxiv',
            'history': 'history',
            'legal': 'legal',
            'ecommerce': 'ecommerce'
        }
        
        # Create gradient directory and subdirectories if they don't exist
        self.gradient_dir = Path("chunks/gradient")
        self.domain_dirs = {
            'pubmed': self.gradient_dir / "pubmed",
            'arxiv': self.gradient_dir / "arxiv",
            'history': self.gradient_dir / "history",
            'legal': self.gradient_dir / "legal",
            'ecommerce': self.gradient_dir / "ecommerce"
        }
        
        # Create all directories
        self.gradient_dir.mkdir(parents=True, exist_ok=True)
        for dir_path in self.domain_dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        print(f"Gradient directory structure created/verified: {self.gradient_dir}")
        self.process_counter = 0
    
    def split_text(self, text: str, source: str, file_name: str) -> List[str]:
        # Map the source to its canonical name
        if source not in self.domain_mapping:
            raise ValueError(f"Source must be one of: {list(self.domain_mapping.keys())}")
        
        canonical_source = self.domain_mapping[source]
        
        self.process_counter += 1
        unique_id = self.generate_unique_id()
        print(f"Processing {canonical_source} document {self.process_counter}: {file_name}")
        
        try:
            self.logger.info(f"Starting GradientChunker.split_text for {canonical_source} document: {unique_id}")
            print("Attempting to create documents...")
            chunks = self.chunker.create_documents([text])
            print(f"Documents created. Number of chunks: {len(chunks)}")
            
            chunk_texts = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_texts.append(chunk.page_content)
                print(f"Chunk {i+1} length: {len(chunk.page_content)}")
            
            self.logger.info(f"GradientChunker created {len(chunk_texts)} chunks for {canonical_source} document: {unique_id}")
            print(f"Final number of chunks: {len(chunk_texts)}")
            
            print("Chunk lengths:")
            for i, chunk in enumerate(chunk_texts):
                print(f"Chunk {i+1}: {len(chunk)} characters")
            
            # Store chunks
            self.store_chunks(chunk_texts, canonical_source, unique_id, file_name)
            
            return chunk_texts
        except Exception as e:
            self.logger.error(f"Error in GradientChunker.split_text for {canonical_source} document {self.process_counter} ({file_name}): {e}")
            print(f"An error occurred: {e}")
            print("Returning original text as a single chunk.")
            return [text]

    def generate_unique_id(self) -> str:
        return str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID

    def store_chunks(self, chunks: List[str], source: str, unique_id: str, file_name: str):
        # Map the source to its canonical name
        canonical_source = self.domain_mapping[source]
        
        print(f"Storing chunks for {canonical_source} document {self.process_counter}: {file_name}")
        source_dir = self.domain_dirs[canonical_source]
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
            "source": canonical_source,
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

print("GradientChunker class defined successfully with support for PubMed, arXiv, history, legal, and ecommerce sources")
