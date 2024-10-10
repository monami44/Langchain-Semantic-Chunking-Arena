import os
import json
import logging
import requests
import git
import wikipediaapi
from Bio import Entrez
import arxiv
import pandas as pd
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    filename='logs/data_loader.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def sanitize_filename(name):
    """Sanitize the filename by replacing invalid characters."""
    return "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in name).rstrip()

def download_wikipedia_articles(num_articles=100):
    """
    Downloads a specified number of Wikipedia articles across selected categories.
    """
    try:
        logger.info("Starting download of Wikipedia articles.")
        wiki_wiki = wikipediaapi.Wikipedia('en')
        articles = {}
        categories = ['Technology', 'Health', 'Finance', 'Education', 'Legal', 'Science']

        for category in categories:
            cat = wiki_wiki.page(f"Category:{category}")
            if not cat.exists():
                logger.warning(f"Category '{category}' does not exist in Wikipedia.")
                continue
            for c in tqdm(cat.categorymembers.values(), desc=f"Fetching category: {category}"):
                if c.ns == wikipediaapi.Namespace.MAIN and len(articles) < num_articles:
                    articles[c.title] = c.text
                    if len(articles) >= num_articles:
                        break

        # Save raw articles
        os.makedirs('data/raw/articles/', exist_ok=True)
        for title, text in articles.items():
            file_path = os.path.join('data/raw/articles/', f"{sanitize_filename(title)}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
        logger.info(f"Downloaded and saved {len(articles)} Wikipedia articles.")
    except Exception as e:
        logger.error(f"Error downloading Wikipedia articles: {e}")

def download_pubmed_articles(email, search_term, max_results=100):
    """
    Downloads PubMed articles based on a search term.
    """
    try:
        logger.info(f"Starting download of PubMed articles with search term '{search_term}'.")
        Entrez.email = email
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
        record = Entrez.read(handle)
        id_list = record['IdList']
        handle.close()

        if not id_list:
            logger.warning("No PubMed articles found for the given search term.")
            return

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
        abstracts = handle.read().split('\n\n\n')
        handle.close()

        # Save raw medical articles
        os.makedirs('data/raw/medical/', exist_ok=True)
        saved_count = 0
        for abstract in abstracts:
            if abstract.strip() and saved_count < max_results:
                file_path = os.path.join('data/raw/medical/', f"pubmed_{saved_count}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(abstract.strip())
                saved_count += 1
            if saved_count >= max_results:
                break
        logger.info(f"Downloaded and saved {saved_count} PubMed articles.")
    except Exception as e:
        logger.error(f"Error downloading PubMed articles: {e}")

def clone_repository(repo_url, destination):
    """
    Clones a git repository to the specified destination.
    """
    try:
        logger.info(f"Cloning repository '{repo_url}' into '{destination}'.")
        git.Repo.clone_from(repo_url, destination)
        logger.info(f"Successfully cloned '{repo_url}'.")
    except git.exc.GitError as e:
        logger.error(f"Git error while cloning '{repo_url}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error while cloning '{repo_url}': {e}")

def download_technical_repositories(repos, destination_base='data/raw/technical/'):
    """
    Downloads multiple technical repositories.
    """
    os.makedirs(destination_base, exist_ok=True)
    for repo in repos:
        repo_name = repo.split('/')[-1].replace('.git', '')
        destination = os.path.join(destination_base, repo_name)
        if os.path.exists(destination):
            logger.info(f"Repository '{repo_name}' already exists. Skipping clone.")
            continue
        clone_repository(repo, destination)

def download_legal_documents(query, max_results=100):
    """
    Downloads legal documents based on a query using the CourtListener API.
    """
    try:
        logger.info(f"Starting download of legal documents with query '{query}'.")
        url = "https://www.courtlistener.com/api/rest/v3/opinions/"
        params = {
            'q': query,
            'page_size': max_results
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        os.makedirs('data/raw/legal/', exist_ok=True)
        saved_count = 0
        for idx, opinion in enumerate(data.get('results', [])):
            text = opinion.get('text', '')
            if text:
                file_path = os.path.join('data/raw/legal/', f"opinion_{idx}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                saved_count += 1
        logger.info(f"Downloaded and saved {saved_count} legal documents.")
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error while downloading legal documents: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while downloading legal documents: {e}")

def download_arxiv_papers(query, max_results=100):
    """
    Downloads arXiv papers based on a query.
    """
    try:
        logger.info(f"Starting download of arXiv papers with query '{query}'.")
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        os.makedirs('data/raw/scientific/', exist_ok=True)
        saved_count = 0
        for result in tqdm(search.results(), desc="Downloading arXiv papers", total=max_results):
            if saved_count >= max_results:
                break
            file_path = os.path.join('data/raw/scientific/', f"arxiv_{saved_count}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(result.summary.strip())
            saved_count += 1
        logger.info(f"Downloaded and saved {saved_count} arXiv papers.")
    except Exception as e:
        logger.error(f"Error downloading arXiv papers: {e}")

def load_datasets(processed_data_path='data/processed/'):
    """
    Loads processed datasets from the specified directory.
    
    Returns:
        dict: A dictionary with domain names as keys and another dictionary of document_id: text as values.
    """
    try:
        logger.info(f"Loading datasets from '{processed_data_path}'.")
        datasets = {}
        for domain in os.listdir(processed_data_path):
            domain_path = os.path.join(processed_data_path, domain)
            if os.path.isdir(domain_path):
                datasets[domain] = {}
                for file_name in tqdm(os.listdir(domain_path), desc=f"Loading domain: {domain}"):
                    file_path = os.path.join(domain_path, file_name)
                    if os.path.isfile(file_path) and file_name.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            datasets[domain][file_name] = text
        logger.info(f"Successfully loaded datasets from '{processed_data_path}'.")
        return datasets
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return {}

def download_all_datasets(config):
    """
    Orchestrates the download of medical and scientific datasets based on the provided configuration.
    """
    logger.info("Starting download of medical and scientific datasets.")
    
    # PubMed Articles
    pubmed_config = config.get('pubmed', {})
    if pubmed_config.get('email') and pubmed_config.get('search_term'):
        try:
            download_pubmed_articles(
                email=pubmed_config['email'],
                search_term=pubmed_config['search_term'],
                max_results=pubmed_config.get('max_results', 100)
            )
        except Exception as e:
            logger.error(f"Error downloading PubMed articles: {e}")
    else:
        logger.warning("PubMed configuration incomplete. Skipping PubMed articles download.")

    # arXiv Papers
    arxiv_config = config.get('arxiv', {})
    if arxiv_config.get('query'):
        try:
            download_arxiv_papers(
                query=arxiv_config['query'],
                max_results=arxiv_config.get('max_results', 100)
            )
        except Exception as e:
            logger.error(f"Error downloading arXiv papers: {e}")
    else:
        logger.warning("arXiv configuration incomplete. Skipping arXiv papers download.")

    logger.info("Completed download of medical and scientific datasets.")