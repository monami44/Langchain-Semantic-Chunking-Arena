import os
import json
import logging
import requests
import arxiv
import pandas as pd
from tqdm import tqdm
import time

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
    Orchestrates the download of all datasets based on the provided configuration.
    """
    logger.info("Starting download of all datasets.")
    
    # Wikipedia Articles
    wiki_config = config.get('wikipedia', {})
    if wiki_config.get('num_articles'):
        try:
            download_wikipedia_articles(wiki_config['num_articles'])
        except Exception as e:
            logger.error(f"Error downloading Wikipedia articles: {e}")
    
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

    # OpenAlex Articles
    openalex_config = config.get('openalex', {})
    for domain, settings in openalex_config.items():
        try:
            download_openalex_articles(
                domain=domain,
                search_query=settings['search_query'],
                max_results=settings.get('max_results', 100)
            )
        except Exception as e:
            logger.error(f"Error downloading OpenAlex {domain} articles: {e}")

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

    logger.info("Completed download of all datasets.")

def download_openalex_articles(domain, search_query, max_results=100):
    """
    Downloads articles from OpenAlex API for a specific domain.
    
    Args:
        domain (str): Domain name (e.g., 'legal', 'history', 'ecommerce')
        search_query (str): Search query for the domain
        max_results (int): Maximum number of articles to download
    """
    try:
        logger.info(f"Starting download of OpenAlex articles for domain: {domain}")
        
        BASE_URL = 'https://api.openalex.org/works'
        params = {
            'search': search_query,
            'filter': 'is_oa:true,publication_year:2010-2024,has_abstract:true',
            'per-page': 25,
            'mailto': 'lyaminky2@gmail.com'
        }

        save_dir = f'data/raw/{domain}'
        os.makedirs(save_dir, exist_ok=True)

        saved_count = 0
        page = 1

        while saved_count < max_results:
            params['page'] = page
            response = requests.get(BASE_URL, params=params)

            if response.status_code == 200:
                data = response.json()
                works = data.get('results', [])

                if not works:
                    logger.warning(f"No more works found for domain: {domain}")
                    break

                for work in works:
                    if saved_count >= max_results:
                        break

                    abstract_inverted_index = work.get('abstract_inverted_index', None)
                    if abstract_inverted_index:
                        abstract_text = work.get('abstract')
                        if not abstract_text:
                            word_positions = []
                            for word, positions in abstract_inverted_index.items():
                                for pos in positions:
                                    word_positions.append((pos, word))
                            
                            sorted_words = [word for _, word in sorted(word_positions)]
                            cleaned_words = []
                            for word in sorted_words:
                                if not cleaned_words or word != cleaned_words[-1]:
                                    cleaned_words.append(word)
                            
                            abstract_text = ' '.join(cleaned_words)

                        filename = f'openalex_{domain}_{saved_count + 1}.txt'
                        file_path = os.path.join(save_dir, filename)

                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write(abstract_text)

                        saved_count += 1
                        logger.info(f"Saved {filename}")

                page += 1
                time.sleep(1)
            else:
                logger.error(f"Error {response.status_code} downloading OpenAlex articles")
                break

        logger.info(f"Total {domain} articles saved: {saved_count}")
    except Exception as e:
        logger.error(f"Error downloading OpenAlex articles for {domain}: {e}")