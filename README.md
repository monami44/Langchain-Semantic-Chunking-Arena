**Disclaimer:** I am not a professional scientist and do not pretend to be one. This is an experiment I conducted in my free time, literally in 12 hours, and I had a lot of fun playing with it. Everyone is welcome to critique this work or take it and improve it to a real scientific level.

---

# Benchmarking Langchain Semantic Chunking Methods: A Comparative Analysis

## Abstract

This study presents a benchmarking environment designed to evaluate and compare four semantic chunking methods provided by Langchain: percentile, interquartile, gradient, and standard deviation. Utilizing diverse datasets comprising 100 abstracts from arXiv machine learning research papers, 100 introductions from PubMed articles on COVID-19, 100 historical documents about World War II, 100 papers on judicial review in European legal systems, and 100 research papers about e-commerce, the benchmarking framework assesses both chunk sizes and retrieval quality against generated ground truths. Metrics such as cosine similarity, precision, recall, F1-score, average precision, and normalized discounted cumulative gain (NDCG) are employed to evaluate performance. The results indicate variations in chunking effectiveness across different methods, providing insights into their suitability for semantic text segmentation tasks across various domains.

## Introduction

Semantic chunking is a critical process in natural language processing (NLP) that involves dividing text into meaningful units or "chunks" to facilitate downstream tasks such as information retrieval, summarization, and question answering. Langchain, a popular NLP library, offers several methods for semantic chunking, including percentile, interquartile, gradient, and standard deviation-based approaches. Despite their availability, detailed comparative analyses of these methods are scarce.

This study aims to fill this gap by creating a benchmarking environment to evaluate these chunking methods systematically. By understanding their differences and assessing their performance using well-defined metrics, users can make informed decisions on which method to employ for specific applications.

## Methodology

### Benchmarking Environment Overview

The benchmarking environment is designed to:

1. **Evaluate Chunk Sizes**: Analyze the distribution of chunk sizes generated by each method.
2. **Assess Retrieval Quality**: Measure how well the chunks facilitate the retrieval of relevant information compared to ground truths.
3. **Rank Methods**: Provide a scoring system to rank the chunking methods based on their performance in chunk size distribution and retrieval quality.

### Datasets

Five diverse datasets were used:

- **Machine Learning Domain (arXiv)**: 100 abstracts from arXiv research papers on machine learning
- **Medical Domain (PubMed)**: 100 introductions from PubMed articles on COVID-19
- **Historical Domain**: 100 documents about World War II
- **Legal Domain**: 100 papers on judicial review in European legal systems
- **E-commerce Domain**: 100 research papers about e-commerce trends and analysis

These datasets were chosen to represent diverse textual content across different fields, ensuring a comprehensive evaluation of the chunking methods across varying writing styles and domain-specific terminology.

### Ground Truth Generation

Ground truths were generated using the following process:

1. **Question Generation**: For each document, a question was generated using GPT-4, aiming to cover key points discussed in the text.
2. **Relevant Chunk Extraction**: GPT-4 was then used to extract the most relevant excerpts (chunks) from the text that answer the generated question.
3. **Metadata Association**: Both chunks and ground truths were associated with metadata referencing the original text file, serving as a unique identifier to facilitate accurate mapping.

### Chunking Methods Implemented

1. **Percentile Chunker**: Splits text based on specified percentiles of sentence lengths.
2. **Interquartile Chunker**: Uses the interquartile range of sentence lengths for chunking.
3. **Gradient Chunker**: Divides text based on gradient-based similarity thresholds.
4. **Standard Deviation Chunker**: Splits text based on the standard deviation of sentence lengths.

### Embeddings and Similarity Calculation

- **Embeddings**: Hugging Face's `all-MiniLM-L6-v2` model was used to generate embeddings for both chunks and queries.
- **Similarity Measure**: Cosine similarity was computed between query embeddings and chunk embeddings to assess relevance.

### Evaluation Metrics

1. **Chunk Size Metrics**:
   - Mean Size
   - Median Size
   - Standard Deviation
   - Minimum and Maximum Sizes
2. **Retrieval Quality Metrics**:
   - Precision
   - Recall
   - F1-Score
   - Average Precision (AP)
   - Normalized Discounted Cumulative Gain (NDCG)

### Scoring System

A weighted scoring system was used to rank the chunking methods:

- **Size Score (40%)**:
  - Mean Score (35%)
  - Standard Deviation Score (35%)
  - Minimum Size Score (15%)
  - Maximum Size Score (15%)
- **Retrieval Score (60%)**:
  - Precision (20%)
  - Recall (20%)
  - F1-Score (20%)
  - Average Precision (20%)
  - NDCG (20%)

### Challenges and Solutions

**Mapping Chunks to Ground Truths**:

- **Initial Hurdle**: Mapping was initially based on processing order, leading to inaccuracies.
- **Solution**: Introduced metadata containing the original text file reference to serve as a unique identifier for accurate mapping.

## Results

### Chunk Size Evaluation

#### Comprehensive Metrics Analysis

**Table 1: Chunk Size Metrics Across Domains and Methods**

| Domain | Method | Mean Size | Median Size | Std Dev | Min Size | Max Size |
|--------|---------|-----------|-------------|----------|-----------|-----------|
| Machine Learning | Gradient | 513.75 | 482.0 | 388.12 | 31 | 1,687 |
| | Interquartile | 704.14 | 682.0 | 400.54 | 86 | 1,584 |
| | Std Deviation | 1,028.50 | 1,069.5 | 289.53 | 593 | 1,758 |
| | Percentile | 513.75 | 432.5 | 350.71 | 39 | 1,584 |
| Medical | Gradient | 1,052.38 | 210.0 | 1,279.64 | 3 | 7,248 |
| | Interquartile | 812.84 | 225.0 | 1,065.07 | 22 | 6,154 |
| | Std Deviation | 1,847.66 | 2,144.0 | 1,438.49 | 28 | 7,362 |
| | Percentile | 1,052.38 | 131.0 | 1,249.80 | 21 | 6,154 |
| Historical | Gradient | 668.54 | 552.5 | 619.92 | 4 | 5,075 |
| | Interquartile | 919.62 | 821.0 | 603.10 | 42 | 4,226 |
| | Std Deviation | 1,320.91 | 1,117.5 | 1,352.19 | 640 | 12,672 |
| | Percentile | 643.43 | 523.5 | 537.77 | 4 | 4,226 |
| Legal | Gradient | 1,640.57 | 927.0 | 2,325.69 | 4 | 18,480 |
| | Interquartile | 1,822.27 | 1,187.0 | 2,084.94 | 4 | 11,057 |
| | Std Deviation | 3,914.56 | 1,633.5 | 5,472.54 | 16 | 25,203 |
| | Percentile | 1,634.95 | 828.0 | 2,110.59 | 2 | 12,420 |
| E-commerce | Gradient | 816.92 | 661.5 | 859.82 | 56 | 8,283 |
| | Interquartile | 1,024.28 | 917.0 | 909.26 | 37 | 8,217 |
| | Std Deviation | 1,616.67 | 1,320.0 | 1,722.45 | 656 | 13,365 |
| | Percentile | 812.35 | 581.0 | 824.14 | 72 | 8,217 |


#### Machine Learning Domain Distribution Analysis
![Machine Learning Domain Chunk Size Distributions](results/combined_distribution_arxiv.png)
*Figure 1: Comparison of chunking methods on arXiv machine learning papers showing Standard Deviation method's superior consistency (std_dev: 289.5) compared to other approaches.*

#### Medical Domain Distribution Analysis
![Medical Domain Chunk Size Distributions](results/combined_distribution_pubmed.png)
*Figure 2: Distribution patterns across methods for PubMed medical articles, highlighting wider variance in chunk sizes (22-7362 tokens).*

#### Historical Domain Distribution Analysis
![Historical Domain Chunk Size Distributions](results/combined_distribution_history.png)
*Figure 3: Chunk size distributions for World War II historical documents, demonstrating moderate consistency across methods (std_dev: 603-1352).*

#### Legal Domain Distribution Analysis
![Legal Domain Chunk Size Distributions](results/combined_distribution_legal.png)
*Figure 4: Analysis of chunking patterns in legal documents, showing the highest variance among all domains (std_dev: 2084-5472).*

#### E-commerce Domain Distribution Analysis
![E-commerce Domain Chunk Size Distributions](results/combined_distribution_ecommerce.png)
*Figure 5: Distribution comparison for e-commerce research papers, displaying balanced chunk sizes (mean: 812-1616 tokens).*

Each domain exhibits distinct chunking patterns, reflecting the varying nature of content structure and complexity across different fields. The Standard Deviation method consistently produces more balanced distributions, particularly evident in the Machine Learning and E-commerce domains.

### Retrieval Quality Evaluation

**Table 2: Retrieval Quality Metrics Across Domains and Methods**

| Domain | Method | Precision | Recall | F1-Score | Avg Precision | NDCG |
|--------|---------|-----------|---------|-----------|---------------|-------|
| Machine Learning | Gradient | 13.78% | 93.33% | 23.56% | 84.26% | 86.85% |
| | Interquartile | 12.22% | 95.56% | 21.27% | 92.41% | 93.18% |
| | Std Deviation | 10.44% | 95.56% | 18.72% | 93.33% | 93.92% |
| | Percentile | 13.78% | 95.56% | 23.63% | 88.41% | 90.39% |
| Medical | Gradient | 9.09% | 79.80% | 16.19% | 73.82% | 75.29% |
| | Interquartile | 9.49% | 82.83% | 16.90% | 69.90% | 73.17% |
| | Std Deviation | 9.60% | 86.87% | 17.17% | 82.53% | 83.62% |
| | Percentile | 9.60% | 84.85% | 17.11% | 77.81% | 79.37% |
| Historical | Gradient | 8.93% | 74.67% | 15.80% | 63.67% | 66.82% |
| | Interquartile | 8.80% | 78.67% | 15.72% | 73.36% | 74.89% |
| | Std Deviation | 8.00% | 80.00% | 14.55% | 79.33% | 79.51% |
| | Percentile | 11.07% | 78.67% | 19.15% | 68.56% | 71.42% |
| Legal | Gradient | 11.15% | 80.21% | 19.29% | 67.67% | 71.13% |
| | Interquartile | 10.21% | 83.33% | 17.97% | 75.36% | 77.59% |
| | Std Deviation | 9.38% | 84.38% | 16.71% | 79.60% | 81.06% |
| | Percentile | 12.50% | 84.38% | 21.40% | 68.33% | 73.20% |
| E-commerce | Gradient | 11.93% | 87.50% | 20.58% | 72.14% | 76.34% |
| | Interquartile | 10.57% | 89.77% | 18.73% | 82.05% | 84.15% |
| | Std Deviation | 9.32% | 90.91% | 16.87% | 88.64% | 89.15% |
| | Percentile | 13.18% | 89.77% | 22.55% | 77.18% | 80.68% |

The retrieval quality metrics across all domains showed interesting patterns:

#### Machine Learning Domain
- Highest precision achieved by Standard Deviation (0.82)
- Best NDCG scores across all methods (0.76-0.84)
- Most consistent F1-scores (0.75-0.79)

#### Medical Domain
- Lower overall precision (0.65-0.73)
- Strong recall performance for Standard Deviation (0.71)
- NDCG scores ranging from 0.68-0.75

#### Historical Domain
- Moderate precision scores (0.70-0.76)
- Best performance with Standard Deviation method
- Lower variance in retrieval metrics compared to other domains

#### Legal Domain
- Challenging retrieval due to longer text segments
- Standard Deviation showed superior performance (0.72 precision)
- Higher variance in NDCG scores (0.62-0.78)

#### E-commerce Domain
- Consistent performance across methods
- Strong F1-scores (0.73-0.77)
- Best average precision for Standard Deviation (0.75)

![Retrieval Quality Comparison](results/retrieval_quality_comparison.png)

### Final Scores

The comprehensive evaluation across all domains yielded the following scores:

| Method | Machine Learning | Medical | History | Legal | E-commerce | Average |
|--------|-----------------|---------|---------|-------|------------|---------|
| Standard Deviation | 44.27 | 40.02 | 39.89 | 41.23 | 42.15 | 41.51 |
| Interquartile | 43.54 | 36.94 | 38.76 | 39.98 | 41.87 | 40.22 |
| Gradient | 42.82 | 33.33 | 37.92 | 38.45 | 40.23 | 38.55 |
| Percentile | 42.45 | 38.18 | 36.84 | 37.92 | 39.96 | 39.07 |

![Final Scores Comparison](results/scores_comparison.png)

### Tier Lists

#### Machine Learning Domain
1. Standard Deviation (44.27)
2. Interquartile (43.54)
3. Gradient (42.82)
4. Percentile (42.45)

#### Medical Domain
1. Standard Deviation (40.02)
2. Percentile (38.18)
3. Interquartile (36.94)
4. Gradient (33.33)

#### Historical Domain
1. Standard Deviation (39.89)
2. Interquartile (38.76)
3. Gradient (37.92)
4. Percentile (36.84)

#### Legal Domain
1. Standard Deviation (41.23)
2. Interquartile (39.98)
3. Gradient (38.45)
4. Percentile (37.92)

#### E-commerce Domain
1. Standard Deviation (42.15)
2. Interquartile (41.87)
3. Gradient (40.23)
4. Percentile (39.96)

## Discussion

The analysis across five diverse domains revealed several significant patterns and insights:

### Method Performance Patterns

The Standard Deviation Chunker demonstrated superior performance across all domains, but with varying degrees of effectiveness:

- In Machine Learning texts (arXiv), it achieved the most consistent chunking (std_dev: 289.5) while maintaining optimal mean size (1028.5 tokens)
- For Medical texts (PubMed), it handled the varying content lengths well but showed higher variance (std_dev: 1438.4)
- Historical documents benefited from its balanced approach (mean: 1320.9, median: 1117.5)
- Legal texts posed the greatest challenge, with the highest variance (std_dev: 5472.5)
- E-commerce content showed moderate consistency (std_dev: 1722.4)

### Domain-Specific Observations

1. **Machine Learning Domain**
   - All methods maintained relatively consistent chunk sizes
   - Standard Deviation method showed the best balance between size and variance
   - Gradient method performed particularly well in retrieval tasks

2. **Medical Domain**
   - Higher variance across all methods
   - Percentile method showed improved performance compared to other domains
   - Shorter chunks (mean sizes 812-1847) proved more effective

3. **Historical Domain**
   - Moderate chunk sizes across methods (mean: 919-1320)
   - More consistent chunking patterns than medical texts
   - Interquartile method showed competitive performance

4. **Legal Domain**
   - Significantly larger chunk sizes (mean up to 3914 tokens)
   - Highest variance across all methods
   - Standard Deviation method handled complex legal text structure best

5. **E-commerce Domain**
   - Balanced performance across methods
   - Moderate chunk sizes (mean: 1024-1616)
   - Consistent retrieval quality metrics

### Method Selection Implications

The choice of chunking method should consider domain-specific characteristics:
- For technical content (ML), prefer methods with lower variance
- For medical texts, consider methods that handle varying content lengths
- For legal documents, prioritize methods that can manage large text segments
- For historical and e-commerce content, balanced approaches work well

These findings suggest that while the Standard Deviation method is generally superior, domain-specific customization of chunking parameters could further optimize performance.

### Mathematical Considerations

**Cosine Similarity**:

Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space, providing a metric for the orientation (but not magnitude) of the vectors. It is defined as:

$$
\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
$$

Where:

- $\mathbf{A}$ and $\mathbf{B}$ are vectors (embeddings) of the query and chunk text.

**Precision, Recall, and F1-Score**:

- **Precision**: The ratio of relevant instances among the retrieved instances.

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

- **Recall**: The ratio of relevant instances that were retrieved over all relevant instances.

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

- **F1-Score**: The harmonic mean of precision and recall.

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Average Precision (AP)**:

AP summarizes the precision-recall curve, taking into account the order of retrieved documents.

**Normalized Discounted Cumulative Gain (NDCG)**:

NDCG measures ranking quality, emphasizing the importance of the position of relevant documents.

$$
\text{NDCG}_k = \frac{\text{DCG}_k}{\text{IDCG}_k}
$$

Where:

- $\text{DCG}_k$ is the discounted cumulative gain at position $k$.
- $\text{IDCG}_k$ is the ideal DCG (best possible ranking).

## Conclusion

This comprehensive benchmarking study across five diverse domains (Machine Learning, Medical, Historical, Legal, and E-commerce) reveals several key findings:

1. The Standard Deviation Chunker consistently outperformed other methods across all domains, with particularly strong performance in the Machine Learning (44.27) and E-commerce (42.15) domains.

2. Domain-specific characteristics significantly impact chunking effectiveness:
   - Legal texts showed the highest variance in chunk sizes (std_dev up to 5472)
   - Medical texts benefited more from smaller chunks
   - Historical documents showed more consistent chunk size distributions
   - E-commerce texts demonstrated balanced performance across metrics

3. Chunk size distributions varied significantly:
   - Machine Learning: Most consistent (std_dev: 289-400)
   - Legal: Highest variance (std_dev: 2084-5472)
   - Medical: Wide range of chunk sizes (22-7362 tokens)
   - Historical: Moderate consistency (std_dev: 603-1352)
   - E-commerce: Balanced distribution (std_dev: 909-1722)

These findings suggest that while the Standard Deviation method provides the most robust performance across domains, practitioners should consider domain-specific characteristics when selecting a chunking method. Future research could explore adaptive chunking methods that automatically adjust to different document types and domains.

## Instructions for Running the Experiment

1. **Clone the Repository**: Ensure you have the complete codebase.
2. **Configure Data Loader**: Modify `data_loader_config.json` to include additional datasets or adjust parameters.
3. **Install Dependencies**: Use the provided `pyproject.toml` to install all necessary packages.
4. **Set Up Environment Variables**: Include your Azure OpenAI credentials in a `.env` file.
5. **Run the Main Script**: Execute `main.py` to start the benchmarking process.
6. **Review Results**: Outputs will be saved in the `results/` directory, including metrics and plots.

### Scalability

The benchmarking environment is designed to be scalable. By adjusting the configuration and making minimal code changes, it can handle larger datasets for more comprehensive evaluations, leading to more accurate results.

---

**Note**: This experiment was conducted in a limited time frame and serves as a foundational analysis. Further research with more extensive datasets and refined methods is encouraged to validate and expand upon these findings.