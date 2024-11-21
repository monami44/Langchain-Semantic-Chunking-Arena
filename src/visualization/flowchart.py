from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Benchmarking Flowchart', format='png')
dot.attr(
    rankdir='LR',  # Left to right orientation
    size='24,16',  # Doubled the size
    dpi='300'      # Added high DPI setting
)

# Define global node attributes for styling
dot.attr('node', 
    shape='rectangle', 
    style='rounded, filled', 
    fillcolor='lightgray', 
    fontname='Helvetica', 
    fontsize='14'  # Increased font size
)

# Start Node
dot.node('Start', 'Start', fillcolor='lightgreen')

# Data Collection
dot.node('DataCollection', 'Data Collection')
dot.node('MLDataset', 'Machine Learning Dataset')
dot.node('MedicalDataset', 'Medical Dataset')
dot.node('HistoricalDataset', 'Historical Dataset')
dot.node('LegalDataset', 'Legal Dataset')
dot.node('EcommerceDataset', 'E-commerce Dataset')

# Ground Truth Generation
dot.node('GroundTruth', 'Ground Truth Generation')
dot.node('QuestionGen', 'Question Generation\n(using GPT-4)')
dot.node('ChunkExtraction', 'Relevant Chunk Extraction\n(using GPT-4)')
dot.node('MetadataAssoc', 'Metadata Association')

# Apply Chunking Methods
dot.node('ChunkingMethods', 'Apply Chunking Methods')
dot.node('PercentileChunker', 'Percentile Chunker')
dot.node('InterquartileChunker', 'Interquartile Chunker')
dot.node('GradientChunker', 'Gradient Chunker')
dot.node('StdDevChunker', 'Standard Deviation Chunker')

# Embeddings and Similarity Calculation
dot.node('Embeddings', 'Embeddings and Similarity Calculation')
dot.node('GenerateEmbeddings', 'Generate Embeddings\n(using all-MiniLM-L6-v2)')
dot.node('ComputeSimilarity', 'Compute Cosine Similarity')

# Evaluation Metrics Computation
dot.node('EvaluationMetrics', 'Evaluation Metrics Computation')

# Chunk Size Metrics
dot.node('ChunkSizeMetrics', 'Chunk Size Metrics')
dot.node('MeanSize', 'Mean Size')
dot.node('StdDevSize', 'Standard Deviation')
dot.node('MinSize', 'Minimum Size')
dot.node('MaxSize', 'Maximum Size')

# Retrieval Quality Metrics
dot.node('RetrievalQualityMetrics', 'Retrieval Quality Metrics')
dot.node('Precision', 'Precision')
dot.node('Recall', 'Recall')
dot.node('F1Score', 'F1-Score')
dot.node('AveragePrecision', 'Average Precision (AP)')
dot.node('NDCG', 'Normalized Discounted Cumulative Gain (NDCG)')



# Size Score Components
dot.node('SizeScore', 'Size Score (40%)')
dot.node('MeanScore', 'Mean Score (35%)')
dot.node('StdDevScore', 'Std Dev Score (35%)')
dot.node('MinSizeScore', 'Min Size Score (15%)')
dot.node('MaxSizeScore', 'Max Size Score (15%)')

# Retrieval Score Components
dot.node('RetrievalScore', 'Retrieval Score (60%)')
dot.node('PrecisionScore', 'Precision (20%)')
dot.node('RecallScore', 'Recall (20%)')
dot.node('F1ScoreScore', 'F1-Score (20%)')
dot.node('APScore', 'Average Precision (20%)')
dot.node('NDCGScore', 'NDCG (20%)')

# Results and Analysis
dot.node('Results', 'Results and Analysis', fillcolor='lightblue')
dot.node('End', 'End', fillcolor='lightgreen')

# Edges
dot.edge('Start', 'DataCollection')

# Data Collection to Datasets
dot.edges([
    ('DataCollection', 'MLDataset'),
    ('DataCollection', 'MedicalDataset'),
    ('DataCollection', 'HistoricalDataset'),
    ('DataCollection', 'LegalDataset'),
    ('DataCollection', 'EcommerceDataset')
])

# Datasets to Ground Truth Generation
dot.edges([
    ('MLDataset', 'GroundTruth'),
    ('MedicalDataset', 'GroundTruth'),
    ('HistoricalDataset', 'GroundTruth'),
    ('LegalDataset', 'GroundTruth'),
    ('EcommerceDataset', 'GroundTruth')
])

# Ground Truth Generation Steps
dot.edge('GroundTruth', 'QuestionGen')
dot.edge('QuestionGen', 'ChunkExtraction')
dot.edge('ChunkExtraction', 'MetadataAssoc')

# Metadata Association to Apply Chunking Methods
dot.edge('MetadataAssoc', 'ChunkingMethods')

# Apply Chunking Methods to Individual Methods
dot.edges([
    ('ChunkingMethods', 'PercentileChunker'),
    ('ChunkingMethods', 'InterquartileChunker'),
    ('ChunkingMethods', 'GradientChunker'),
    ('ChunkingMethods', 'StdDevChunker')
])

# Chunking Methods to Embeddings and Similarity Calculation
dot.edges([
    ('PercentileChunker', 'Embeddings'),
    ('InterquartileChunker', 'Embeddings'),
    ('GradientChunker', 'Embeddings'),
    ('StdDevChunker', 'Embeddings')
])

# Embeddings and Similarity Calculation Steps
dot.edge('Embeddings', 'GenerateEmbeddings')
dot.edge('GenerateEmbeddings', 'ComputeSimilarity')

# Compute Similarity to Evaluation Metrics Computation
dot.edge('ComputeSimilarity', 'EvaluationMetrics')

# Evaluation Metrics Computation Steps
dot.edge('EvaluationMetrics', 'ChunkSizeMetrics')
dot.edge('EvaluationMetrics', 'RetrievalQualityMetrics')

# Chunk Size Metrics Details
dot.edges([
    ('ChunkSizeMetrics', 'MeanSize'),
    ('ChunkSizeMetrics', 'StdDevSize'),
    ('ChunkSizeMetrics', 'MinSize'),
    ('ChunkSizeMetrics', 'MaxSize')
])

# Retrieval Quality Metrics Details
dot.edges([
    ('RetrievalQualityMetrics', 'Precision'),
    ('RetrievalQualityMetrics', 'Recall'),
    ('RetrievalQualityMetrics', 'F1Score'),
    ('RetrievalQualityMetrics', 'AveragePrecision'),
    ('RetrievalQualityMetrics', 'NDCG')
])

# Metrics to Scoring System Application
dot.edges([
    ('MeanSize', 'SizeScore'),
    ('StdDevSize', 'SizeScore'),
    ('MinSize', 'SizeScore'),
    ('MaxSize', 'SizeScore'),
    ('Precision', 'RetrievalScore'),
    ('Recall', 'RetrievalScore'),
    ('F1Score', 'RetrievalScore'),
    ('AveragePrecision', 'RetrievalScore'),
    ('NDCG', 'RetrievalScore')
])

# Scoring System Application Steps
# Size Score Components
dot.edges([
    ('SizeScore', 'MeanScore'),
    ('SizeScore', 'StdDevScore'),
    ('SizeScore', 'MinSizeScore'),
    ('SizeScore', 'MaxSizeScore')
])

# Retrieval Score Components
dot.edges([
    ('RetrievalScore', 'PrecisionScore'),
    ('RetrievalScore', 'RecallScore'),
    ('RetrievalScore', 'F1ScoreScore'),
    ('RetrievalScore', 'APScore'),
    ('RetrievalScore', 'NDCGScore')
])

# Scoring to Results
dot.edges([
    ('MeanScore', 'Results'),
    ('StdDevScore', 'Results'),
    ('MinSizeScore', 'Results'),
    ('MaxSizeScore', 'Results'),
    ('PrecisionScore', 'Results'),
    ('RecallScore', 'Results'),
    ('F1ScoreScore', 'Results'),
    ('APScore', 'Results'),
    ('NDCGScore', 'Results')
])

# Results to End
dot.edge('Results', 'End')

# Render the graph to a file and display it
dot.render('benchmarking_flowchart', view=True)
