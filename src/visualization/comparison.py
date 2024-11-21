import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Gradient', 'Interquartile', 'Standard Deviation', 'Percentile']
domains = ['Machine Learning', 'Medical', 'History', 'Legal', 'E-commerce']

# Scores data
scores = {
    'Standard Deviation': [44.27, 40.02, 39.89, 41.23, 42.15],
    'Interquartile': [43.54, 36.94, 38.76, 39.98, 41.87],
    'Gradient': [42.82, 33.33, 37.92, 38.45, 40.23],
    'Percentile': [42.45, 38.18, 36.84, 37.92, 39.96]
}

# Define colors for each domain
domain_colors = {
    'Machine Learning': '#2ecc71',  # green
    'Medical': '#3498db',          # blue
    'History': '#e74c3c',          # red
    'Legal': '#f1c40f',           # yellow
    'E-commerce': '#9b59b6'       # purple
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 8))
bar_width = 0.15

# Plot bars for each domain
index = np.arange(len(methods))
for i, domain in enumerate(domains):
    domain_scores = [scores[method][i] for method in methods]
    bars = ax.bar([x + i*bar_width for x in index], 
                 domain_scores, 
                 bar_width, 
                 label=domain,
                 color=domain_colors[domain],
                 alpha=0.8)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=8)

# Customize the plot
ax.set_xlabel('Methods', fontsize=12, labelpad=10)
ax.set_ylabel('Scores', fontsize=12, labelpad=10)
ax.set_title('Comparison of Scores Across Methods and Domains', fontsize=14, pad=20)

# Set y-axis limits to emphasize differences
ax.set_ylim(32, 45)

# Center x-tick labels
ax.set_xticks([x + (bar_width * (len(domains)-1))/2 for x in index])
ax.set_xticklabels(methods, rotation=45)

# Add legend with better positioning
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add more frequent y-axis ticks
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

# Add grid
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout()
plt.savefig('scores_comparison_new.png', bbox_inches='tight', dpi=300)
plt.close()