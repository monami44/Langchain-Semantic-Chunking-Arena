import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def combine_distribution_plots(domains=["arxiv", "pubmed", "history", "legal", "ecommerce"], 
                             base_output_path="results/combined_distribution"):
    # Process each domain
    for domain in domains:
        # Create a figure with 1x4 subplots
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        
        # Load and display images
        methods = [
            ("Percentile", "percentile"),
            ("Standard Deviation", "std_deviation"),
            ("Interquartile", "interquartile"),
            ("Gradient", "gradient")
        ]
        
        for idx, (title, method) in enumerate(methods):
            img_path = f'results/{method}_{domain}_distribution.png'
            try:
                img = mpimg.imread(img_path)
                axs[idx].imshow(img)
                axs[idx].axis('off')
                axs[idx].set_title(f'{title} Method', pad=20, fontsize=14)
            except FileNotFoundError:
                print(f"Warning: Could not find image {img_path}")
        
        plt.tight_layout()
        output_path = f"{base_output_path}_{domain}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    combine_distribution_plots() 