import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    def __init__(self, style='seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_class_distribution(self, mass_df, calc_df):
        """Plot class distribution of pathology"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mass distribution
        mass_counts = mass_df['pathology'].value_counts()
        axes[0].pie(mass_counts.values, labels=mass_counts.index, autopct='%1.1f%%')
        axes[0].set_title('Mass Pathology Distribution')
        
        # Calcification distribution
        calc_counts = calc_df['pathology'].value_counts()
        axes[1].pie(calc_counts.values, labels=calc_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Calcification Pathology Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_dataset_stats(self, datasets_dict):
        """Display basic statistics for all datasets"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (name, df) in enumerate(datasets_dict.items()):
            if idx < len(axes):
                axes[idx].text(0.1, 0.5, 
                              f"Dataset: {name}\n"
                              f"Shape: {df.shape}\n"
                              f"Columns: {len(df.columns)}\n"
                              f"Samples: {len(df)}",
                              fontsize=10, 
                              verticalalignment='center')
                axes[idx].axis('off')
                axes[idx].set_title(name)
        
        plt.tight_layout()
        plt.show()