"""
Visualization module for creating charts and plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class Visualizer:
    """Creates visualizations for framing bias analysis."""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_source_distribution(
        self,
        source_data: Dict[str, Dict[str, int]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create bar chart of source domains by framing.
        
        Args:
            source_data: Dict mapping framing to domain counts
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        # Prepare data for plotting
        domains = ['wikipedia', 'mainstream_news', 'gov_ngo', 'blog', 'forum', 'other']
        framings = list(source_data.keys())
        
        # Create DataFrame
        data = []
        for framing in framings:
            counts = source_data[framing]
            for domain in domains:
                data.append({
                    'Framing': framing,
                    'Domain': domain,
                    'Count': counts.get(domain, 0)
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars
        x = np.arange(len(domains))
        width = 0.8 / len(framings)
        
        for i, framing in enumerate(framings):
            framing_data = df[df['Framing'] == framing]
            counts = [framing_data[framing_data['Domain'] == d]['Count'].values[0] 
                     if len(framing_data[framing_data['Domain'] == d]) > 0 else 0
                     for d in domains]
            
            ax.bar(x + i * width, counts, width, label=framing)
        
        # Customize plot
        ax.set_xlabel('Source Domain')
        ax.set_ylabel('Number of Snippets')
        ax.set_title('Source Domain Distribution by Framing')
        ax.set_xticks(x + width * (len(framings) - 1) / 2)
        ax.set_xticklabels(domains, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'source_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_sentiment_toxicity(
        self,
        metrics_data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create box plots for sentiment and toxicity by framing.
        
        Args:
            metrics_data: DataFrame with columns: frame_label, sentiment, toxicity
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sentiment box plot
        if 'sentiment' in metrics_data.columns:
            sns.boxplot(data=metrics_data, x='frame_label', y='sentiment', ax=ax1)
            ax1.set_title('Sentiment Distribution by Framing')
            ax1.set_xlabel('Framing Type')
            ax1.set_ylabel('Sentiment Score')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Add horizontal line at 0
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Toxicity box plot
        if 'toxicity' in metrics_data.columns:
            sns.boxplot(data=metrics_data, x='frame_label', y='toxicity', ax=ax2)
            ax2.set_title('Toxicity Distribution by Framing')
            ax2.set_xlabel('Framing Type')
            ax2.set_ylabel('Toxicity Score')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Add horizontal line at threshold
            ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Toxicity Threshold')
            ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'sentiment_toxicity.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_jaccard_heatmap(
        self,
        jaccard_matrix: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create heatmap of Jaccard similarity between framings.
        
        Args:
            jaccard_matrix: DataFrame with Jaccard similarities
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            jaccard_matrix,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            cbar_kws={'label': 'Jaccard Similarity'},
            square=True
        )
        
        plt.title('Document Overlap (Jaccard Similarity) Between Framings')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'jaccard_heatmap.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_cooccurrence(
        self,
        cooccurrence_data: Dict[str, Dict[str, int]],
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create bar chart of top co-occurring terms.
        
        Args:
            cooccurrence_data: Dict mapping framing to co-occurrence counts
            top_n: Number of top terms to show
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        # Aggregate co-occurrence counts
        all_terms = set()
        for counts in cooccurrence_data.values():
            all_terms.update(counts.keys())
        
        # Get top terms by total count
        term_totals = {}
        for term in all_terms:
            total = sum(counts.get(term, 0) for counts in cooccurrence_data.values())
            term_totals[term] = total
        
        top_terms = sorted(term_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_term_names = [t[0] for t in top_terms]
        
        # Prepare data for plotting
        framings = list(cooccurrence_data.keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar chart
        x = np.arange(len(top_term_names))
        width = 0.8 / len(framings)
        
        for i, framing in enumerate(framings):
            counts = [cooccurrence_data[framing].get(term, 0) for term in top_term_names]
            ax.bar(x + i * width, counts, width, label=framing)
        
        # Customize plot
        ax.set_xlabel('Co-occurring Terms')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Top {top_n} Co-occurring Terms by Framing')
        ax.set_xticks(x + width * (len(framings) - 1) / 2)
        ax.set_xticklabels(top_term_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'cooccurrence.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_summary_stats_table(
        self,
        metrics_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create summary statistics table.
        
        Args:
            metrics_df: DataFrame with metrics
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        # Group by frame_label and compute statistics
        summary_stats = metrics_df.groupby('frame_label').agg({
            'sentiment': ['mean', 'std', 'min', 'max'],
            'toxicity': ['mean', 'std', 'min', 'max'],
            'subjectivity': ['mean', 'std'],
            'length': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=summary_stats.values,
            colLabels=summary_stats.columns,
            rowLabels=summary_stats.index,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.title('Summary Statistics by Framing Type', fontsize=14, pad=20)
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'summary_stats.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_all_visualizations(
        self,
        source_data: Dict[str, Dict[str, int]],
        metrics_df: pd.DataFrame,
        jaccard_matrix: pd.DataFrame,
        cooccurrence_data: Dict[str, Dict[str, int]]
    ) -> Dict[str, str]:
        """
        Create all visualizations.
        
        Args:
            source_data: Source domain distribution data
            metrics_df: DataFrame with output metrics
            jaccard_matrix: Jaccard similarity matrix
            cooccurrence_data: Co-occurrence data
            
        Returns:
            Dict mapping visualization names to file paths
        """
        viz_paths = {}
        
        try:
            viz_paths['source_distribution'] = self.plot_source_distribution(source_data)
        except Exception as e:
            logger.error(f"Failed to create source distribution plot: {e}")
            
        try:
            viz_paths['sentiment_toxicity'] = self.plot_sentiment_toxicity(metrics_df)
        except Exception as e:
            logger.error(f"Failed to create sentiment/toxicity plot: {e}")
            
        try:
            viz_paths['jaccard_heatmap'] = self.plot_jaccard_heatmap(jaccard_matrix)
        except Exception as e:
            logger.error(f"Failed to create Jaccard heatmap: {e}")
            
        try:
            viz_paths['cooccurrence'] = self.plot_cooccurrence(cooccurrence_data)
        except Exception as e:
            logger.error(f"Failed to create co-occurrence plot: {e}")
            
        try:
            viz_paths['summary_stats'] = self.create_summary_stats_table(metrics_df)
        except Exception as e:
            logger.error(f"Failed to create summary stats table: {e}")
            
        return viz_paths