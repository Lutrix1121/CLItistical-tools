#!/usr/bin/env python3
"""
CSV Dependency Graph Generator using Seaborn
Creates scatter plots and regression plots to visualize dependencies between two variables in CSV files
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

def create_dependency_graph(csv_file, x_column, y_column, separator=',', output_file=None, 
                           figsize=(10, 8), title=None, plot_type='scatter', color='steelblue',
                           show_regression=False, show_correlation=False, alpha=0.7):
    """
    Create a dependency graph between two variables from a CSV file.
    
    Parameters:
    csv_file (str): Path to the CSV file
    x_column (str): Name of the x-axis column
    y_column (str): Name of the y-axis column
    separator (str): CSV separator character
    output_file (str): Path to save the plot (optional)
    figsize (tuple): Figure size (width, height)
    title (str): Custom title for the plot
    plot_type (str): Type of plot ('scatter', 'reg', 'hex', 'kde')
    color (str): Color of the plot elements
    show_regression (bool): Whether to show regression line
    show_correlation (bool): Whether to display correlation statistics
    alpha (float): Transparency level for points
    """
    
    try:
        df = pd.read_csv(csv_file, sep=separator)
        print(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns.")
        
        missing_columns = []
        if x_column not in df.columns:
            missing_columns.append(x_column)
        if y_column not in df.columns:
            missing_columns.append(y_column)
            
        if missing_columns:
            print(f"Error: Column(s) {missing_columns} not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Check and convert columns to numeric if needed
        for col in [x_column, y_column]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Warning: Column '{col}' is not numeric. Attempting to convert...")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    print(f"Error: Could not convert column '{col}' to numeric values.")
                    return False
        
        # Remove rows with NaN values in either column
        original_count = len(df)
        df_clean = df.dropna(subset=[x_column, y_column])
        
        if len(df_clean) < original_count:
            print(f"Removed {original_count - len(df_clean)} rows with missing values.")
        
        if len(df_clean) == 0:
            print(f"Error: No valid numeric data found for both columns.")
            return False
        
        if len(df_clean) < 2:
            print(f"Error: Need at least 2 data points for dependency analysis.")
            return False
        
        # Set up the plot style
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        
        if plot_type == 'scatter':
            if show_regression:
                sns.regplot(data=df_clean, x=x_column, y=y_column, 
                           scatter_kws={'alpha': alpha, 'color': color},
                           line_kws={'color': 'red', 'alpha': 0.8})
            else:
                sns.scatterplot(data=df_clean, x=x_column, y=y_column, 
                               color=color, alpha=alpha)
        
        elif plot_type == 'reg':
            sns.regplot(data=df_clean, x=x_column, y=y_column, color=color)
        
        elif plot_type == 'hex':
            plt.hexbin(df_clean[x_column], df_clean[y_column], 
                      gridsize=30, cmap='Blues', alpha=0.8)
            plt.colorbar(label='Count')
            if show_regression:
                # Add regression line for hexbin
                z = np.polyfit(df_clean[x_column], df_clean[y_column], 1)
                p = np.poly1d(z)
                plt.plot(df_clean[x_column], p(df_clean[x_column]), 
                        "r--", alpha=0.8, linewidth=2)
        
        elif plot_type == 'kde':
            sns.kdeplot(data=df_clean, x=x_column, y=y_column, 
                       fill=True, alpha=0.6, color=color)
            if show_regression:
                sns.regplot(data=df_clean, x=x_column, y=y_column, 
                           scatter=False, line_kws={'color': 'red', 'alpha': 0.8})
        
        else:
            print(f"Warning: Unknown plot type '{plot_type}'. Using scatter plot.")
            sns.scatterplot(data=df_clean, x=x_column, y=y_column, 
                           color=color, alpha=alpha)
        
        # Customize the plot
        if title is None:
            title = f"Dependency Graph: {y_column} vs {x_column}"
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        
        if show_correlation and len(df_clean) >= 3:
            try:
                pearson_corr, pearson_p = pearsonr(df_clean[x_column], df_clean[y_column])
                spearman_corr, spearman_p = spearmanr(df_clean[x_column], df_clean[y_column])
                
                # Add correlation info to the plot
                corr_text = f'Pearson r = {pearson_corr:.3f} (p = {pearson_p:.3f})\n'
                corr_text += f'Spearman ρ = {spearman_corr:.3f} (p = {spearman_p:.3f})'
                
                plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            except Exception as e:
                print(f"Warning: Could not calculate correlation statistics: {e}")
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Dependency graph saved as '{output_file}'")
        else:
            plt.show()
        
        print(f"\nDependency Analysis for '{x_column}' vs '{y_column}':")
        print(f"Data points: {len(df_clean)}")
        
        if show_correlation and len(df_clean) >= 3:
            try:
                pearson_corr, pearson_p = pearsonr(df_clean[x_column], df_clean[y_column])
                spearman_corr, spearman_p = spearmanr(df_clean[x_column], df_clean[y_column])
                
                print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
                print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
                
                # Interpret correlation strength
                abs_pearson = abs(pearson_corr)
                if abs_pearson >= 0.7:
                    strength = "strong"
                elif abs_pearson >= 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                direction = "positive" if pearson_corr > 0 else "negative"
                print(f"Relationship: {strength} {direction} correlation")
                
            except Exception as e:
                print(f"Could not calculate correlation statistics: {e}")
        
        for col in [x_column, y_column]:
            print(f"\nStatistics for '{col}':")
            print(f"  Mean: {df_clean[col].mean():.2f}")
            print(f"  Std Dev: {df_clean[col].std():.2f}")
            print(f"  Min: {df_clean[col].min():.2f}")
            print(f"  Max: {df_clean[col].max():.2f}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_file}' is empty.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments and run the script."""
    
    parser = argparse.ArgumentParser(description='Create dependency graphs from CSV columns using seaborn')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('x_column', help='Name of the x-axis column')
    parser.add_argument('y_column', help='Name of the y-axis column')
    parser.add_argument('-o', '--output', help='Output file path (e.g., dependency_graph.png)')
    parser.add_argument('-p', '--plot-type', choices=['scatter', 'reg', 'hex', 'kde'], 
                       default='scatter', help='Type of plot (default: scatter)')
    parser.add_argument('--width', type=float, default=10, help='Figure width (default: 10)')
    parser.add_argument('--height', type=float, default=8, help='Figure height (default: 8)')
    parser.add_argument('-t', '--title', help='Custom title for the plot')
    parser.add_argument('--regression', action='store_true', help='Enable regression line')
    parser.add_argument('--correlation', action='store_true', help='Enable correlation statistics')
    parser.add_argument('-c', '--color', default='steelblue', help='Plot color (default: steelblue)')
    parser.add_argument('--alpha', type=float, default=0.7, help='Point transparency (default: 0.7)')
    parser.add_argument('--separator', type=str, default=',',
                        help='Separator used in CSV file (default: ",")')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' does not exist.")
        sys.exit(1)
    
    # Create the dependency graph
    success = create_dependency_graph(
        csv_file=args.csv_file,
        x_column=args.x_column,
        y_column=args.y_column,
        separator=args.separator,
        output_file=args.output,
        figsize=(args.width, args.height),
        title=args.title,
        plot_type=args.plot_type,
        color=args.color,
        show_regression=args.regression,
        show_correlation=args.correlation,
        alpha=args.alpha
    )
    
    if not success:
        sys.exit(1)

# Example usage as a module
def create_multiple_dependency_graphs(csv_file, variable_pairs, output_dir=None, plot_type='scatter'):
    """
    Create dependency graphs for multiple variable pairs at once.
    
    Parameters:
    csv_file (str): Path to the CSV file
    variable_pairs (list): List of tuples containing (x_column, y_column) pairs
    output_dir (str): Directory to save plots (optional)
    plot_type (str): Type of plot to create
    """
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for x_col, y_col in variable_pairs:
        output_file = None
        if output_dir:
            output_file = Path(output_dir) / f"{x_col}_vs_{y_col}_dependency.png"
        
        print(f"\nCreating dependency graph for: {y_col} vs {x_col}")
        create_dependency_graph(
            csv_file, x_col, y_col, 
            output_file=str(output_file) if output_file else None,
            plot_type=plot_type
        )

if __name__ == "__main__":
    main()