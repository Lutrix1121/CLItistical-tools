#!/usr/bin/env python3
"""
CSV Histogram Generator using Seaborn
Creates histograms from specified columns in CSV files
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

def create_histogram(csv_file, column, separator= ',', output_file=None, bins=30, figsize=(10, 6), 
                    title=None, kde=True, color='skyblue'):
    """
    Create a histogram from a specified column in a CSV file.
    
    Parameters:
    csv_file (str): Path to the CSV file
    column (str): Name of the column to create histogram for
    output_file (str): Path to save the plot (optional)
    bins (int): Number of bins for the histogram
    figsize (tuple): Figure size (width, height)
    title (str): Custom title for the plot
    kde (bool): Whether to show kernel density estimate
    color (str): Color of the histogram
    """
    
    try:
        df = pd.read_csv(csv_file, sep = separator)
        print(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns.")
        
        if column not in df.columns:
            print(f"Error: Column '{column}' not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Warning: Column '{column}' is not numeric. Attempting to convert...")
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                # Remove NaN values created during conversion
                original_count = len(df)
                df = df.dropna(subset=[column])
                if len(df) < original_count:
                    print(f"Removed {original_count - len(df)} non-numeric values.")
            except:
                print(f"Error: Could not convert column '{column}' to numeric values.")
                return False
        
        # Remove any remaining NaN values
        df_clean = df.dropna(subset=[column])
        if len(df_clean) == 0:
            print(f"Error: No valid numeric data found in column '{column}'.")
            return False
        
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        
        sns.histplot(data=df_clean, x=column, bins=bins, kde=kde, color=color, alpha=0.7)
        
        if title is None:
            title = f"Distribution of {column}"
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        mean_val = df_clean[column].mean()
        median_val = df_clean[column].median()
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
        plt.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Histogram saved as '{output_file}'")
        else:
            plt.show()
        
        print(f"\nStatistics for '{column}':")
        print(f"Count: {len(df_clean)}")
        print(f"Mean: {mean_val:.2f}")
        print(f"Median: {median_val:.2f}")
        print(f"Std Dev: {df_clean[column].std():.2f}")
        print(f"Min: {df_clean[column].min():.2f}")
        print(f"Max: {df_clean[column].max():.2f}")
        
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
    
    parser = argparse.ArgumentParser(description='Create histograms from CSV columns using seaborn')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('column', help='Name of the column to create histogram for')
    parser.add_argument('-o', '--output', help='Output file path (e.g., histogram.png)')
    parser.add_argument('-b', '--bins', type=int, default=30, help='Number of bins (default: 30)')
    parser.add_argument('--width', type=float, default=10, help='Figure width (default: 10)')
    parser.add_argument('--height', type=float, default=6, help='Figure height (default: 6)')
    parser.add_argument('-t', '--title', help='Custom title for the plot')
    parser.add_argument('--no-kde', action='store_true', help='Disable kernel density estimate')
    parser.add_argument('-c', '--color', default='skyblue', help='Histogram color (default: skyblue)')
    parser.add_argument('--separator', type=str, default=',',
                        help='Separator used in CSV file (default: ",")')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' does not exist.")
        sys.exit(1)
    
    # Create the histogram
    success = create_histogram(
        csv_file=args.csv_file,
        column=args.column,
        output_file=args.output,
        bins=args.bins,
        figsize=(args.width, args.height),
        title=args.title,
        kde=not args.no_kde,
        color=args.color, 
        separator=args.separator
    )
    
    if not success:
        sys.exit(1)

def create_multiple_histograms(csv_file, columns, output_dir=None):
    """
    Create histograms for multiple columns at once.
    
    Parameters:
    csv_file (str): Path to the CSV file
    columns (list): List of column names
    output_dir (str): Directory to save plots (optional)
    """
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for column in columns:
        output_file = None
        if output_dir:
            output_file = Path(output_dir) / f"{column}_histogram.png"
        
        print(f"\nCreating histogram for column: {column}")
        create_histogram(csv_file, column, str(output_file) if output_file else None)

if __name__ == "__main__":
    main()

