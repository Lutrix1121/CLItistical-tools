#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

def create_comparison_graph(csv_file, x_column, y_column, separator=',', output_file=None, 
                           graph_type='scatter', figsize=(10, 6), title=None, 
                           hue=None, palette='deep', style=None):
    """
    Create a comparison graph from specified columns in a CSV file.
    
    Parameters:
    csv_file (str): Path to the CSV file
    x_column (str): Name of the column for x-axis
    y_column (str): Name of the column for y-axis
    separator (str): CSV separator character
    output_file (str): Path to save the plot (optional)
    graph_type (str): Type of graph ('scatter', 'line', 'bar', 'box', 'violin', 'strip')
    figsize (tuple): Figure size (width, height)
    title (str): Custom title for the plot
    hue (str): Column name for color grouping
    palette (str): Color palette to use
    style (str): Column name for style grouping (for line/scatter plots)
    """
    
    try:
        df = pd.read_csv(csv_file, sep=separator)
        print(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns.")
        
        # Check if columns exist
        required_columns = [x_column, y_column]
        if hue:
            required_columns.append(hue)
        if style:
            required_columns.append(style)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Columns {missing_columns} not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Remove rows with NaN values in required columns
        df_clean = df.dropna(subset=required_columns)
        if len(df_clean) == 0:
            print(f"Error: No valid data found after removing NaN values.")
            return False
        
        if len(df_clean) < len(df):
            print(f"Removed {len(df) - len(df_clean)} rows with missing values.")
        
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        
        # Create the appropriate graph type
        if graph_type == 'scatter':
            sns.scatterplot(data=df_clean, x=x_column, y=y_column, hue=hue, 
                          style=style, palette=palette, s=100, alpha=0.7)
        
        elif graph_type == 'line':
            sns.lineplot(data=df_clean, x=x_column, y=y_column, hue=hue, 
                        style=style, palette=palette, markers=True, linewidth=2)
        
        elif graph_type == 'bar':
            sns.barplot(data=df_clean, x=x_column, y=y_column, hue=hue, 
                       palette=palette, errorbar='sd')
        
        elif graph_type == 'box':
            sns.boxplot(data=df_clean, x=x_column, y=y_column, hue=hue, 
                       palette=palette)
        
        elif graph_type == 'violin':
            sns.violinplot(data=df_clean, x=x_column, y=y_column, hue=hue, 
                          palette=palette, inner='box')
        
        elif graph_type == 'strip':
            sns.stripplot(data=df_clean, x=x_column, y=y_column, hue=hue, 
                         palette=palette, alpha=0.7, size=6)
        
        else:
            print(f"Error: Unknown graph type '{graph_type}'.")
            print("Available types: scatter, line, bar, box, violin, strip")
            return False
        
        if title is None:
            title = f"{y_column} vs {x_column}"
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        
        # Rotate x-axis labels if they're too long or there are many categories
        if df_clean[x_column].dtype == 'object':
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Comparison graph saved as '{output_file}'")
        else:
            plt.show()
        
        print(f"\nStatistics:")
        print(f"Total data points: {len(df_clean)}")
        
        if pd.api.types.is_numeric_dtype(df_clean[y_column]):
            print(f"\n{y_column} statistics:")
            print(f"Mean: {df_clean[y_column].mean():.2f}")
            print(f"Median: {df_clean[y_column].median():.2f}")
            print(f"Std Dev: {df_clean[y_column].std():.2f}")
            print(f"Min: {df_clean[y_column].min():.2f}")
            print(f"Max: {df_clean[y_column].max():.2f}")
        
        if hue:
            print(f"\nGroups by {hue}:")
            print(df_clean[hue].value_counts())
        
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
    
    parser = argparse.ArgumentParser(
        description='Create comparison graphs from CSV columns using seaborn',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Graph types:
  scatter - Scatter plot showing relationship between two variables
  line    - Line plot for trends over time or continuous data
  bar     - Bar plot for comparing categories
  box     - Box plot showing distribution across categories
  violin  - Violin plot showing distribution shape across categories
  strip   - Strip plot showing individual data points across categories

Examples:
  %(prog)s data.csv age salary --type scatter
  %(prog)s data.csv month revenue --type line --hue region
  %(prog)s data.csv category price --type bar --output comparison.png
        """
    )
    
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('x_column', help='Name of the column for x-axis')
    parser.add_argument('y_column', help='Name of the column for y-axis')
    parser.add_argument('-o', '--output', help='Output file path (e.g., comparison.png)')
    parser.add_argument('-t', '--type', default='scatter', 
                       choices=['scatter', 'line', 'bar', 'box', 'violin', 'strip'],
                       help='Type of comparison graph (default: scatter)')
    parser.add_argument('--width', type=float, default=10, help='Figure width (default: 10)')
    parser.add_argument('--height', type=float, default=6, help='Figure height (default: 6)')
    parser.add_argument('--title', help='Custom title for the plot')
    parser.add_argument('--hue', help='Column name for color grouping')
    parser.add_argument('--style', help='Column name for style grouping (scatter/line only)')
    parser.add_argument('--palette', default='deep', 
                       help='Color palette (default: deep). Options: deep, muted, bright, pastel, dark, colorblind')
    parser.add_argument('--separator', type=str, default=',',
                       help='Separator used in CSV file (default: ",")')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' does not exist.")
        sys.exit(1)
    
    # Create the comparison graph
    success = create_comparison_graph(
        csv_file=args.csv_file,
        x_column=args.x_column,
        y_column=args.y_column,
        separator=args.separator,
        output_file=args.output,
        graph_type=args.type,
        figsize=(args.width, args.height),
        title=args.title,
        hue=args.hue,
        palette=args.palette,
        style=args.style
    )
    
    if not success:
        sys.exit(1)

def create_multiple_comparisons(csv_file, comparisons, output_dir=None, separator=','):
    """
    Create multiple comparison graphs at once.
    
    Parameters:
    csv_file (str): Path to the CSV file
    comparisons (list): List of tuples (x_column, y_column, graph_type)
    output_dir (str): Directory to save plots (optional)
    separator (str): CSV separator character
    """
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for x_col, y_col, graph_type in comparisons:
        output_file = None
        if output_dir:
            output_file = Path(output_dir) / f"{y_col}_vs_{x_col}_{graph_type}.png"
        
        print(f"\nCreating {graph_type} plot for {y_col} vs {x_col}")
        create_comparison_graph(
            csv_file=csv_file,
            x_column=x_col,
            y_column=y_col,
            separator=separator,
            output_file=str(output_file) if output_file else None,
            graph_type=graph_type
        )

if __name__ == "__main__":
    main()
