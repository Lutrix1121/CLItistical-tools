import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import os
        
def analyze_csv_statistics(file_path, separator = ";", categorical_columns=None, output_file=None):
    """
    Analyze a CSV file and provide comprehensive statistics for all columns.
    
    Args:
        file_path (str): Path to the CSV file
        categorical_columns (list): List of column names to treat as categorical (non-numeric)
        output_file (str): Path to save results. If None, auto-generates filename.
    """
    try:
        # Generate output filename if not provided
        if output_file is None:
            base_name = Path(file_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_statistics_{timestamp}.txt"
        
        df = pd.read_csv(file_path, sep=separator)
        
        output_lines = []
        output_lines.append(f"CSV STATISTICS ANALYSIS REPORT")
        output_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append(f"Source file: {file_path}")
        output_lines.append("=" * 80)
        output_lines.append(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        output_lines.append("=" * 80)
        
        # Get initially numeric columns
        initially_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        initially_non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Handle categorical columns override
        if categorical_columns is None:
            categorical_columns = []
        
        # Validate that specified categorical columns exist
        invalid_cols = [col for col in categorical_columns if col not in df.columns]
        if invalid_cols:
            warning_msg = f"Warning: These columns don't exist in the dataset: {invalid_cols}"
            output_lines.append(f"\n{warning_msg}")
            print(warning_msg)
            categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        # Final separation: move specified categorical columns to non-numeric
        numeric_columns = [col for col in initially_numeric if col not in categorical_columns]
        non_numeric_columns = initially_non_numeric + [col for col in categorical_columns if col in initially_numeric]
        
        # Show column classification
        if categorical_columns:
            classification_msg = f"Columns explicitly treated as categorical: {categorical_columns}"
            output_lines.append(f"\n{classification_msg}")
            output_lines.append("=" * 80)
            print(classification_msg)
            print("=" * 80)
        
        if numeric_columns:
            section_header = "\nNUMERIC COLUMNS STATISTICS"
            output_lines.append(section_header)
            output_lines.append("=" * 50)
            print(section_header)
            print("=" * 50)
            
            for col in numeric_columns:
                col_header = f"\nColumn: {col}"
                col_separator = "-" * 30
                output_lines.append(col_header)
                output_lines.append(col_separator)
                print(col_header)
                print(col_separator)
                
                stats = [
                    f"Count (non-null):     {df[col].count():,}",
                    f"Missing values:       {df[col].isnull().sum():,}",
                    f"Mean:                 {df[col].mean():.4f}",
                    f"Median:               {df[col].median():.4f}",
                    f"Standard Deviation:   {df[col].std():.4f}",
                    f"Variance:             {df[col].var():.4f}",
                    f"Minimum:              {df[col].min():.4f}",
                    f"Maximum:              {df[col].max():.4f}",
                    f"Range:                {df[col].max() - df[col].min():.4f}",
                    f"25th Percentile (Q1): {df[col].quantile(0.25):.4f}",
                    f"75th Percentile (Q3): {df[col].quantile(0.75):.4f}",
                    f"IQR:                  {df[col].quantile(0.75) - df[col].quantile(0.25):.4f}",
                    f"Skewness:             {df[col].skew():.4f}",
                    f"Kurtosis:             {df[col].kurtosis():.4f}"
                ]
                
                for stat in stats:
                    output_lines.append(stat)
                    print(stat)
        
        if non_numeric_columns:
            section_header = "\n\nNON-NUMERIC COLUMNS STATISTICS"
            output_lines.append(section_header)
            output_lines.append("=" * 50)
            print(section_header)
            print("=" * 50)
            
            for col in non_numeric_columns:
                col_header = f"\nColumn: {col}"
                col_separator = "-" * 30
                output_lines.append(col_header)
                output_lines.append(col_separator)
                print(col_header)
                print(col_separator)
                
                basic_info = [
                    f"Count (non-null):     {df[col].count():,}",
                    f"Missing values:       {df[col].isnull().sum():,}",
                    f"Unique values:        {df[col].nunique():,}",
                    f"Most frequent value:  {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}"
                ]
                
                for info in basic_info:
                    output_lines.append(info)
                    print(info)
                
                value_counts = df[col].value_counts()
                freq_header = f"\nTop {min(10, len(value_counts))} most frequent values:"
                output_lines.append(freq_header)
                print(freq_header)
                
                for idx, (value, count) in enumerate(value_counts.head(10).items(), 1):
                    percentage = (count / df[col].count()) * 100
                    freq_line = f"  {idx:2d}. '{value}': {count:,} ({percentage:.2f}%)"
                    output_lines.append(freq_line)
                    print(freq_line)
                
                if len(value_counts) > 10:
                    more_values = f"  ... and {len(value_counts) - 10} more unique values"
                    output_lines.append(more_values)
                    print(more_values)
        
        summary_header = "\n\nDATASET SUMMARY"
        output_lines.append(summary_header)
        output_lines.append("=" * 50)
        print(summary_header)
        print("=" * 50)
        
        summary_stats = [
            f"Total rows:           {df.shape[0]:,}",
            f"Total columns:        {df.shape[1]:,}",
            f"Numeric columns:      {len(numeric_columns)}",
            f"Non-numeric columns:  {len(non_numeric_columns)}",
            f"Total missing values: {df.isnull().sum().sum():,}",
            f"Memory usage:         {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        ]
        
        for stat in summary_stats:
            output_lines.append(stat)
            print(stat)
        
        missing_summary = df.isnull().sum()
        columns_with_missing = missing_summary[missing_summary > 0]
        if not columns_with_missing.empty:
            missing_header = f"\nColumns with missing values:"
            output_lines.append(missing_header)
            print(missing_header)
            for col, missing_count in columns_with_missing.items():
                percentage = (missing_count / len(df)) * 100
                missing_line = f"  {col}: {missing_count:,} ({percentage:.2f}%)"
                output_lines.append(missing_line)
                print(missing_line)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {os.path.abspath(output_file)}")
        print(f"{'='*80}")
        
    except FileNotFoundError:
        error_msg = f"Error: File '{file_path}' not found."
        print(error_msg)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        error_msg = "Error: The CSV file is empty."
        print(error_msg)
        sys.exit(1)
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing CSV file: {e}"
        print(error_msg)
        sys.exit(1)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        sys.exit(1)

def main():
    """
    Main function to handle command line arguments and run the analysis.
    """
    if len(sys.argv) < 3:
        print("Usage: python csv_analyzer.py <path_to_csv_file> <separator> [categorical_columns]")
        print("\nExamples:")
        print("  python csv_analyzer.py data.csv")
        print("  python csv_analyzer.py data.csv zip_code,product_id,phone")
        print("  python csv_analyzer.py data.csv \"ZIP Code,Product ID,Phone Number\"")
        print("\nNote: Categorical columns should be comma-separated with no spaces around commas")
        print("      Use quotes if column names contain spaces")
        sys.exit(1)
    
    file_path = sys.argv[1]
    separator = sys.argv[2]
    
    # Parse categorical columns if provided
    categorical_columns = []
    if len(sys.argv) > 2:
        categorical_str = sys.argv[3]
        categorical_columns = [col.strip() for col in categorical_str.split(',')]
        categorical_columns = [col for col in categorical_columns if col]  # Remove empty strings
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    
    # Run the analysis
    analyze_csv_statistics(file_path, separator, categorical_columns)

if __name__ == "__main__":
    main()
