import pandas as pd
import scipy.stats as stats
import argparse
import sys
import os

def perform_t_test(file1, file2, column_name, alpha=0.05, separator = ","):
    """
    Perform a t-test on a column from two different CSV files.
    
    Parameters:
    - file1: Path to first CSV file
    - file2: Path to second CSV file
    - column_name: Name of the column to compare
    - alpha: Significance level (default: 0.05)
    
    Returns:
    - t_statistic: The t-statistic value
    - p_value: The p-value of the test
    - conclusion: Text interpretation of the result
    """
    try:
        df1 = pd.read_csv(file1, sep = separator)
        df2 = pd.read_csv(file2, sep = separator)
        
        if column_name not in df1.columns:
            raise ValueError(f"Column '{column_name}' not found in {file1}")
        if column_name not in df2.columns:
            raise ValueError(f"Column '{column_name}' not found in {file2}")
        
        data1 = df1[column_name].dropna()
        data2 = df2[column_name].dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            raise ValueError("Each group must have at least 2 observations")
        
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)  # Using Welch's t-test (unequal variance)
        
        conclusion = "There is " + ("" if p_val <= alpha else "not ") + \
                    f"a statistically significant difference between the two datasets at alpha = {alpha}."
        
        results = {
            't_statistic': t_stat,
            'p_value': p_val,
            'df1_mean': data1.mean(),
            'df2_mean': data2.mean(),
            'df1_std': data1.std(),
            'df2_std': data2.std(),
            'df1_count': len(data1),
            'df2_count': len(data2),
            'conclusion': conclusion,
            'alpha': alpha,
            'file1': file1,
            'file2': file2,
            'column': column_name
        }
        
        return results
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def print_results(results, output_file=None):
    """Print formatted results of t-test analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')

    print("\nT-Test Analysis Results:")
    print("=" * 80)
    print(f"Column analyzed: {results['column']}")
    print(f"File 1: {results['file1']}")
    print(f"File 2: {results['file2']}")
    print(f"Significance level (alpha): {results['alpha']}")
    print("=" * 80)
    
    
    print("\nTEST DETAILS:")
    print("-" * 40)
    print("Test type: Independent samples t-test (Welch's t-test)")
    print("Assumptions: Assumes unequal variances")
    print("Missing values: Automatically excluded from analysis")

    print("\nTEST STATISTICS:")
    print("-" * 40)
    print(f"t-statistic: {results['t_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    
    print("\nDESCRIPTIVE STATISTICS:")
    print("-" * 40)
    print(f"File 1 ({results['file1']}):")
    print(f"  Sample size: {results['df1_count']}")
    print(f"  Mean: {results['df1_mean']:.4f}")
    print(f"  Standard deviation: {results['df1_std']:.4f}")
    
    print(f"\nFile 2 ({results['file2']}):")
    print(f"  Sample size: {results['df2_count']}")
    print(f"  Mean: {results['df2_mean']:.4f}")
    print(f"  Standard deviation: {results['df2_std']:.4f}")
    
    print(f"\nMean difference: {abs(results['df1_mean'] - results['df2_mean']):.4f}")
    
    print("\nSTATISTICAL CONCLUSION:")
    print("-" * 40)
    print(f"Conclusion: {results['conclusion']}")
    
    print("\nINTERPRETATION:")
    print("-" * 40)
    if results['p_value'] <= results['alpha']:
        if results['df1_mean'] > results['df2_mean']:
            print(f"File 1 has a significantly higher mean than File 2.")
        else:
            print(f"File 2 has a significantly higher mean than File 1.")
        
        if results['p_value'] <= 0.001:
            print("The difference is highly significant (p ≤ 0.001).")
        elif results['p_value'] <= 0.01:
            print("The difference is very significant (p ≤ 0.01).")
        else:
            print("The difference is significant (p ≤ 0.05).")
    else:
        print("No significant difference was found between the two groups.")
        print("This could mean:")
        print("- The groups are truly similar")
        print("- The sample size is too small to detect a difference")
        print("- The effect size is very small")
    
    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")
        os.startfile(output_file)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Perform a t-test on columns from two CSV files.')
    parser.add_argument('file1', help='Path to the first CSV file')
    parser.add_argument('file2', help='Path to the second CSV file')
    parser.add_argument('column', help='Name of the column to compare')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default: 0.05)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    parser.add_argument('--separator', type=str, default=',', 
                        help='Separator used in CSV file (default: ",")')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Perform t-test
    results = perform_t_test(args.file1, args.file2, args.column, args.alpha, args.separator)
    
    # Display results
    print_results(results, args.output)

if __name__ == "__main__":
    main()