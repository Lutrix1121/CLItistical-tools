import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import sys
import os
from scipy.stats.contingency import expected_freq

def perform_chi_square_test(csv_file, column1, column2=None, alpha=0.05, test_type='independence', separator = ","):
    """
    Perform a chi-square test on categorical data from a CSV file.
    
    Parameters:
    - csv_file: Path to CSV file
    - column1: Name of the first categorical column
    - column2: Name of the second categorical column (for independence test)
    - alpha: Significance level (default: 0.05)
    - test_type: Type of test ('independence' or 'goodness_of_fit')
    
    Returns:
    - Dictionary containing test results and statistics
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file, sep = separator)
        
        # Verify that the columns exist
        if column1 not in df.columns:
            raise ValueError(f"Column '{column1}' not found in {csv_file}")
        
        if test_type == 'independence':
            if column2 is None:
                raise ValueError("Column2 is required for independence test")
            if column2 not in df.columns:
                raise ValueError(f"Column '{column2}' not found in {csv_file}")
            
            # Remove rows with missing values in either column
            clean_df = df[[column1, column2]].dropna()
            
            if len(clean_df) == 0:
                raise ValueError("No valid data remaining after removing missing values")
            
            # Create contingency table
            contingency_table = pd.crosstab(clean_df[column1], clean_df[column2])
            
            # Perform chi-square test of independence
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate effect size (Cramér's V)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            # Check assumptions
            min_expected = expected.min()
            cells_below_5 = (expected < 5).sum()
            total_cells = expected.size
            percent_below_5 = (cells_below_5 / total_cells) * 100
            
            assumption_met = min_expected >= 1 and percent_below_5 <= 20
            
            results = {
                'test_type': 'Chi-square test of independence',
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'contingency_table': contingency_table,
                'expected_frequencies': pd.DataFrame(expected, 
                                                   index=contingency_table.index,
                                                   columns=contingency_table.columns),
                'cramers_v': cramers_v,
                'sample_size': n,
                'min_expected_freq': min_expected,
                'cells_below_5': cells_below_5,
                'percent_cells_below_5': percent_below_5,
                'assumptions_met': assumption_met,
                'alpha': alpha,
                'csv_file': csv_file,
                'column1': column1,
                'column2': column2
            }
            
        else:  # goodness_of_fit test
            # Remove rows with missing values
            clean_data = df[column1].dropna()
            
            if len(clean_data) == 0:
                raise ValueError("No valid data remaining after removing missing values")
            
            # Get observed frequencies
            observed_counts = clean_data.value_counts().sort_index()
            
            # For goodness of fit, assume equal expected frequencies by default
            n_categories = len(observed_counts)
            total_n = observed_counts.sum()
            expected_counts = [total_n / n_categories] * n_categories
            
            # Perform chi-square goodness of fit test
            chi2_stat, p_value = stats.chisquare(observed_counts.values, expected_counts)
            dof = n_categories - 1
            
            # Check assumptions
            min_expected = min(expected_counts)
            assumption_met = min_expected >= 5
            
            results = {
                'test_type': 'Chi-square goodness of fit test',
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'observed_frequencies': observed_counts,
                'expected_frequencies': pd.Series(expected_counts, index=observed_counts.index),
                'sample_size': total_n,
                'n_categories': n_categories,
                'min_expected_freq': min_expected,
                'assumptions_met': assumption_met,
                'alpha': alpha,
                'csv_file': csv_file,
                'column1': column1,
                'column2': None
            }
        
        # Add conclusion
        if p_value <= alpha:
            if test_type == 'independence':
                conclusion = f"There is a statistically significant association between {column1} and {column2} at alpha = {alpha}."
            else:
                conclusion = f"The observed distribution of {column1} significantly differs from the expected uniform distribution at alpha = {alpha}."
        else:
            if test_type == 'independence':
                conclusion = f"There is no statistically significant association between {column1} and {column2} at alpha = {alpha}."
            else:
                conclusion = f"The observed distribution of {column1} does not significantly differ from the expected uniform distribution at alpha = {alpha}."
        
        results['conclusion'] = conclusion
        
        return results
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def print_results(results, output_file=None):
    """Print formatted results of chi-square test analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')

    print("\nChi-Square Test Analysis Results:")
    print("=" * 80)
    print(f"Test type: {results['test_type']}")
    print(f"Data file: {results['csv_file']}")
    print(f"Column 1: {results['column1']}")
    if results['column2']:
        print(f"Column 2: {results['column2']}")
    print(f"Significance level (alpha): {results['alpha']}")
    print("=" * 80)

        # Test Details
    print("\nTEST DETAILS:")
    print("-" * 40)
    if results['test_type'] == 'Chi-square test of independence':
        print("Test: Pearson's chi-square test of independence")
        print("Null hypothesis: The two variables are independent")
        print("Alternative hypothesis: The two variables are associated")
    else:
        print("Test: Pearson's chi-square goodness of fit test")
        print("Null hypothesis: The observed distribution follows the expected distribution")
        print("Alternative hypothesis: The observed distribution differs from expected")
    
    print("Missing values: Automatically excluded from analysis")
    
    # Test Statistics
    print("\nTEST STATISTICS:")
    print("-" * 40)
    print(f"Chi-square statistic: {results['chi2_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Degrees of freedom: {results['degrees_of_freedom']}")
    print(f"Sample size: {results['sample_size']}")
    
    if 'cramers_v' in results:
        print(f"Cramér's V (effect size): {results['cramers_v']:.4f}")
    
    # Frequency Tables
    print("\nFREQUENCY TABLES:")
    print("-" * 40)
    
    if results['test_type'] == 'Chi-square test of independence':
        print("Observed Frequencies (Contingency Table):")
        print(results['contingency_table'])
        print("\nExpected Frequencies:")
        print(results['expected_frequencies'].round(2))
        
        # Residuals
        observed = results['contingency_table'].values
        expected = results['expected_frequencies'].values
        residuals = observed - expected
        standardized_residuals = residuals / np.sqrt(expected)
        
        print("\nStandardized Residuals:")
        print(pd.DataFrame(standardized_residuals.round(2), 
                          index=results['contingency_table'].index,
                          columns=results['contingency_table'].columns))
    
    else:
        print("Observed vs Expected Frequencies:")
        freq_comparison = pd.DataFrame({
            'Observed': results['observed_frequencies'],
            'Expected': results['expected_frequencies']
        })
        freq_comparison['Residual'] = freq_comparison['Observed'] - freq_comparison['Expected']
        print(freq_comparison)
    
    # Assumptions Check
    print("\nASSUMPTION CHECKS:")
    print("-" * 40)
    print(f"Minimum expected frequency: {results['min_expected_freq']:.2f}")
    
    if results['test_type'] == 'Chi-square test of independence':
        print(f"Cells with expected frequency < 5: {results['cells_below_5']} ({results['percent_cells_below_5']:.1f}%)")
        print(f"Assumptions met: {'Yes' if results['assumptions_met'] else 'No'}")
        
        if not results['assumptions_met']:
            print("\nWARNING: Chi-square test assumptions may be violated!")
            print("Consider:")
            print("- Combining categories with low frequencies")
            print("- Using Fisher's exact test for 2x2 tables")
            print("- Collecting more data")
    else:
        print(f"All expected frequencies ≥ 5: {'Yes' if results['assumptions_met'] else 'No'}")
        if not results['assumptions_met']:
            print("\nWARNING: Chi-square test assumptions may be violated!")
            print("Consider collecting more data or using exact tests.")
    
    # Statistical Conclusion
    print("\nSTATISTICAL CONCLUSION:")
    print("-" * 40)
    print(f"Conclusion: {results['conclusion']}")
    
    # Effect Size Interpretation
    print("\nINTERPRETATION:")
    print("-" * 40)
    
    if results['p_value'] <= results['alpha']:
        if results['p_value'] <= 0.001:
            print("The association/difference is highly significant (p ≤ 0.001).")
        elif results['p_value'] <= 0.01:
            print("The association/difference is very significant (p ≤ 0.01).")
        else:
            print("The association/difference is significant (p ≤ 0.05).")
        
        if 'cramers_v' in results:
            if results['cramers_v'] < 0.1:
                effect_size = "negligible"
            elif results['cramers_v'] < 0.3:
                effect_size = "small"
            elif results['cramers_v'] < 0.5:
                effect_size = "medium"
            else:
                effect_size = "large"
            print(f"Effect size (Cramér's V = {results['cramers_v']:.3f}) is {effect_size}.")
    else:
        print("No significant association or difference was found.")
        print("This could mean:")
        print("- The variables are truly independent")
        print("- The sample size is too small to detect an association")
        print("- The effect size is very small")
    
    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")
        os.startfile(output_file)

def main():
    parser = argparse.ArgumentParser(description='Perform chi-square tests on categorical data from CSV files.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('column1', help='Name of the first categorical column')
    parser.add_argument('--column2', help='Name of the second categorical column (required for independence test)')
    parser.add_argument('--test_type', choices=['independence', 'goodness_of_fit'], 
                        default='independence', help='Type of chi-square test (default: independence)')
    parser.add_argument('--alpha', type=float, default=0.05, 
                        help='Significance level (default: 0.05)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    parser.add_argument('--separator', type=str, default=',', 
                        help='Separator used in CSV file (default: ",")')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_type == 'independence' and args.column2 is None:
        parser.error("--column2 is required for independence test")
    
    # Perform chi-square test
    results = perform_chi_square_test(
        args.csv_file, args.column1, args.column2, 
        args.alpha, args.test_type, args.separator
    )
    
    # Display results
    print_results(results, args.output)

if __name__ == "__main__":
    main()