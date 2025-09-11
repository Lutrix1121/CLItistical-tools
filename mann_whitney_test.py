import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import argparse
from datetime import datetime
import os

def perform_mann_whitney_test(csv_file1, csv_file2=None, column=None, column1=None, column2=None,
                             grouping_variable=None, group1=None, group2=None, 
                             separator=',', alternative='two-sided'):
    """
    Performs Mann-Whitney U test for variables from one or two CSV files.
    
    Parameters:
    csv_file1 (str): Path to the first CSV file.
    csv_file2 (str, optional): Path to the second CSV file. If provided, data from two files are compared.
    column (str, optional): Name of the column to compare between two files.
    column1 (str, optional): Name of the column containing the first variable (for single file).
    column2 (str, optional): Name of the column containing the second variable (for single file).
    grouping_variable (str, optional): Name of the column for grouping data (for single file).
    group1 (str/int/float, optional): Value of the first group for the grouping variable.
    group2 (str/int/float, optional): Value of the second group for the grouping variable.
    separator (str, optional): Separator used in CSV file (default: ',').
    alternative (str, optional): Alternative hypothesis ('two-sided', 'less', 'greater').
    
    Returns:
    dict: Test results containing U statistic, p-value, and sample sizes.
    """
    try:
        # Determine working mode based on provided parameters
        if csv_file2 is not None:
            # Two-file comparison mode
            if column is None:
                raise ValueError("When comparing two files, you must specify the column name for comparison ('column' parameter).")
            
            # Load data from both CSV files
            data1 = pd.read_csv(csv_file1, sep=separator)
            data2 = pd.read_csv(csv_file2, sep=separator)
            
            # Check if the specified column exists in both datasets
            if column not in data1.columns:
                raise ValueError(f"Column '{column}' does not exist in the first dataset.")
            if column not in data2.columns:
                raise ValueError(f"Column '{column}' does not exist in the second dataset.")
            
            values_group1 = data1[column].dropna().values
            values_group2 = data2[column].dropna().values
            
            description_group1 = f"File: {csv_file1}, column: {column}"
            description_group2 = f"File: {csv_file2}, column: {column}"
            
        else:
            # Single file analysis mode
            data = pd.read_csv(csv_file1, sep=separator)
            
            if grouping_variable is not None:
                # Grouping by variable mode
                if grouping_variable not in data.columns:
                    raise ValueError(f"Grouping variable '{grouping_variable}' does not exist in the dataset.")
                if group1 is None or group2 is None:
                    raise ValueError("When grouping variable is provided, you must specify both groups.")
                
                if column is None and (column1 is None or column2 is None):
                    raise ValueError("You must specify either a column for analysis ('column' parameter) or two columns ('column1' and 'column2' parameters).")
                
                # Filter data by grouping variable
                data_group1 = data[data[grouping_variable] == group1]
                data_group2 = data[data[grouping_variable] == group2]
                
                if data_group1.empty:
                    raise ValueError(f"No data for group '{group1}'.")
                if data_group2.empty:
                    raise ValueError(f"No data for group '{group2}'.")
                
                if column is not None:
                    if column not in data.columns:
                        raise ValueError(f"Column '{column}' does not exist in the dataset.")
                    values_group1 = data_group1[column].dropna().values
                    values_group2 = data_group2[column].dropna().values
                    column_description = f"column: {column}"
                else:
                    if column1 not in data.columns:
                        raise ValueError(f"Column '{column1}' does not exist in the dataset.")
                    if column2 not in data.columns:
                        raise ValueError(f"Column '{column2}' does not exist in the dataset.")
                    values_group1 = data_group1[column1].dropna().values
                    values_group2 = data_group2[column2].dropna().values
                    column_description = f"columns: {column1} and {column2}"
                
                description_group1 = f"Group: {group1}, {column_description}"
                description_group2 = f"Group: {group2}, {column_description}"
            
            else:
                if column1 is None or column2 is None:
                    raise ValueError("When comparing two columns from one file, you must specify both column names ('column1' and 'column2' parameters).")
                
                if column1 not in data.columns:
                    raise ValueError(f"Column '{column1}' does not exist in the dataset.")
                if column2 not in data.columns:
                    raise ValueError(f"Column '{column2}' does not exist in the dataset.")
                
                # Get column values
                values_group1 = data[column1].dropna().values
                values_group2 = data[column2].dropna().values
                
                description_group1 = f"Column: {column1}"
                description_group2 = f"Column: {column2}"
        
        if len(values_group1) < 2 or len(values_group2) < 2:
            raise ValueError("Each group must contain at least 2 observations.")
        
        u_statistic, p_value = mannwhitneyu(values_group1, values_group2, alternative=alternative)
        
        results = {
            'u_statistic': u_statistic,
            'p_value': p_value,
            'sample_size_1': len(values_group1),
            'sample_size_2': len(values_group2),
            'mean_group1': np.mean(values_group1),
            'mean_group2': np.mean(values_group2),
            'median_group1': np.median(values_group1),
            'median_group2': np.median(values_group2),
            'alternative_hypothesis': alternative,
            'description_group1': description_group1,
            'description_group2': description_group2
        }
        
        return results
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def print_results(results, output_file=None, alpha=0.05):
    """
    Print Mann-Whitney U test results to console and optionally to a file.
    
    Parameters:
    results (dict): Test results from perform_mann_whitney_test function.
    output_file (str, optional): Path to output file for saving results.
    alpha (float, optional): Significance level for interpretation (default: 0.05).
    """
    if results is None:
        print("No results to display.")
        return
    
    output_lines = []
    output_lines.append("Mann-Whitney U Test Results:")
    output_lines.append("=" * 60)
    output_lines.append(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("-" * 60)
    output_lines.append(f"Group 1: {results['description_group1']}")
    output_lines.append(f"Group 2: {results['description_group2']}")
    output_lines.append("-" * 60)
    output_lines.append(f"U Statistic: {results['u_statistic']:.4f}")
    output_lines.append(f"P-value: {results['p_value']:.4f}")
    output_lines.append(f"Alternative hypothesis: {results['alternative_hypothesis']}")
    output_lines.append("-" * 60)
    output_lines.append(f"Sample size 1: {results['sample_size_1']}")
    output_lines.append(f"Sample size 2: {results['sample_size_2']}")
    output_lines.append("-" * 60)
    output_lines.append(f"Mean group 1: {results['mean_group1']:.4f}")
    output_lines.append(f"Mean group 2: {results['mean_group2']:.4f}")
    output_lines.append(f"Median group 1: {results['median_group1']:.4f}")
    output_lines.append(f"Median group 2: {results['median_group2']:.4f}")
    output_lines.append("-" * 60)
    
    if results['p_value'] < alpha:
        output_lines.append(f"At significance level α = {alpha}, we reject the null hypothesis.")
        output_lines.append("There is a statistically significant difference between groups.")
    else:
        output_lines.append(f"At significance level α = {alpha}, there is no basis to reject the null hypothesis.")
        output_lines.append("There is no statistically significant difference between groups.")
    
    output_lines.append("=" * 60)
    
    for line in output_lines:
        print(line)
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\nResults saved to: {output_file}")
	    os.startfile(output_file)
        except Exception as e:
            print(f"\nError saving results to file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Mann-Whitney U test for data from CSV files')
    parser.add_argument('csv_file1', type=str, help='Path to the first CSV file')
    parser.add_argument('--csv_file2', type=str, default=None, 
                        help='Path to the second CSV file (optional)')
    parser.add_argument('--column', type=str, default=None, 
                        help='Name of the column to compare between two files')
    parser.add_argument('--column1', type=str, default=None, 
                        help='Name of the first column to compare in one file')
    parser.add_argument('--column2', type=str, default=None, 
                        help='Name of the second column to compare in one file')
    parser.add_argument('--grouping_variable', type=str, default=None, 
                        help='Name of the grouping column (optional)')
    parser.add_argument('--group1', type=str, default=None, 
                        help='Value of the first group for the grouping variable')
    parser.add_argument('--group2', type=str, default=None, 
                        help='Value of the second group for the grouping variable')
    parser.add_argument('--separator', type=str, default=',', 
                        help='Separator used in CSV file (default: ",")')
    parser.add_argument('--alternative', type=str, default='two-sided', 
                        choices=['two-sided', 'less', 'greater'],
                        help='Alternative hypothesis (default: "two-sided")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for interpretation (default: 0.05)')
    
    args = parser.parse_args()
    
    # Convert group arguments to appropriate data types (if possible)
    group1 = args.group1
    group2 = args.group2
    
    if group1 is not None:
        try:
            group1 = int(group1)
        except ValueError:
            try:
                group1 = float(group1)
            except ValueError:
                pass  # Keep as string
    
    if group2 is not None:
        try:
            group2 = int(group2)
        except ValueError:
            try:
                group2 = float(group2)
            except ValueError:
                pass  # Keep as string
    
    # Perform the test
    results = perform_mann_whitney_test(
        args.csv_file1, args.csv_file2, args.column, args.column1, args.column2, 
        args.grouping_variable, group1, group2, args.separator, args.alternative
    )
    
    print_results(results, args.output, args.alpha)

if __name__ == "__main__":
    main()