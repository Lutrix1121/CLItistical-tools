# CLItistical Tools

A comprehensive set of command-line statistical analysis tools for CSV data. Perform various statistical tests, generate visualizations, and analyze data directly from your terminal.

## Features

### Statistical Tests
- **ANOVA** - One-way Analysis of Variance with post-hoc tests
- **Chi-Square Test** - Test for independence and goodness of fit
- **Friedman Test** - Non-parametric repeated measures test
- **Mann-Whitney U Test** - Non-parametric test for two independent samples
- **Power Analysis** - Calculate sample size, effect size, or statistical power

### Regression Analysis
- **Linear Regression** - Multiple linear regression with diagnostics
- **Logistic Regression** - Binary classification with performance metrics

### Data Visualization
- **Histograms** - Distribution visualization with KDE
- **Dependency Graphs** - Scatter plots and regression plots
- **Mean Comparison Graphs** - Compare statistics across multiple columns

### Descriptive Statistics
- **Basic Statistics** - Comprehensive statistical summaries for CSV files

## Requirements

```bash
pip install pandas numpy scipy scikit-learn seaborn matplotlib
```

## Usage

### ANOVA Test

Perform one-way ANOVA to compare means across multiple groups:

```bash
# Wide format (each column is a group)
python anova_script.py data.csv "group1,group2,group3" --alpha 0.05 --output results.txt

# Long format (groups in one column)
python anova_script.py data.csv value_column --data_format long --group_col group_column
```

**Options:**
- `--alpha`: Significance level (default: 0.05)
- `--separator`: CSV separator (default: ",")
- `--output`: Save results to file
- `--no_assumptions`: Skip assumption checking

### Chi-Square Test

Test for association between categorical variables:

```bash
# Test of independence
python chi_square_test.py data.csv column1 --column2 column2 --alpha 0.05

# Goodness of fit test
python chi_square_test.py data.csv column1 --test_type goodness_of_fit
```

**Options:**
- `--separator`: CSV separator (default: ",")
- `--alpha`: Significance level (default: 0.05)
- `--output`: Save results to file

### Friedman Test

Non-parametric repeated measures test:

```bash
# Wide format
python friedman_test.py data.csv "condition1,condition2,condition3"

# Long format
python friedman_test.py data.csv condition_col --data_format long --subject_col subject_id
```

### Mann-Whitney U Test

Compare two independent samples:

```bash
# Two columns from one file
python mann_whitney_test.py data.csv --column1 group1 --column2 group2

# Two separate files
python mann_whitney_test.py file1.csv --csv_file2 file2.csv --column value

# Using grouping variable
python mann_whitney_test.py data.csv --grouping_variable treatment --group1 control --group2 experimental --column score
```

**Options:**
- `--alternative`: 'two-sided', 'less', 'greater' (default: 'two-sided')
- `--separator`: CSV separator (default: ",")

### Linear Regression

Build predictive models:

```bash
python linear_regression_tool.py data.csv target_column --predictor_columns feature1 feature2 feature3

# Use all numeric columns as predictors
python linear_regression_tool.py data.csv target_column
```

**Options:**
- `--test_size`: Train/test split ratio (default: 0.2)
- `--no_standardize`: Don't standardize features
- `--separator`: CSV separator (default: ",")
- `--output`: Save results to file

### Logistic Regression

Binary classification:

```bash
python logistic_regression_tool.py data.csv target_column --predictor_columns feature1 feature2
```

**Options:**
- `--test_size`: Train/test split ratio (default: 0.2)
- `--solver`: Optimization algorithm (default: 'lbfgs')
- `--max_iter`: Maximum iterations (default: 1000)

### Power Analysis

Calculate sample size, effect size, or power:

```bash
# Calculate required sample size for t-test
python power_analysis.py ttest --effect_size 0.5 --power 0.8

# Calculate power for ANOVA
python power_analysis.py anova --effect_size 0.3 --sample_size 30 --groups 3

# Calculate effect size for chi-square test
python power_analysis.py chisquare --sample_size 100 --power 0.8 --df 2
```

**Test Types:**
- `ttest`: T-test (one-sample, two-sample, paired)
- `anova`: One-way ANOVA
- `chisquare`: Chi-square test
- `correlation`: Correlation test

### Histogram

Create distribution visualizations:

```bash
python histogram.py data.csv column_name --bins 30 -o histogram.png
```

**Options:**
- `--bins`: Number of bins (default: 30)
- `--no-kde`: Disable kernel density estimate
- `--color`: Histogram color (default: 'skyblue')
- `--separator`: CSV separator (default: ",")

### Dependency Graph

Visualize relationships between variables:

```bash
python dependency_graph.py data.csv x_column y_column -o graph.png

# With regression line and correlation stats
python dependency_graph.py data.csv x_column y_column --regression --correlation
```

**Plot Types:**
- `scatter`: Scatter plot (default)
- `reg`: Regression plot
- `hex`: Hexbin plot
- `kde`: Kernel density estimate plot

**Options:**
- `--plot-type`: Type of visualization
- `--regression`: Show regression line
- `--correlation`: Display correlation statistics

### Mean Comparison Graph

Compare statistics across multiple columns:

```bash
python mean_comparison_graph.py data.csv col1 col2 col3 --type mean -o comparison.png

# Show all statistics
python mean_comparison_graph.py data.csv col1 col2 col3 --type all

# Group by categorical variable
python mean_comparison_graph.py data.csv col1 col2 --type mean --group-by category_column
```

**Statistics Types:**
- `mean`: Compare means
- `std`: Compare standard deviations
- `variance`: Compare variances
- `all`: Show all three statistics

### Basic Statistics

Generate comprehensive statistical summaries:

```bash
python basic_statistics.py data.csv ";" --output statistics.txt

# Treat specific columns as categorical
python basic_statistics.py data.csv ";" "zip_code,product_id"
```

## Output

All statistical tests provide detailed output including:
- Test statistics and p-values
- Effect sizes and confidence intervals
- Assumption checks (when applicable)
- Post-hoc analyses (when significant)
- Clear interpretations and recommendations

Results can be saved to text files using the `--output` parameter.

## Data Format

### CSV Requirements
- First row should contain column headers
- Numeric columns for quantitative analyses
- Categorical columns for chi-square and grouping variables
- Missing values are automatically handled (rows removed)

### Wide vs Long Format

**Wide Format** (default for most tests):
```
Subject, Condition1, Condition2, Condition3
1,       23,         25,         28
2,       19,         22,         24
```

**Long Format** (requires `--data_format long`):
```
Subject, Condition, Value
1,       Cond1,     23
1,       Cond2,     25
1,       Cond3,     28
```

## Examples

### Complete ANOVA Analysis
```bash
python anova_script.py experiment.csv "control,treatment1,treatment2" \
  --alpha 0.05 \
  --output anova_results.txt
```

### Regression with Visualization
```bash
# Run regression
python linear_regression_tool.py sales.csv revenue \
  --predictor_columns advertising price season \
  --output regression_results.txt

# Visualize relationship
python dependency_graph.py sales.csv advertising revenue \
  --regression --correlation \
  -o advertising_revenue.png
```

### Power Analysis Workflow
```bash
# Calculate required sample size
python power_analysis.py ttest \
  --effect_size 0.5 \
  --power 0.8 \
  --alpha 0.05 \
  --output power_analysis.txt
```

## Tips

1. **Check assumptions**: Most tests include automatic assumption checking
2. **Save outputs**: Use `--output` to save detailed results
3. **Visualize first**: Create histograms and dependency graphs before running tests
4. **Choose appropriate tests**: Use non-parametric tests (Friedman, Mann-Whitney) when assumptions are violated
5. **Consider effect sizes**: Statistical significance ≠ practical significance

## License

This project is open source and available for academic and research purposes.

## Contributing

Contributions are welcome! Please ensure all statistical tests follow best practices and include comprehensive output.

## Author

Created by Lutrix1121

## Support

For issues or questions, please open an issue on the GitHub repository.