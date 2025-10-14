# CLItistical Tools

A comprehensive set of command-line statistical analysis tools for CSV data. Perform various statistical tests, generate visualizations, and analyze data directly from your terminal.

## Features

### Statistical Tests:
- **ANOVA** - One-way Analysis of Variance with post-hoc tests,
- **Chi-Square Test** - Test for independence and goodness of fit,
- **Friedman Test** - Non-parametric repeated measures test,
- **Mann-Whitney U Test** - Non-parametric test for two independent samples,
- **Power Analysis** - Calculate sample size, effect size, or statistical power;

### Machine Learning:
- **ML Classifier** - Multiple classification algorithms with comprehensive evaluation,
- **ML Clustering** - Unsupervised clustering with multiple algorithms and visualizations,
- **Linear Regression** - Multiple linear regression with diagnostics,
- **Logistic Regression** - Binary classification with performance metrics,
- **Polynomial regression** - With expanded feature space tracking and overfitting detection;

### Data Visualization:
- **Histograms** - Distribution visualization with KDE,
- **Dependency Graphs** - Scatter plots and regression plots,
- **Mean Comparison Graphs** - Compare statistics across multiple columns;

### Descriptive Statistics:
- **Basic Statistics** - Comprehensive statistical summaries for CSV files;

## Requirements

```bash
pip install pandas numpy scipy scikit-learn scikit-learn-extra seaborn matplotlib
```

## Usage

### ML Clustering Tool:

Discover natural groupings in your data using various clustering algorithms

```bash
# K-Means clustering with 3 clusters
python ml_clustering_tool.py data.csv --algorithm kmeans --n_clusters 3

# DBSCAN density-based clustering
python ml_clustering_tool.py data.csv --algorithm dbscan --eps 0.5 --min_samples 5

# Agglomerative hierarchical clustering
python ml_clustering_tool.py data.csv --algorithm agglomerative --n_clusters 4 --linkage complete

# K-Medoids clustering (more robust to outliers)
python ml_clustering_tool.py data.csv --algorithm kmedoids --n_clusters 3

# With specific feature columns
python ml_clustering_tool.py data.csv --algorithm kmeans --n_clusters 3 --feature_columns age income spending_score

# Find optimal number of clusters with elbow and silhouette analysis
python ml_clustering_tool.py data.csv --algorithm kmeans --elbow_analysis --silhouette_analysis

# Complete analysis with visualizations
python ml_clustering_tool.py data.csv --algorithm kmeans --n_clusters 3 --visualize --output results.txt --output_data clustered_data.csv
```

**Available Algorithms:**
- `kmeans`: K-Means - Fast, efficient partitioning method
- `kmedoids`: K-Medoids - Similar to K-Means but more robust to outliers
- `agglomerative`: Agglomerative Hierarchical - Bottom-up hierarchical clustering
- `dbscan`: DBSCAN - Density-based, can find arbitrary shapes and outliers

**Options:**
- `--n_clusters`: Number of clusters (default: 3, not used for DBSCAN),
- `--feature_columns`: Specify features (default: all numeric columns),
- `--no_standardize`: Skip feature standardization,
- `--separator`: CSV separator (default: ","),
- `--random_state`: Set seed for reproducibility (default: 42),
- `--eps`: DBSCAN epsilon parameter - maximum distance between points (default: 0.5),
- `--min_samples`: DBSCAN minimum samples per cluster (default: 5),
- `--linkage`: Hierarchical linkage method: ward, complete, average, single (default: ward),
- `--elbow_analysis`: Perform elbow method to find optimal K,
- `--silhouette_analysis`: Perform silhouette analysis to find optimal K,
- `--visualize`: Create PCA plots, cluster distributions, and heatmaps,
- `--output`: Save results to file,
- `--output_data`: Save original data with cluster assignments;

**Output Includes:**
- Number of clusters found
- Cluster sizes and distribution
- Silhouette Score (cluster separation quality)
- Calinski-Harabasz Score (cluster density)
- Davies-Bouldin Score (cluster similarity)
- Within-cluster sum of squares (inertia)
- Cluster statistics (means and standard deviations)
- Optimal K recommendations (with elbow/silhouette analysis)
- Quality interpretation and warnings
- PCA visualization plots
- Cluster distribution charts
- Feature importance heatmaps

### ML Classifier Tool:

Advanced machine learning classification with multiple algorithms and comprehensive evaluation

```bash
# Single algorithm classification
python ml_classifier_tool.py data.csv target_column --algorithm random_forest

# Compare all algorithms
python ml_classifier_tool.py data.csv target_column --algorithm all

# With specific predictor columns
python ml_classifier_tool.py data.csv target_column --algorithm gradient_boost --predictor_columns feature1 feature2 feature3

# With cross-validation
python ml_classifier_tool.py data.csv target_column --algorithm svm --cross_validation --cv_folds 10
```

**Available Algorithms:**
- `logistic`: Logistic Regression - Fast, interpretable baseline
- `decision_tree`: Decision Tree - Non-linear, interpretable
- `random_forest`: Random Forest - Ensemble method, handles non-linearity well
- `gradient_boost`: Gradient Boosting - Often highest accuracy
- `svm`: Support Vector Machine - Effective in high-dimensional spaces
- `knn`: K-Nearest Neighbors - Simple, non-parametric
- `naive_bayes`: Naive Bayes - Fast, works well with small datasets
- `neural_network`: Multi-Layer Perceptron - Deep learning approach
- `all`: Run all algorithms and compare performance

**Options:**
- `--predictor_columns`: Specify features (default: all numeric columns),
- `--test_size`: Train/test split ratio (default: 0.2),
- `--random_state`: Set seed for reproducibility (default: 42),
- `--no_standardize`: Skip feature standardization,
- `--cross_validation`: Perform k-fold cross-validation,
- `--cv_folds`: Number of CV folds (default: 5),
- `--separator`: CSV separator (default: ","),
- `--output`: Save results to file;

**Output Includes:**
- Accuracy, Precision, Recall, F1-Score (train and test)
- Confusion Matrix
- Classification Report per class
- Feature Importance/Coefficients
- Cross-validation scores (if enabled)
- Overfitting detection
- Performance recommendations

### ANOVA Test:

Perform one-way ANOVA to compare means across multiple groups

```bash
# Wide format (each column is a group)
python anova_script.py data.csv "group1,group2,group3" --alpha 0.05 --output results.txt

# Long format (groups in one column)
python anova_script.py data.csv value_column --data_format long --group_col group_column
```

**Options:**
- `--alpha`: Significance level (default: 0.05),
- `--separator`: CSV separator (default: ","),
- `--output`: Save results to file,
- `--no_assumptions`: Skip assumption checking;

### Chi-Square Test:

Test for association between categorical variables

```bash
# Test of independence
python chi_square_test.py data.csv column1 --column2 column2 --alpha 0.05

# Goodness of fit test
python chi_square_test.py data.csv column1 --test_type goodness_of_fit
```

**Options:**
- `--separator`: CSV separator (default: ","),
- `--alpha`: Significance level (default: 0.05),
- `--output`: Save results to file;

### Friedman Test:

Non-parametric repeated measures test

```bash
# Wide format
python friedman_test.py data.csv "condition1,condition2,condition3"

# Long format
python friedman_test.py data.csv condition_col --data_format long --subject_col subject_id
```

### Mann-Whitney U Test:

Compare two independent samples

```bash
# Two columns from one file
python mann_whitney_test.py data.csv --column1 group1 --column2 group2

# Two separate files
python mann_whitney_test.py file1.csv --csv_file2 file2.csv --column value

# Using grouping variable
python mann_whitney_test.py data.csv --grouping_variable treatment --group1 control --group2 experimental --column score
```

**Options:**
- `--alternative`: 'two-sided', 'less', 'greater' (default: 'two-sided'),
- `--separator`: CSV separator (default: ",");

### Linear Regression:

Build predictive models

```bash
python linear_regression_tool.py data.csv target_column --predictor_columns feature1 feature2 feature3

# Use all numeric columns as predictors
python linear_regression_tool.py data.csv target_column
```

**Options:**
- `--test_size`: Train/test split ratio (default: 0.2),
- `--no_standardize`: Don't standardize features,
- `--separator`: CSV separator (default: ","),
- `--output`: Save results to file;

### Logistic Regression:

Binary classification

```bash
python logistic_regression_tool.py data.csv target_column --predictor_columns feature1 feature2
```

**Options:**
- `--test_size`: Train/test split ratio (default: 0.2),
- `--solver`: Optimization algorithm (default: 'lbfgs'),
- `--max_iter`: Maximum iterations (default: 1000);

### Power Analysis:

Calculate sample size, effect size, or power

```bash
# Calculate required sample size for t-test
python power_analysis.py ttest --effect_size 0.5 --power 0.8

# Calculate power for ANOVA
python power_analysis.py anova --effect_size 0.3 --sample_size 30 --groups 3

# Calculate effect size for chi-square test
python power_analysis.py chisquare --sample_size 100 --power 0.8 --df 2
```

**Test Types:**
- `ttest`: T-test (one-sample, two-sample, paired),
- `anova`: One-way ANOVA,
- `chisquare`: Chi-square test,
- `correlation`: Correlation test;

### Histogram:

Create distribution visualizations

```bash
python histogram.py data.csv column_name --bins 30 -o histogram.png
```

**Options:**
- `--bins`: Number of bins (default: 30),
- `--no-kde`: Disable kernel density estimate,
- `--color`: Histogram color (default: 'skyblue'),
- `--separator`: CSV separator (default: ",");

### Dependency Graph:

Visualize relationships between variables

```bash
python dependency_graph.py data.csv x_column y_column -o graph.png

# With regression line and correlation stats
python dependency_graph.py data.csv x_column y_column --regression --correlation
```

**Plot Types:**
- `scatter`: Scatter plot (default),
- `reg`: Regression plot,
- `hex`: Hexbin plot,
- `kde`: Kernel density estimate plot;

**Options:**
- `--plot-type`: Type of visualization,
- `--regression`: Show regression line,
- `--correlation`: Display correlation statistics;

### Mean Comparison Graph:

Compare statistics across multiple columns

```bash
python mean_comparison_graph.py data.csv col1 col2 col3 --type mean -o comparison.png

# Show all statistics
python mean_comparison_graph.py data.csv col1 col2 col3 --type all

# Group by categorical variable
python mean_comparison_graph.py data.csv col1 col2 --type mean --group-by category_column
```

**Statistics Types:**
- `mean`: Compare means,
- `std`: Compare standard deviations,
- `variance`: Compare variances,
- `all`: Show all three statistics;

### Basic Statistics:

Generate comprehensive statistical summaries

```bash
python basic_statistics.py data.csv ";" --output statistics.txt

# Treat specific columns as categorical
python basic_statistics.py data.csv ";" "zip_code,product_id"
```

## Output:

All statistical tests provide detailed output including:
- Test statistics and p-values,
- Effect sizes and confidence intervals,
- Assumption checks (when applicable),
- Post-hoc analyses (when significant),
- Clear interpretations and recommendations;

Results can be saved to text files using the `--output` parameter.

## Data Format:

### CSV Requirements:
- First row should contain column headers,
- Numeric columns for quantitative analyses,
- Categorical columns for chi-square and grouping variables,
- Missing values are automatically handled (rows removed);

### Wide vs Long Format:

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

## Examples:

### ML Clustering Workflow
```bash
# Quick K-Means clustering
python ml_clustering_tool.py customer_data.csv --algorithm kmeans --n_clusters 3 --output kmeans_results.txt

# Find optimal number of clusters
python ml_clustering_tool.py customer_data.csv --algorithm kmeans --elbow_analysis --silhouette_analysis

# Complete clustering analysis with visualizations
python ml_clustering_tool.py customer_data.csv \
  --algorithm kmeans \
  --n_clusters 4 \
  --feature_columns age income purchase_frequency recency \
  --visualize \
  --output cluster_analysis.txt \
  --output_data customers_with_clusters.csv

# Density-based clustering to find outliers
python ml_clustering_tool.py sensor_data.csv \
  --algorithm dbscan \
  --eps 0.3 \
  --min_samples 10 \
  --visualize \
  --output outlier_detection.txt

# Hierarchical clustering with complete linkage
python ml_clustering_tool.py gene_expression.csv \
  --algorithm agglomerative \
  --n_clusters 5 \
  --linkage complete \
  --visualize \
  --output hierarchical_results.txt
```

### ML Classification Workflow
```bash
# Quick single algorithm test
python ml_classifier_tool.py customer_data.csv churn --algorithm random_forest --output rf_results.txt

# Compare all algorithms to find the best
python ml_classifier_tool.py customer_data.csv churn --algorithm all --cross_validation --output comparison.txt

# Fine-tuned analysis with specific features
python ml_classifier_tool.py customer_data.csv churn \
  --algorithm gradient_boost \
  --predictor_columns age income purchase_frequency tenure \
  --test_size 0.25 \
  --cross_validation \
  --cv_folds 10 \
  --output final_model.txt
```

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

## Tips:

1. **Check assumptions**: Most tests include automatic assumption checking
2. **Save outputs**: Use `--output` to save detailed results
3. **Choose appropriate tests**: Use non-parametric tests (Friedman, Mann-Whitney) when assumptions are violated
4. **Consider effect sizes**: Statistical significance ≠ practical significance
5. **Algorithm selection**: Start with `--algorithm all` to compare, then focus on best performers
6. **Feature engineering**: ML tools handle categorical encoding automatically
7. **Cross-validation**: Always use `--cross_validation` for robust performance estimates
8. **Overfitting check**: Monitor train vs test performance differences
9. **Clustering validation**: Use elbow and silhouette analysis to determine optimal number of clusters
10. **Standardization**: Always standardize features for distance-based algorithms (K-Means, DBSCAN, K-Medoids)

## License

This project is open source and available for academic and research purposes.

## Contributing

Contributions are welcome! Please ensure all statistical tests follow best practices and include comprehensive output.

## Support

For issues or questions, please open an issue on the GitHub repository.
