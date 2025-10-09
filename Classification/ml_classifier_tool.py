import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import warnings
import time

AVAILABLE_ALGORITHMS = {
    'logistic': LogisticRegression,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'gradient_boost': GradientBoostingClassifier,
    'svm': SVC,
    'knn': KNeighborsClassifier,
    'naive_bayes': GaussianNB,
    'neural_network': MLPClassifier
}

def get_model_params(algorithm, random_state=42):
    """Get default parameters for each algorithm."""
    params = {
        'logistic': {'max_iter': 1000, 'random_state': random_state},
        'decision_tree': {'random_state': random_state, 'max_depth': 10},
        'random_forest': {'n_estimators': 100, 'random_state': random_state, 'max_depth': 10},
        'gradient_boost': {'n_estimators': 100, 'random_state': random_state, 'max_depth': 5},
        'svm': {'probability': True, 'random_state': random_state, 'max_iter': 1000},
        'knn': {'n_neighbors': 5},
        'naive_bayes': {},
        'neural_network': {'hidden_layer_sizes': (100,), 'max_iter': 1000, 'random_state': random_state}
    }
    return params.get(algorithm, {})

def perform_classification(csv_file, target_column, algorithm='logistic', 
                          predictor_columns=None, test_size=0.2, random_state=42, 
                          standardize=True, separator=',', cross_validation=False, cv_folds=5):
    """
    Performs classification analysis on data from a CSV file using various ML algorithms.
    
    Parameters:
    csv_file (str): Path to the CSV file.
    target_column (str): Name of the target/dependent variable column.
    algorithm (str): ML algorithm to use ('logistic', 'decision_tree', 'random_forest', 
                    'gradient_boost', 'svm', 'knn', 'naive_bayes', 'neural_network', 'all').
    predictor_columns (list, optional): List of predictor column names.
    test_size (float, optional): Proportion of data for testing (default: 0.2).
    random_state (int, optional): Random state for reproducibility (default: 42).
    standardize (bool, optional): Whether to standardize features (default: True).
    separator (str, optional): Separator used in CSV file (default: ',').
    cross_validation (bool, optional): Whether to perform cross-validation (default: False).
    cv_folds (int, optional): Number of cross-validation folds (default: 5).
    
    Returns:
    dict: Results containing model performance metrics and statistics.
    """
    try:
        data = pd.read_csv(csv_file, sep=separator)
        print(f"Loaded data with shape: {data.shape}")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")
        
        if data[target_column].isnull().any():
            print(f"Warning: Found {data[target_column].isnull().sum()} missing values in target column. Removing these rows.")
            data = data.dropna(subset=[target_column])
        
        y = data[target_column].copy()
        
        label_encoder = None
        if not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print(f"Target variable encoded. Classes: {label_encoder.classes_}")
        
        unique_values = np.unique(y)
        n_classes = len(unique_values)
        print(f"Number of classes: {n_classes}")
        
        if n_classes < 2:
            raise ValueError("Classification requires at least 2 classes.")
        
        if predictor_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)
            predictor_columns = numeric_columns
            print(f"Using all numeric columns as predictors: {predictor_columns}")
        else:
            missing_columns = [col for col in predictor_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Predictor columns do not exist in dataset: {missing_columns}")
        
        if not predictor_columns:
            raise ValueError("No predictor columns available for analysis.")
        
        X = data[predictor_columns].copy()
        
        if X.isnull().any().any():
            missing_info = X.isnull().sum()
            missing_cols = missing_info[missing_info > 0]
            print(f"Warning: Found missing values in predictor columns:\n{missing_cols}")
            print("Removing rows with missing predictor values.")
            X = X.dropna()
            y = y[X.index]
        
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            print(f"Encoding categorical columns: {categorical_columns}")
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            predictor_columns = X.columns.tolist()
        
        if len(X) < 30:
            raise ValueError("Insufficient data for analysis. Need at least 30 observations after cleaning.")
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Target variable distribution: {np.bincount(y)}")
        
        # Split data into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            print("Warning: Could not stratify split. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Standardize features if requested
        scaler = None
        if standardize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Train models
        if algorithm == 'all':
            algorithms_to_run = list(AVAILABLE_ALGORITHMS.keys())
        else:
            if algorithm not in AVAILABLE_ALGORITHMS:
                raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(AVAILABLE_ALGORITHMS.keys())}")
            algorithms_to_run = [algorithm]
        
        all_results = {}
        
        for algo_name in algorithms_to_run:
            print(f"\n{'='*60}")
            print(f"Training {algo_name.upper().replace('_', ' ')}...")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            model_class = AVAILABLE_ALGORITHMS[algo_name]
            model_params = get_model_params(algo_name, random_state)
            
            # Initialize and train model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model_class(**model_params)
                model.fit(X_train_scaled, y_train)
            
            training_time = time.time() - start_time
            
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Predict probabilities (if available)
            y_prob_train = None
            y_prob_test = None
            if hasattr(model, 'predict_proba'):
                y_prob_train = model.predict_proba(X_train_scaled)
                y_prob_test = model.predict_proba(X_test_scaled)
            
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            train_precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            train_recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            train_auc = None
            test_auc = None
            if n_classes == 2 and y_prob_test is not None:
                train_auc = roc_auc_score(y_train, y_prob_train[:, 1])
                test_auc = roc_auc_score(y_test, y_prob_test[:, 1])
            
            cm_test = confusion_matrix(y_test, y_pred_test)
            
            class_report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            
            cv_scores = None
            if cross_validation:
                print(f"Performing {cv_folds}-fold cross-validation...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
            
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': predictor_columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'coef_'):
                coefficients = model.coef_[0] if n_classes == 2 else np.mean(np.abs(model.coef_), axis=0)
                feature_importance = pd.DataFrame({
                    'feature': predictor_columns,
                    'coefficient': coefficients,
                    'abs_coefficient': np.abs(coefficients)
                }).sort_values('abs_coefficient', ascending=False)
            
            results = {
                'algorithm': algo_name,
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'feature_columns': predictor_columns,
                'target_column': target_column,
                'n_classes': n_classes,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_precision': train_precision,
                'test_precision': test_precision,
                'train_recall': train_recall,
                'test_recall': test_recall,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'train_auc': train_auc,
                'test_auc': test_auc,
                'confusion_matrix': cm_test,
                'classification_report': class_report,
                'feature_importance': feature_importance,
                'train_size': len(y_train),
                'test_size': len(y_test),
                'n_features': len(predictor_columns),
                'standardized': standardize,
                'training_time': training_time,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores) if cv_scores is not None else None,
                'cv_std': np.std(cv_scores) if cv_scores is not None else None
            }
            
            all_results[algo_name] = results
            print(f"Training completed in {training_time:.2f} seconds")
        
        return all_results if algorithm == 'all' else all_results[algorithm]
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_results(results, output_file=None):
    """Print formatted results of classification analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        import sys
        import os
        
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')
    
    # Handle multiple algorithms
    if isinstance(results, dict) and 'algorithm' not in results:
        # Multiple algorithms were run
        print("\nCLASSIFICATION ANALYSIS - ALGORITHM COMPARISON")
        print("=" * 100)
        
        # Create comparison table
        print(f"\n{'Algorithm':<20} {'Test Acc':<10} {'Test F1':<10} {'Test Prec':<12} {'Test Recall':<12} {'Time (s)':<10}")
        print("-" * 100)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        for algo_name, result in sorted_results:
            print(f"{algo_name.replace('_', ' ').title():<20} "
                  f"{result['test_accuracy']:<10.4f} "
                  f"{result['test_f1']:<10.4f} "
                  f"{result['test_precision']:<12.4f} "
                  f"{result['test_recall']:<12.4f} "
                  f"{result['training_time']:<10.2f}")
        
        print("\n" + "=" * 100)
        print(f"Best performing algorithm: {sorted_results[0][0].replace('_', ' ').title()}")
        print(f"Best test accuracy: {sorted_results[0][1]['test_accuracy']:.4f}")
        print("=" * 100)
        
        for algo_name, result in sorted_results:
            print(f"\n\n{'#'*100}")
            print_single_result(result)
    else:
        print_single_result(results)
    
    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")

def print_single_result(result):
    """Print results for a single algorithm."""
    print(f"\n{result['algorithm'].upper().replace('_', ' ')} - CLASSIFICATION RESULTS")
    print("=" * 100)
    print(f"Target variable: {result['target_column']}")
    print(f"Number of classes: {result['n_classes']}")
    print(f"Number of features: {result['n_features']}")
    print(f"Training set size: {result['train_size']}")
    print(f"Test set size: {result['test_size']}")
    print(f"Features standardized: {result['standardized']}")
    print(f"Training time: {result['training_time']:.2f} seconds")
    print("=" * 100)
    
    print("\nMODEL PERFORMANCE:")
    print("-" * 50)
    print(f"Training Accuracy:   {result['train_accuracy']:.4f}")
    print(f"Test Accuracy:       {result['test_accuracy']:.4f}")
    print(f"Training Precision:  {result['train_precision']:.4f}")
    print(f"Test Precision:      {result['test_precision']:.4f}")
    print(f"Training Recall:     {result['train_recall']:.4f}")
    print(f"Test Recall:         {result['test_recall']:.4f}")
    print(f"Training F1-Score:   {result['train_f1']:.4f}")
    print(f"Test F1-Score:       {result['test_f1']:.4f}")
    
    if result['train_auc'] is not None:
        print(f"Training AUC-ROC:    {result['train_auc']:.4f}")
        print(f"Test AUC-ROC:        {result['test_auc']:.4f}")
    
    if result['cv_scores'] is not None:
        print("\nCROSS-VALIDATION RESULTS:")
        print("-" * 50)
        print(f"CV Mean Accuracy: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        print(f"CV Scores: {[f'{score:.4f}' for score in result['cv_scores']]}")
    
    print("\nCONFUSION MATRIX (Test Set):")
    print("-" * 50)
    cm = result['confusion_matrix']
    
    if result['n_classes'] == 2:
        print(f"True Negative:  {cm[0,0]:4d}  |  False Positive: {cm[0,1]:4d}")
        print(f"False Negative: {cm[1,0]:4d}  |  True Positive:  {cm[1,1]:4d}")
    else:
        print(pd.DataFrame(cm))
    
    print("\nCLASSIFICATION METRICS (Test Set):")
    print("-" * 50)
    cr = result['classification_report']
    
    for class_label in sorted([k for k in cr.keys() if k.isdigit() or k == '0']):
        print(f"Class {class_label}:")
        print(f"  Precision: {cr[class_label]['precision']:.4f}")
        print(f"  Recall:    {cr[class_label]['recall']:.4f}")
        print(f"  F1-Score:  {cr[class_label]['f1-score']:.4f}")
        print(f"  Support:   {cr[class_label]['support']}")
    
    if 'macro avg' in cr:
        print(f"\nMacro Average F1-Score:    {cr['macro avg']['f1-score']:.4f}")
        print(f"Weighted Average F1-Score: {cr['weighted avg']['f1-score']:.4f}")
    
    if result['feature_importance'] is not None:
        print("\nFEATURE IMPORTANCE (Top 15):")
        print("-" * 70)
        
        if 'importance' in result['feature_importance'].columns:
            print(f"{'Feature':<35} {'Importance':<15}")
            print("-" * 70)
            top_features = result['feature_importance'].head(15)
            for _, row in top_features.iterrows():
                print(f"{row['feature']:<35} {row['importance']:<15.6f}")
        else:
            print(f"{'Feature':<35} {'Coefficient':<15}")
            print("-" * 70)
            top_features = result['feature_importance'].head(15)
            for _, row in top_features.iterrows():
                print(f"{row['feature']:<35} {row['coefficient']:<15.6f}")
    
    print("\nMODEL INTERPRETATION:")
    print("-" * 50)
    
    if result['test_accuracy'] > 0.85:
        performance = "excellent"
    elif result['test_accuracy'] > 0.75:
        performance = "good"
    elif result['test_accuracy'] > 0.65:
        performance = "moderate"
    else:
        performance = "limited"
    
    print(f"The model shows {performance} predictive performance.")
    
    if result['test_accuracy'] < 0.7:
        print("\nConsider improving the model by:")
        print("- Adding more relevant features")
        print("- Feature engineering")
        print("- Hyperparameter tuning")
        print("- Trying different algorithms")
        print("- Collecting more training data")
    
    # Check for overfitting
    accuracy_diff = result['train_accuracy'] - result['test_accuracy']
    if accuracy_diff > 0.1:
        print(f"\nWarning: Possible overfitting detected (train-test accuracy gap: {accuracy_diff:.4f})")
        print("Consider:")
        print("- Using regularization")
        print("- Reducing model complexity")
        print("- Collecting more training data")

def main():
    parser = argparse.ArgumentParser(
        description='Machine Learning Classification Tool - Supports multiple algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Algorithms:
  logistic        - Logistic Regression
  decision_tree   - Decision Tree Classifier
  random_forest   - Random Forest Classifier
  gradient_boost  - Gradient Boosting Classifier
  svm             - Support Vector Machine
  knn             - K-Nearest Neighbors
  naive_bayes     - Naive Bayes
  neural_network  - Multi-Layer Perceptron (Neural Network)
  all             - Run all algorithms and compare

Examples:
  python ml_classifier_tool.py data.csv target_column --algorithm random_forest
  python ml_classifier_tool.py data.csv target_column --algorithm all --cross_validation
  python ml_classifier_tool.py data.csv target_column --algorithm svm --output results.txt
        """
    )
    
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('target_column', type=str, help='Name of the target/dependent variable column')
    parser.add_argument('--algorithm', type=str, default='logistic',
                        choices=['logistic', 'decision_tree', 'random_forest', 'gradient_boost',
                                'svm', 'knn', 'naive_bayes', 'neural_network', 'all'],
                        help='ML algorithm to use (default: logistic)')
    parser.add_argument('--predictor_columns', type=str, nargs='+', default=None,
                        help='Names of predictor columns (if not specified, uses all numeric columns)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--no_standardize', action='store_true',
                        help='Do not standardize features (default: standardize)')
    parser.add_argument('--separator', type=str, default=',',
                        help='Separator used in CSV file (default: ",")')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    
    args = parser.parse_args()
    
    results = perform_classification(
        args.csv_file, args.target_column, args.algorithm,
        args.predictor_columns, args.test_size, args.random_state,
        not args.no_standardize, args.separator, args.cross_validation, args.cv_folds
    )
    
    print_results(results, args.output)

if __name__ == "__main__":
    main()
