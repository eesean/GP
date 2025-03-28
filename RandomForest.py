import json
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import os

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the obesity prediction dataset.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file

    Returns:
    --------
    X_train_scaled : numpy array
        Scaled training features
    X_test_scaled : numpy array
        Scaled testing features
    y_train : numpy array
        Training labels
    y_test : numpy array
        Testing labels
    feature_columns : list
        Names of the features used
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        print("Dataset Shape:", df.shape)
        print("\nData Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isnull().sum())

        # Check for required columns
        required_columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC',
                            'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
                            'TUE', 'CALC', 'MTRANS', 'Obesity']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None, None, None, None, None

        # Create BMI feature
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

        # Create interaction features
        df['Activity_Tech_Balance'] = df['FAF'] / (df['TUE'] + 1)  # Add 1 to avoid division by zero
        df['Meal_Water_Ratio'] = df['NCP'] / (df['CH2O'] + 1)

        # Encode categorical variables
        le = LabelEncoder()
        categorical_columns = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'Obesity']

        for column in categorical_columns:
            df[column + '_encoded'] = le.fit_transform(df[column])

        # Select features for the model
        feature_columns = [col for col in df.columns if col.endswith('_encoded') and col != 'Obesity_encoded']
        feature_columns.extend(['BMI', 'Activity_Tech_Balance', 'Meal_Water_Ratio',
                                'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'])

        X = df[feature_columns]
        y = df['Obesity_encoded']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("\nFeatures used:", feature_columns)
        print("\nNumber of classes:", len(np.unique(y)))

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

    except Exception as e:
        print(f"Error processing the data: {str(e)}")
        return None, None, None, None, None

def test_for_overfitting(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, cv=5):
    """
    Test a model for overfitting using a RandomForestClassifier.

    Parameters:
    -----------
    X_train_scaled : numpy array
        Scaled training features
    X_test_scaled : numpy array
        Scaled testing features
    y_train : numpy array
        Training labels
    y_test : numpy array
        Testing labels
    feature_columns : list
        Names of the features used
    cv : int
        Number of cross-validation folds

    Returns:
    --------
    results : dict
        Dictionary containing training and testing results
    """
    # Define parameter grid for random forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    # Create and fit grid search
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = grid_search.predict(X_train_scaled)
    y_pred_test = grid_search.predict(X_test_scaled)

    # Calculate metrics
    train_accuracy = (y_pred_train == y_train).mean()
    test_accuracy = (y_pred_test == y_test).mean()

    # Store results
    results = {
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'accuracy_diff': train_accuracy - test_accuracy,
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'feature_importance': pd.DataFrame({
            'feature': feature_columns,
            'importance': grid_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False),
        'model': grid_search.best_estimator_
    }

    return results

def visualize_results(results, X_train_scaled, y_train, X_test_scaled, y_test, feature_columns):
    """
    Visualize model results and check for overfitting.

    Parameters:
    -----------
    results : dict
        Results from the test_for_overfitting function
    X_train_scaled : numpy array
        Scaled training features
    y_train : numpy array
        Training labels
    X_test_scaled : numpy array
        Scaled testing features
    y_test : numpy array
        Testing labels
    feature_columns : list
        Names of features
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot confusion matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # Plot feature importance
    top_10_features = results['feature_importance'].head(10)
    sns.barplot(data=top_10_features, x='importance', y='feature', ax=axes[0, 1])
    axes[0, 1].set_title('Top 10 Most Important Features')

    # Plot learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        results['model'], X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    axes[1, 0].plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    axes[1, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    axes[1, 0].plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    axes[1, 0].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    axes[1, 0].set_title('Learning Curve')
    axes[1, 0].set_xlabel('Training Size')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True)

    # Plot comparison of train vs test accuracy
    comparison = pd.DataFrame({
        'Dataset': ['Training', 'Testing', 'Cross-Validation'],
        'Accuracy': [results['train_accuracy'], results['test_accuracy'], results['cv_score']]
    })

    sns.barplot(data=comparison, x='Dataset', y='Accuracy', ax=axes[1, 1])
    axes[1, 1].set_title('Accuracy Comparison')
    axes[1, 1].set_ylim(max(0.8, min(comparison['Accuracy']) - 0.05), 1.01)  # Adjust y-axis to better see differences

    # Display metric values on the plot
    for i, v in enumerate(comparison['Accuracy']):
        axes[1, 1].text(i, v + 0.01, f"{v:.4f}", ha='center')

    # Add text with overfitting diagnosis
    overfitting_threshold = 0.05  # Threshold for significant overfitting
    if results['accuracy_diff'] > overfitting_threshold:
        overfitting_status = f"SIGNIFICANT OVERFITTING DETECTED\nTrain-Test Gap: {results['accuracy_diff']:.4f}"
        color = 'red'
    elif results['accuracy_diff'] > 0.02:
        overfitting_status = f"MILD OVERFITTING\nTrain-Test Gap: {results['accuracy_diff']:.4f}"
        color = 'orange'
    else:
        overfitting_status = f"NO SIGNIFICANT OVERFITTING\nTrain-Test Gap: {results['accuracy_diff']:.4f}"
        color = 'green'

    fig.text(0.5, 0.02, overfitting_status, ha='center', fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1', edgecolor=color))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle(f"Model Evaluation - {len(np.unique(y_test))} Classes, {len(feature_columns)} Features",
                 fontsize=16)
    plt.subplots_adjust(top=0.92)

    return fig

def estimate_overfitting(results):
    """
    Estimate the level of overfitting and provide recommendations.

    Parameters:
    -----------
    results : dict
        Results from the test_for_overfitting function

    Returns:
    --------
    diagnosis : dict
        Dictionary with overfitting diagnosis and recommendations
    """
    train_acc = results['train_accuracy']
    test_acc = results['test_accuracy']
    cv_score = results['cv_score']
    acc_diff = train_acc - test_acc

    # Define thresholds for overfitting
    severe_threshold = 0.1
    moderate_threshold = 0.05
    mild_threshold = 0.02

    # Assess overfitting level
    if acc_diff > severe_threshold:
        level = "Severe"
        confidence = "High"
    elif acc_diff > moderate_threshold:
        level = "Moderate"
        confidence = "Medium-High"
    elif acc_diff > mild_threshold:
        level = "Mild"
        confidence = "Medium"
    else:
        level = "Minimal or None"
        confidence = "Low"

    # Generate recommendations based on overfitting level
    recommendations = []

    if level != "Minimal or None":
        recommendations.append("Consider using stronger regularization techniques")
        recommendations.append("Try reducing model complexity (fewer estimators or shallower trees)")

        if results['best_params']['max_depth'] is None or results['best_params']['max_depth'] > 10:
            recommendations.append("Limit tree depth to prevent overfitting")

        if level == "Severe":
            recommendations.append("Collect more training data if possible")
            recommendations.append("Try feature selection to reduce dimensionality")
            recommendations.append("Consider a simpler model architecture")
    else:
        recommendations.append("Current model shows good generalization")

        if train_acc == 1.0 and test_acc > 0.98:
            recommendations.append("The task might be too easy - consider a more challenging problem or add noise")

    # Calculate feature concentration (how much the model relies on top features)
    top3_importance = results['feature_importance'].head(3)['importance'].sum()

    if top3_importance > 0.7:
        recommendations.append(
            f"Model relies heavily on top 3 features ({top3_importance:.2f} importance) - investigate potential data leakage")

    # Analyze performance by class
    class_report = results['classification_report']
    class_f1_scores = {class_idx: class_report[str(class_idx)]['f1-score']
                       for class_idx in range(len(class_report) - 3)}  # Exclude 'accuracy', 'macro avg', 'weighted avg'

    worst_class = min(class_f1_scores.items(), key=lambda x: x[1])
    if worst_class[1] < 0.9 and worst_class[1] < min(train_acc, test_acc) - 0.1:
        recommendations.append(
            f"Poor performance on class {worst_class[0]} (F1={worst_class[1]:.2f}) - consider class balancing")

    # Create diagnosis dictionary
    diagnosis = {
        'overfitting_level': level,
        'confidence': confidence,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_score': cv_score,
        'accuracy_difference': acc_diff,
        'recommendations': recommendations,
        'top_features': results['feature_importance'].head(5).to_dict(),
        'class_performance': class_f1_scores
    }

    return diagnosis

def compare_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns):
    """
    Compare different model configurations to address overfitting.

    Parameters:
    -----------
    X_train_scaled : numpy array
        Scaled training features
    X_test_scaled : numpy array
        Scaled testing features
    y_train : numpy array
        Training labels
    y_test : numpy array
        Testing labels
    feature_columns : list
        Names of the features used

    Returns:
    --------
    comparison_results : dict
        Dictionary containing comparison results
    """
    models = {
        'Default RF': RandomForestClassifier(random_state=42),
        'Regularized RF': RandomForestClassifier(
            max_depth=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        # Fit the model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Calculate metrics
        train_accuracy = (y_pred_train == y_train).mean()
        test_accuracy = (y_pred_test == y_test).mean()

        # Cross-validation score
        cv_score = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring='accuracy'
        ).mean()

        # Store results
        results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_score': cv_score,
            'accuracy_diff': train_accuracy - test_accuracy,
            'model': model
        }

    # Feature selection using the best model from above
    best_model_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_model = results[best_model_name]['model']

    # Use SelectFromModel for feature selection
    selector = SelectFromModel(best_model, threshold='mean')
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_columns[i] for i in selected_indices]

    # Train model with selected features
    feature_selected_model = models[best_model_name].__class__(**models[best_model_name].get_params())
    feature_selected_model.fit(X_train_selected, y_train)

    # Make predictions
    y_pred_train = feature_selected_model.predict(X_train_selected)
    y_pred_test = feature_selected_model.predict(X_test_selected)

    # Calculate metrics
    train_accuracy = (y_pred_train == y_train).mean()
    test_accuracy = (y_pred_test == y_test).mean()

    # Cross-validation score
    cv_score = cross_val_score(
        feature_selected_model, X_train_selected, y_train, cv=5, scoring='accuracy'
    ).mean()

    # Add to results
    results['Feature Selection'] = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_score': cv_score,
        'accuracy_diff': train_accuracy - test_accuracy,
        'model': feature_selected_model,
        'selected_features': selected_features,
        'n_features': len(selected_features)
    }

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train Accuracy': [results[model]['train_accuracy'] for model in results],
        'Test Accuracy': [results[model]['test_accuracy'] for model in results],
        'CV Score': [results[model]['cv_score'] for model in results],
        'Accuracy Diff': [results[model]['accuracy_diff'] for model in results]
    })

    # Sort by test accuracy
    comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)

    return {
        'comparison_df': comparison_df,
        'detailed_results': results
    }


def visualize_model_comparison(comparison_results):
    """
    Visualize model comparison results.

    Parameters:
    -----------
    comparison_results : dict
        Results from compare_models function

    Returns:
    --------
    fig : matplotlib figure
        Figure with model comparison visualizations
    """
    comparison_df = comparison_results['comparison_df']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot comparison of model accuracies
    comparison_melted = pd.melt(
        comparison_df,
        id_vars=['Model'],
        value_vars=['Train Accuracy', 'Test Accuracy', 'CV Score'],
        var_name='Metric', value_name='Accuracy'
    )

    sns.barplot(data=comparison_melted, x='Model', y='Accuracy', hue='Metric', ax=axes[0])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylim(0.9, 1.01)  # Adjust y-axis to better see differences
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('')

    # Plot accuracy difference (measure of overfitting)
    sns.barplot(data=comparison_df, x='Model', y='Accuracy Diff', ax=axes[1])
    axes[1].set_title('Train-Test Accuracy Gap (Overfitting Measure)')
    axes[1].set_ylabel('Train Accuracy - Test Accuracy')
    axes[1].set_xlabel('')

    # Display metric values on the plot
    for i, model in enumerate(comparison_df['Model']):
        axes[1].text(i, comparison_df.iloc[i]['Accuracy Diff'] + 0.001,
                     f"{comparison_df.iloc[i]['Accuracy Diff']:.4f}", ha='center')

    # Add information about feature selection
    if 'Feature Selection' in comparison_results['detailed_results']:
        fs_results = comparison_results['detailed_results']['Feature Selection']
        fig.text(0.5, 0.01,
                 f"Feature Selection: Reduced from {len(fs_results['selected_features'])} to "
                 f"{fs_results['n_features']} features",
                 ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle("Model Comparison for Addressing Overfitting", fontsize=16)
    plt.subplots_adjust(top=0.9)

    return fig

class ResultsManager:
    """
    A class to manage and store intermediate results from ML model analysis.
    """

    def __init__(self, base_dir="results", experiment_name=None):
        """
        Initialize the ResultsManager with a base directory.

        Parameters:
        -----------
        base_dir : str
            Base directory for storing results
        experiment_name : str or None
            Name of the experiment; if None, a timestamp will be used
        """
        # Create a timestamp for unique experiment identifier if not provided
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_dir = os.path.join(base_dir, experiment_name)

        # Create directory structure for organizing results
        self.dirs = {
            'data': os.path.join(self.experiment_dir, 'data'),
            'models': os.path.join(self.experiment_dir, 'models'),
            'metrics': os.path.join(self.experiment_dir, 'metrics'),
            'visualizations': os.path.join(self.experiment_dir, 'visualizations'),
            'overfitting_analysis': os.path.join(self.experiment_dir, 'overfitting_analysis')
        }

        # Create all directories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        print(f"Results will be stored in: {self.experiment_dir}")

    def store_data_stats(self, df_stats, feature_columns, X_train_shape, X_test_shape):
        """
        Store dataset statistics.

        Parameters:
        -----------
        df_stats : dict
            Dictionary containing dataset statistics
        feature_columns : list
            List of feature column names
        X_train_shape : tuple
            Shape of the training data
        X_test_shape : tuple
            Shape of the testing data
        """
        data_stats = {
            'dataset_stats': df_stats,
            'feature_columns': feature_columns,
            'train_samples': X_train_shape[0],
            'test_samples': X_test_shape[0],
            'n_features': len(feature_columns),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save as JSON
        with open(os.path.join(self.dirs['data'], 'data_stats.json'), 'w') as f:
            json.dump(data_stats, f, indent=4, default=self._serialize_numpy)

        # Save feature columns as CSV
        pd.DataFrame({'feature': feature_columns}).to_csv(
            os.path.join(self.dirs['data'], 'features.csv'), index=False
        )

        print(f"Dataset statistics stored in {self.dirs['data']}")

    def store_model(self, model, model_name="model"):
        """
        Store a trained model.

        Parameters:
        -----------
        model : estimator
            Trained sklearn model
        model_name : str
            Name for the model file
        """
        # Save the model using pickle
        model_path = os.path.join(self.dirs['models'], f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Store model parameters
        model_params = {
            'model_type': type(model).__name__,
            'parameters': model.get_params(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        params_path = os.path.join(self.dirs['models'], f"{model_name}_params.json")
        with open(params_path, 'w') as f:
            json.dump(model_params, f, indent=4)

        print(f"Model stored at {model_path}")
        return model_path

    def store_metrics(self, results, model_name="model"):
        """
        Store model metrics.

        Parameters:
        -----------
        results : dict
            Dictionary containing model metrics
        model_name : str
            Name for the metrics file
        """
        # Extract key metrics
        metrics = {
            'train_accuracy': results.get('train_accuracy'),
            'test_accuracy': results.get('test_accuracy'),
            'cv_score': results.get('cv_score'),
            'accuracy_diff': results.get('accuracy_diff'),
            'best_params': results.get('best_params'),
            'classification_report': results.get('classification_report'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save as JSON
        metrics_path = os.path.join(self.dirs['metrics'], f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=self._serialize_numpy)

        # Save feature importance if available
        if 'feature_importance' in results:
            importance_df = results['feature_importance']
            importance_path = os.path.join(self.dirs['metrics'], f"{model_name}_feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)

        print(f"Metrics stored at {metrics_path}")
        return metrics_path

    def store_visualizations(self, fig, name="visualization"):
        """
        Store visualizations.

        Parameters:
        -----------
        fig : matplotlib figure
            Figure to save
        name : str
            Name for the figure file
        """
        # Save as PNG and PDF
        png_path = os.path.join(self.dirs['visualizations'], f"{name}.png")
        pdf_path = os.path.join(self.dirs['visualizations'], f"{name}.pdf")

        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')

        print(f"Visualization stored at {png_path} and {pdf_path}")
        return png_path

    def store_overfitting_analysis(self, diagnosis, comparison_results=None):
        """
        Store overfitting analysis results.

        Parameters:
        -----------
        diagnosis : dict
            Dictionary containing overfitting diagnosis
        comparison_results : dict or None
            Dictionary containing model comparison results
        """
        # Save diagnosis as JSON
        diagnosis_path = os.path.join(self.dirs['overfitting_analysis'], 'diagnosis.json')
        with open(diagnosis_path, 'w') as f:
            json.dump(diagnosis, f, indent=4, default=self._serialize_numpy)

        # Save evidence of non-overfitting
        evidence = self._extract_non_overfitting_evidence(diagnosis, comparison_results)
        evidence_path = os.path.join(self.dirs['overfitting_analysis'], 'non_overfitting_evidence.json')
        with open(evidence_path, 'w') as f:
            json.dump(evidence, f, indent=4, default=self._serialize_numpy)

        # If comparison results are available, save them
        if comparison_results:
            # Save comparison DataFrame to CSV
            comparison_path = os.path.join(self.dirs['overfitting_analysis'], 'model_comparison.csv')
            comparison_results['comparison_df'].to_csv(comparison_path, index=False)

        print(f"Overfitting analysis stored at {self.dirs['overfitting_analysis']}")
        return evidence_path

    def generate_summary_report(self):
        """
        Generate a summary report of all stored results.

        Returns:
        --------
        summary : dict
            Dictionary containing summary information
        """
        # Gather all metrics files
        metrics_files = [f for f in os.listdir(self.dirs['metrics']) if f.endswith('_metrics.json')]

        models_metrics = []
        for file in metrics_files:
            with open(os.path.join(self.dirs['metrics'], file), 'r') as f:
                metrics = json.load(f)
                model_name = file.replace('_metrics.json', '')
                metrics['model_name'] = model_name
                models_metrics.append(metrics)

        # Get overfitting diagnosis
        diagnosis_path = os.path.join(self.dirs['overfitting_analysis'], 'diagnosis.json')
        if os.path.exists(diagnosis_path):
            with open(diagnosis_path, 'r') as f:
                diagnosis = json.load(f)
        else:
            diagnosis = {}

        # Create summary
        summary = {
            'experiment_dir': self.experiment_dir,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_count': len(models_metrics),
            'models_metrics': models_metrics,
            'overfitting_diagnosis': diagnosis
        }

        # Save summary
        summary_path = os.path.join(self.experiment_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"Summary report generated at {summary_path}")
        return summary

    def _extract_non_overfitting_evidence(self, diagnosis, comparison_results=None):
        """
        Extract evidence of non-overfitting from diagnosis and comparison results.

        Parameters:
        -----------
        diagnosis : dict
            Dictionary containing overfitting diagnosis
        comparison_results : dict or None
            Dictionary containing model comparison results

        Returns:
        --------
        evidence : dict
            Dictionary containing evidence of non-overfitting
        """
        evidence = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evidence_points': []
        }

        # Check train-test gap
        train_test_gap = diagnosis.get('accuracy_difference', 1.0)
        if train_test_gap < 0.05:
            evidence['evidence_points'].append({
                'type': 'train_test_gap',
                'description': f"Small difference between training and testing accuracy: {train_test_gap:.4f}",
                'strength': 'Strong' if train_test_gap < 0.02 else 'Moderate'
            })

        # Check CV score vs test score
        cv_score = diagnosis.get('cv_score', 0)
        test_score = diagnosis.get('test_accuracy', 0)
        cv_test_diff = abs(cv_score - test_score)

        if cv_test_diff < 0.03:
            evidence['evidence_points'].append({
                'type': 'cv_test_consistency',
                'description': f"Consistent performance between CV and test set: diff={cv_test_diff:.4f}",
                'strength': 'Strong' if cv_test_diff < 0.01 else 'Moderate'
            })

        # Check overfitting level from diagnosis
        if diagnosis.get('overfitting_level') in ['Minimal or None', 'Mild']:
            evidence['evidence_points'].append({
                'type': 'diagnosis',
                'description': f"Overfitting diagnosis: {diagnosis.get('overfitting_level')}",
                'strength': 'Strong' if diagnosis.get('overfitting_level') == 'Minimal or None' else 'Moderate'
            })

        # If we have comparison results, check consistency across models
        if comparison_results and 'comparison_df' in comparison_results:
            df = comparison_results['comparison_df']

            # Calculate the range of test accuracies
            test_acc_range = df['Test Accuracy'].max() - df['Test Accuracy'].min()

            if test_acc_range < 0.05:
                evidence['evidence_points'].append({
                    'type': 'model_consistency',
                    'description': f"Consistent test performance across different model types: range={test_acc_range:.4f}",
                    'strength': 'Strong' if test_acc_range < 0.02 else 'Moderate'
                })

            # Find models with small train-test gaps
            good_models = df[df['Accuracy Diff'] < 0.05]
            if len(good_models) > 0:
                evidence['evidence_points'].append({
                    'type': 'multiple_good_models',
                    'description': f"Multiple models show little overfitting ({len(good_models)} models with gap < 0.05)",
                    'strength': 'Moderate'
                })

        # Add overall assessment
        if len(evidence['evidence_points']) >= 3:
            evidence['overall_assessment'] = 'Strong evidence of good generalization'
        elif len(evidence['evidence_points']) >= 1:
            evidence['overall_assessment'] = 'Moderate evidence of good generalization'
        else:
            evidence['overall_assessment'] = 'Insufficient evidence of good generalization'

        return evidence

    def _serialize_numpy(self, obj):
        """Helper method to serialize numpy types for JSON."""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)


# Example usage in modified main function
def enhanced_main(file_path=os.path.join('data', 'obecity_prediction.csv'),
                  experiment_name=None):
    """
    Enhanced main function that stores intermediate results.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    experiment_name : str or None
        Name for the experiment
    """
    # Initialize results manager
    results_manager = ResultsManager(experiment_name=experiment_name)

    print("Loading and preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_columns = load_and_preprocess_data(file_path)

    # Check if data loading failed
    if X_train_scaled is None or y_train is None:
        print(f"Error: Could not process data from file: {file_path}")
        return None, None

    # Store data statistics
    data_stats = {
        'file_path': file_path,
        'train_shape': X_train_scaled.shape,
        'test_shape': X_test_scaled.shape,
        'n_classes': len(np.unique(y_train))
    }
    results_manager.store_data_stats(data_stats, feature_columns,
                                     X_train_scaled.shape, X_test_scaled.shape)

    try:
        print("\nTesting for overfitting...")
        results = test_for_overfitting(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)

        print("\nModel Performance Summary:")
        print(f"Best Model Parameters: {results['best_params']}")
        print(f"Cross-validation Score: {results['cv_score']:.4f}")
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"Testing Accuracy: {results['test_accuracy']:.4f}")
        print(f"Accuracy Difference (Train - Test): {results['accuracy_diff']:.4f}")

        print("\nTop 5 Most Important Features:")
        print(results['feature_importance'].head())

        # Store the model and its metrics
        results_manager.store_model(results['model'], model_name="random_forest_best")
        results_manager.store_metrics(results, model_name="random_forest_best")

    except Exception as e:
        print(f"Error during model training and evaluation: {str(e)}")
        return None, None

    # Generate and print diagnosis
    diagnosis = estimate_overfitting(results)
    print("\nOverfitting Diagnosis:")
    print(f"Level: {diagnosis['overfitting_level']} (Confidence: {diagnosis['confidence']})")
    print("\nRecommendations:")
    for i, rec in enumerate(diagnosis['recommendations'], 1):
        print(f"{i}. {rec}")

    # Visualize results
    print("\nGenerating visualizations...")
    fig1 = visualize_results(results, X_train_scaled, y_train, X_test_scaled, y_test, feature_columns)
    results_manager.store_visualizations(fig1, name="model_evaluation")

    # Compare models if there's overfitting
    comparison_results = None
    if diagnosis['overfitting_level'] != "Minimal or None":
        print("\nComparing different models to address overfitting...")
        comparison_results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)

        print("\nModel Comparison Results:")
        print(comparison_results['comparison_df'])

        # Visualize model comparison
        fig2 = visualize_model_comparison(comparison_results)
        results_manager.store_visualizations(fig2, name="model_comparison")

        print("\nBest Model Based on Test Accuracy:")
        best_model = comparison_results['comparison_df'].iloc[0]['Model']
        print(f"- {best_model}")

        if best_model == 'Feature Selection':
            print(
                f"- Reduced from {len(feature_columns)} to {comparison_results['detailed_results']['Feature Selection']['n_features']} features")
            print("- Selected features:")
            for feature in comparison_results['detailed_results']['Feature Selection']['selected_features']:
                print(f"  * {feature}")

            # Store the feature-selected model
            fs_model = comparison_results['detailed_results']['Feature Selection']['model']
            results_manager.store_model(fs_model, model_name="feature_selected_model")

            # Create and store metrics for the feature-selected model
            fs_metrics = {
                'train_accuracy': comparison_results['detailed_results']['Feature Selection']['train_accuracy'],
                'test_accuracy': comparison_results['detailed_results']['Feature Selection']['test_accuracy'],
                'cv_score': comparison_results['detailed_results']['Feature Selection']['cv_score'],
                'accuracy_diff': comparison_results['detailed_results']['Feature Selection']['accuracy_diff'],
                'selected_features': comparison_results['detailed_results']['Feature Selection']['selected_features'],
                'n_original_features': len(feature_columns),
                'n_selected_features': comparison_results['detailed_results']['Feature Selection']['n_features']
            }
            results_manager.store_metrics(fs_metrics, model_name="feature_selected_model")

    # Store overfitting analysis and generate summary
    results_manager.store_overfitting_analysis(diagnosis, comparison_results)
    summary = results_manager.generate_summary_report()

    plt.show()
    return results, diagnosis, results_manager.experiment_dir


# Add a function specifically for recording evidence of not overfitting
def record_non_overfitting_evidence(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns,
                                    output_dir='non_overfitting_evidence'):
    """
    Specifically analyze and record evidence that a model is not overfitting.

    Parameters:
    -----------
    X_train_scaled : numpy array
        Scaled training features
    X_test_scaled : numpy array
        Scaled testing features
    y_train : numpy array
        Training labels
    y_test : numpy array
        Testing labels
    feature_columns : list
        Names of the features used
    output_dir : str
        Directory to save evidence

    Returns:
    --------
    evidence_summary : dict
        Summary of non-overfitting evidence
    """
    os.makedirs(output_dir, exist_ok=True)

    evidence_points = []
    evidence_plots = []

    # 1. Train multiple models with varying complexity
    print("Training models with varying complexity to check for stable performance...")
    models = {
        'Simple RF (5 trees)': RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42),
        'Medium RF (50 trees)': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'Complex RF (200 trees)': RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    }

    model_results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()

        model_results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_score': cv_score,
            'gap': train_acc - test_acc
        }

    # Check if performance is stable across model complexity
    test_accs = [res['test_acc'] for res in model_results.values()]
    test_acc_range = max(test_accs) - min(test_accs)

    if test_acc_range < 0.05:
        evidence_points.append({
            'title': 'Stable Performance Across Model Complexity',
            'description': f"Test accuracy range is only {test_acc_range:.4f} across models of varying complexity",
            'strength': 'Strong' if test_acc_range < 0.02 else 'Moderate'
        })

    # Check for modest train-test gaps across models
    gaps = [res['gap'] for res in model_results.values()]
    if all(gap < 0.05 for gap in gaps):
        evidence_points.append({
            'title': 'Consistent Small Train-Test Gaps',
            'description': f"All models show train-test accuracy gaps below 0.05 (max: {max(gaps):.4f})",
            'strength': 'Strong' if max(gaps) < 0.03 else 'Moderate'
        })

    # 2. Learning curve analysis
    print("Generating learning curves to analyze training dynamics...")
    fig, ax = plt.subplots(figsize=(10, 6))

    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(random_state=42),
        X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    ax.set_title('Learning Curve Analysis')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')
    ax.grid(True)

    # Check if curves converge
    final_gap = train_mean[-1] - test_mean[-1]
    if final_gap < 0.05:
        evidence_points.append({
            'title': 'Converging Learning Curves',
            'description': f"Training and validation curves converge with a final gap of {final_gap:.4f}",
            'strength': 'Strong' if final_gap < 0.03 else 'Moderate'
        })

    # Check if validation performance stabilizes
    test_diff = test_mean[-1] - test_mean[-3]
    if abs(test_diff) < 0.03:
        evidence_points.append({
            'title': 'Stable Validation Performance',
            'description': f"Validation performance stabilizes with little change ({test_diff:.4f}) in later iterations",
            'strength': 'Strong' if abs(test_diff) < 0.01 else 'Moderate'
        })

    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    evidence_plots.append('learning_curves.png')

    # 3. Check feature importance distribution
    print("Analyzing feature importance distribution...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)

    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]

    # Gini importance
    gini_sum_top5 = sum(importance[indices[:5]])
    gini_sum_top10 = sum(importance[indices[:10]])

    if gini_sum_top5 < 0.5:
        evidence_points.append({
            'title': 'Balanced Feature Importance',
            'description': f"Top 5 features comprise only {gini_sum_top5:.2f} of total importance",
            'strength': 'Strong' if gini_sum_top5 < 0.4 else 'Moderate'
        })

    # 4. Create comprehensive evidence report
    evidence_summary = {
        'evidence_points': evidence_points,
        'model_results': model_results,
        'evidence_plots': evidence_plots,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Calculate overall assessment
    strong_points = sum(1 for point in evidence_points if point['strength'] == 'Strong')
    moderate_points = sum(1 for point in evidence_points if point['strength'] == 'Moderate')

    if strong_points >= 2:
        evidence_summary['overall_assessment'] = 'Strong evidence of non-overfitting'
    elif strong_points + moderate_points >= 3:
        evidence_summary['overall_assessment'] = 'Moderate evidence of non-overfitting'
    else:
        evidence_summary['overall_assessment'] = 'Limited evidence of non-overfitting'

    # Save evidence summary
    with open(os.path.join(output_dir, 'evidence_summary.json'), 'w') as f:
        json.dump(evidence_summary, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else str(x))

    print(f"Non-overfitting evidence saved to {output_dir}")
    print(f"Overall assessment: {evidence_summary['overall_assessment']}")

    return evidence_summary

if __name__ == '__main__':
    results, diagnosis, experiment_dir = enhanced_main()