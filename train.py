import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SocialMediaAddictionMLTrainer:
    def __init__(self, csv_file='user_response_data.csv'):
        self.csv_file = csv_file
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.training_info = {}
        
        # Create directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories for organizing outputs"""
        directories = ['TRAIN_Analysis', 'MODELS']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
    def load_and_prepare_data(self):
        """Load data and prepare features and target"""
        print("Loading and preparing data...")
        
        # Load the CSV file
        self.df = pd.read_csv(self.csv_file)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Store training information
        self.training_info['dataset_shape'] = self.df.shape
        self.training_info['columns'] = list(self.df.columns)
        self.training_info['missing_values'] = self.df.isnull().sum().to_dict()
        self.training_info['target_distribution'] = self.df['ResultBand'].value_counts().to_dict()
        
        # Display basic info about the dataset
        print("\nDataset Info:")
        print(self.df.info())
        
        # Check for missing values
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        # Display target distribution
        print(f"\nTarget distribution:\n{self.df['ResultBand'].value_counts()}")
        
        # Prepare features (Q1-Q22) and target (ResultBand)
        # Note: We exclude ResultScore as it's calculated from Q1-Q22 (data leakage)
        feature_columns = [f'Q{i}' for i in range(1, 23)]
        self.X = self.df[feature_columns]
        self.y = self.df['ResultBand']
        
        print(f"\nFeature columns: {feature_columns}")
        print(f"Target column: ResultBand")
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        # Encode target labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        print(f"Encoded target classes: {self.label_encoder.classes_}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
    def define_models(self):
        """Define the top 5 ML algorithms"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean'
            )
        }
        
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("TRAINING AND EVALUATING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"{name}_SocialMediaAddiction"):
                # Choose scaled or unscaled data based on model
                if name in ['Support Vector Machine', 'Logistic Regression', 'K-Nearest Neighbors']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test
                
                # Train the model
                model.fit(X_train_use, self.y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_use)
                y_pred_proba = model.predict_proba(X_test_use) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Log parameters
                mlflow.log_param("model_name", name)
                mlflow.log_param("test_size", 0.2)
                mlflow.log_param("random_state", 42)
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("cv_accuracy_mean", cv_mean)
                mlflow.log_metric("cv_accuracy_std", cv_std)
                
                # Log model
                mlflow.sklearn.log_model(model, f"{name.lower().replace(' ', '_')}_model")
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Print results
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
                
                # Classification report
                print(f"\nClassification Report for {name}:")
                print(classification_report(self.y_test, y_pred, 
                                         target_names=self.label_encoder.classes_))
    
    def find_best_model(self):
        """Find and save the best performing model"""
        print("\n" + "="*60)
        print("FINDING BEST MODEL")
        print("="*60)
        
        # Find best model based on F1-score
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_model = self.results[best_model_name]['model']
        best_accuracy = self.results[best_model_name]['accuracy']
        best_f1 = self.results[best_model_name]['f1_score']
        
        print(f"Best Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Best F1-Score: {best_f1:.4f}")
        
        # Save the best model in MODELS folder
        model_path = f"MODELS/best_model_{best_model_name.lower().replace(' ', '_')}"
        mlflow.sklearn.save_model(
            best_model, 
            model_path
        )
        
        # Save model info
        model_info = {
            'best_model_name': best_model_name,
            'best_accuracy': best_accuracy,
            'best_f1_score': best_f1,
            'all_results': {name: {
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'cv_mean': results['cv_mean']
            } for name, results in self.results.items()}
        }
        
        import json
        with open('model_performance.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return best_model_name, best_model
    
    def create_visualizations(self):
        """Create performance visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Model comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[0, 1].bar(model_names, f1_scores, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cross-validation comparison
        axes[1, 0].bar(model_names, cv_means, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Cross-Validation Accuracy Comparison')
        axes[1, 0].set_ylabel('CV Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Combined metrics
        x = np.arange(len(model_names))
        width = 0.25
        axes[1, 1].bar(x - width, accuracies, width, label='Accuracy', alpha=0.7)
        axes[1, 1].bar(x, f1_scores, width, label='F1-Score', alpha=0.7)
        axes[1, 1].bar(x + width, cv_means, width, label='CV Accuracy', alpha=0.7)
        axes[1, 1].set_title('Combined Metrics Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('TRAIN_Analysis/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion matrix for best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_predictions = self.results[best_model_name]['predictions']
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('TRAIN_Analysis/confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance for tree-based models
        tree_models = ['Random Forest', 'Gradient Boosting']
        for model_name in tree_models:
            if model_name in self.results:
                model = self.results[model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    plt.figure(figsize=(12, 8))
                    feature_names = [f'Q{i}' for i in range(1, 23)]
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    plt.title(f'Feature Importance - {model_name}')
                    plt.bar(range(len(importances)), importances[indices])
                    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
                    plt.xlabel('Features')
                    plt.ylabel('Importance')
                    plt.tight_layout()
                    plt.savefig(f'TRAIN_Analysis/feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                              dpi=300, bbox_inches='tight')
                    plt.show()
    
    def print_detailed_results(self):
        """Print detailed results for all models"""
        print("\n" + "="*80)
        print("DETAILED RESULTS SUMMARY")
        print("="*80)
        
        # Create results DataFrame
        results_data = []
        for name, results in self.results.items():
            results_data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'CV Accuracy': f"{results['cv_mean']:.4f}",
                'CV Std': f"{results['cv_std']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        # Find and highlight best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        print(f"\nBEST MODEL: {best_model_name}")
        print(f"   Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        
        # Sample predictions
        print(f"\nSAMPLE PREDICTIONS ({best_model_name}):")
        sample_indices = np.random.choice(len(self.y_test), 10, replace=False)
        for i, idx in enumerate(sample_indices):
            actual = self.label_encoder.classes_[self.y_test[idx]]
            predicted = self.label_encoder.classes_[self.results[best_model_name]['predictions'][idx]]
            print(f"   Sample {i+1}: Actual={actual}, Predicted={predicted}")
    
    def create_markdown_report(self):
        """Create a comprehensive markdown report of the training process"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_results = self.results[best_model_name]
        
        # Get confusion matrix data
        cm = confusion_matrix(self.y_test, best_results['predictions'])
        
        report_content = f"""# Social Media Addiction Assessment - ML Training Report

## Training Summary
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset**: {self.csv_file}
- **Total Samples**: {self.training_info['dataset_shape'][0]}
- **Features**: {self.training_info['dataset_shape'][1] - 2} (Q1-Q22, excluding ResultScore and ResultBand)

## Dataset Analysis

### Dataset Characteristics
- **Shape**: {self.training_info['dataset_shape']}
- **Features**: 22 questionnaire responses (Q1-Q22)
- **Target Variable**: ResultBand (4 risk levels)
- **Missing Values**: {sum(self.training_info['missing_values'].values())} total missing values

### Target Distribution
"""
        
        for class_name, count in self.training_info['target_distribution'].items():
            percentage = (count / self.training_info['dataset_shape'][0]) * 100
            report_content += f"- **{class_name}**: {count} samples ({percentage:.1f}%)\n"
        
        report_content += f"""

## Models Tested

The following 5 machine learning algorithms were trained and evaluated:

1. **Random Forest Classifier**
   - Accuracy: {self.results['Random Forest']['accuracy']:.4f}
   - F1-Score: {self.results['Random Forest']['f1_score']:.4f}
   - Cross-Validation: {self.results['Random Forest']['cv_mean']:.4f} ± {self.results['Random Forest']['cv_std']:.4f}

2. **Gradient Boosting Classifier**
   - Accuracy: {self.results['Gradient Boosting']['accuracy']:.4f}
   - F1-Score: {self.results['Gradient Boosting']['f1_score']:.4f}
   - Cross-Validation: {self.results['Gradient Boosting']['cv_mean']:.4f} ± {self.results['Gradient Boosting']['cv_std']:.4f}

3. **Support Vector Machine**
   - Accuracy: {self.results['Support Vector Machine']['accuracy']:.4f}
   - F1-Score: {self.results['Support Vector Machine']['f1_score']:.4f}
   - Cross-Validation: {self.results['Support Vector Machine']['cv_mean']:.4f} ± {self.results['Support Vector Machine']['cv_std']:.4f}

4. **Logistic Regression**
   - Accuracy: {self.results['Logistic Regression']['accuracy']:.4f}
   - F1-Score: {self.results['Logistic Regression']['f1_score']:.4f}
   - Cross-Validation: {self.results['Logistic Regression']['cv_mean']:.4f} ± {self.results['Logistic Regression']['cv_std']:.4f}

5. **K-Nearest Neighbors**
   - Accuracy: {self.results['K-Nearest Neighbors']['accuracy']:.4f}
   - F1-Score: {self.results['K-Nearest Neighbors']['f1_score']:.4f}
   - Cross-Validation: {self.results['K-Nearest Neighbors']['cv_mean']:.4f} ± {self.results['K-Nearest Neighbors']['cv_std']:.4f}

## Best Performing Model

### {best_model_name}
- **Accuracy**: {best_results['accuracy']:.4f}
- **Precision**: {best_results['precision']:.4f}
- **Recall**: {best_results['recall']:.4f}
- **F1-Score**: {best_results['f1_score']:.4f}
- **Cross-Validation Accuracy**: {best_results['cv_mean']:.4f} ± {best_results['cv_std']:.4f}

### Why This Model Was Selected
The {best_model_name} was selected as the best model based on F1-Score, which provides a balanced measure of precision and recall. This is particularly important for social media addiction assessment where both false positives and false negatives can have significant implications.

## Confusion Matrix Analysis

The confusion matrix for the best model shows the following performance:

```
Confusion Matrix for {best_model_name}:
"""
        
        # Add confusion matrix in text format
        class_names = self.label_encoder.classes_
        report_content += "```\n"
        report_content += " " * 20
        for name in class_names:
            report_content += f"{name[:15]:<15}"
        report_content += "\n"
        
        for i, actual in enumerate(class_names):
            report_content += f"{actual[:15]:<15}"
            for j in range(len(class_names)):
                report_content += f"{cm[i][j]:<15}"
            report_content += "\n"
        
        report_content += "```\n\n"
        
        # Add detailed analysis
        report_content += f"""### Detailed Performance Analysis

- **True Positives**: {np.diag(cm).sum()} correct predictions
- **Total Predictions**: {cm.sum()}
- **Overall Accuracy**: {(np.diag(cm).sum() / cm.sum() * 100):.2f}%

#### Per-Class Performance:
"""
        
        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report_content += f"- **{class_name}**:\n"
            report_content += f"  - Precision: {precision:.3f}\n"
            report_content += f"  - Recall: {recall:.3f}\n"
            report_content += f"  - F1-Score: {f1:.3f}\n"
        
        report_content += f"""

## Model Deployment

The best model has been saved to the `MODELS/` folder and can be used by the Flask application (`app.py`) for real-time predictions.

### Model Files:
- **Model Path**: `MODELS/best_model_{best_model_name.lower().replace(' ', '_')}/`
- **Performance Data**: `model_performance.json`

## Visualizations Generated

The following visualizations have been saved in the `TRAIN_Analysis/` folder:

1. **model_performance_comparison.png** - Comparison of all models across different metrics
2. **confusion_matrix_best_model.png** - Confusion matrix for the best performing model
3. **feature_importance_*.png** - Feature importance plots for tree-based models (if applicable)

## Training Configuration

- **Test Size**: 20% of the dataset
- **Random State**: 42 (for reproducibility)
- **Cross-Validation**: 5-fold
- **Feature Scaling**: Applied for SVM, Logistic Regression, and KNN
- **Data Leakage Prevention**: ResultScore excluded from features

## Conclusion

The {best_model_name} model achieved the best performance with an F1-Score of {best_results['f1_score']:.4f}, making it suitable for deployment in the social media addiction assessment system. The model demonstrates good generalization capabilities as evidenced by the cross-validation results.

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save the report
        with open('TRAIN_Analysis/training_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("Training report saved to: TRAIN_Analysis/training_report.md")
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("STARTING SOCIAL MEDIA ADDICTION ML TRAINING")
        print("="*80)
        
        # Set MLflow experiment
        mlflow.set_experiment("Social_Media_Addiction_Classification")
        
        try:
            # Load and prepare data
            self.load_and_prepare_data()
            
            # Define models
            self.define_models()
            
            # Train and evaluate models
            self.train_and_evaluate_models()
            
            # Find best model
            best_model_name, best_model = self.find_best_model()
            
            # Create visualizations
            self.create_visualizations()
            
            # Create markdown report
            self.create_markdown_report()
            
            # Print detailed results
            self.print_detailed_results()
            
            print("\nTRAINING COMPLETED SUCCESSFULLY!")
            print(f"Best model saved to: MODELS/best_model_{best_model_name.lower().replace(' ', '_')}")
            print("Performance metrics saved to: model_performance.json")
            print("Visualizations saved to: TRAIN_Analysis/ folder")
            print("Training report saved to: TRAIN_Analysis/training_report.md")
            print("MLflow experiments logged successfully")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize trainer
    trainer = SocialMediaAddictionMLTrainer()
    
    # Run complete training pipeline
    trainer.run_complete_training()
