# Social Media Addiction Assessment - ML Training Report

## Training Summary
- **Training Date**: 2025-10-01 20:23:26
- **Dataset**: user_response_data.csv
- **Total Samples**: 5000
- **Features**: 22 (Q1-Q22, excluding ResultScore and ResultBand)

## Dataset Analysis

### Dataset Characteristics
- **Shape**: (5000, 24)
- **Features**: 22 questionnaire responses (Q1-Q22)
- **Target Variable**: ResultBand (4 risk levels)
- **Missing Values**: 0 total missing values

### Target Distribution
- **High risk / addictive pattern (consider referral)**: 1250 samples (25.0%)
- **Problematic use likely (structured assessment)**: 1250 samples (25.0%)
- **At-risk (brief advice/monitor)**: 1250 samples (25.0%)
- **Low risk**: 1250 samples (25.0%)


## Models Tested

The following 5 machine learning algorithms were trained and evaluated:

1. **Random Forest Classifier**
   - Accuracy: 0.8660
   - F1-Score: 0.8639
   - Cross-Validation: 0.8363 ± 0.0124

2. **Gradient Boosting Classifier**
   - Accuracy: 0.8500
   - F1-Score: 0.8482
   - Cross-Validation: 0.8408 ± 0.0043

3. **Support Vector Machine**
   - Accuracy: 0.9570
   - F1-Score: 0.9568
   - Cross-Validation: 0.9572 ± 0.0115

4. **Logistic Regression**
   - Accuracy: 0.9850
   - F1-Score: 0.9850
   - Cross-Validation: 0.9858 ± 0.0023

5. **K-Nearest Neighbors**
   - Accuracy: 0.7290
   - F1-Score: 0.7215
   - Cross-Validation: 0.7215 ± 0.0099

## Best Performing Model

### Logistic Regression
- **Accuracy**: 0.9850
- **Precision**: 0.9857
- **Recall**: 0.9850
- **F1-Score**: 0.9850
- **Cross-Validation Accuracy**: 0.9858 ± 0.0023

### Why This Model Was Selected
The Logistic Regression was selected as the best model based on F1-Score, which provides a balanced measure of precision and recall. This is particularly important for social media addiction assessment where both false positives and false negatives can have significant implications.

## Confusion Matrix Analysis

The confusion matrix for the best model shows the following performance:

```
Confusion Matrix for Logistic Regression:
```
                    At-risk (brief High risk / addLow risk       Problematic use
At-risk (brief 249            0              1              0              
High risk / add0              250            0              0              
Low risk       0              0              250            0              
Problematic use14             0              0              236            
```

### Detailed Performance Analysis

- **True Positives**: 985 correct predictions
- **Total Predictions**: 1000
- **Overall Accuracy**: 98.50%

#### Per-Class Performance:
- **At-risk (brief advice/monitor)**:
  - Precision: 0.947
  - Recall: 0.996
  - F1-Score: 0.971
- **High risk / addictive pattern (consider referral)**:
  - Precision: 1.000
  - Recall: 1.000
  - F1-Score: 1.000
- **Low risk**:
  - Precision: 0.996
  - Recall: 1.000
  - F1-Score: 0.998
- **Problematic use likely (structured assessment)**:
  - Precision: 1.000
  - Recall: 0.944
  - F1-Score: 0.971


## Model Deployment

The best model has been saved to the `MODELS/` folder and can be used by the Flask application (`app.py`) for real-time predictions.

### Model Files:
- **Model Path**: `MODELS/best_model_logistic_regression/`
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

The Logistic Regression model achieved the best performance with an F1-Score of 0.9850, making it suitable for deployment in the social media addiction assessment system. The model demonstrates good generalization capabilities as evidenced by the cross-validation results.

---
*Report generated on 2025-10-01 20:23:26*
