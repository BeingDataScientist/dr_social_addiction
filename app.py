from flask import Flask, render_template, request, jsonify
import csv
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for ML model
ml_model = None
label_encoder = None
scaler = None
model_loaded = False

def load_ml_model():
    """Load the trained ML model and preprocessing objects"""
    global ml_model, label_encoder, scaler, model_loaded
    
    try:
        # Try to load model performance info
        if os.path.exists('model_performance.json'):
            with open('model_performance.json', 'r') as f:
                model_info = json.load(f)
                best_model_name = model_info['best_model_name']
                print(f"Loading best model: {best_model_name}")
        else:
            print("No model performance file found. Using default model name.")
            best_model_name = "random_forest"  # Default fallback
        
        # Try to load the model from MLflow
        model_path = f"MODELS/best_model_{best_model_name.lower().replace(' ', '_')}"
        if os.path.exists(model_path):
            ml_model = mlflow.sklearn.load_model(model_path)
            print(f"ML model loaded successfully from {model_path}")
        else:
            print(f"Model path {model_path} not found. ML predictions disabled.")
            return False
        
        # Load label encoder and scaler from training data
        # We'll recreate them from the original data
        if os.path.exists('user_response_data.csv'):
            df = pd.read_csv('user_response_data.csv')
            label_encoder = LabelEncoder()
            label_encoder.fit(df['ResultBand'])
            
            # Create scaler (we'll use the same scaling as in training)
            scaler = StandardScaler()
            feature_columns = [f'Q{i}' for i in range(1, 23)]
            X = df[feature_columns]
            scaler.fit(X)
            
            model_loaded = True
            print("Preprocessing objects loaded successfully")
            return True
        else:
            print("Training data not found. Cannot load preprocessing objects.")
            return False
            
    except Exception as e:
        print(f"Error loading ML model: {str(e)}")
        model_loaded = False
        return False

def predict_with_ml_model(answers):
    """Make prediction using the loaded ML model"""
    global ml_model, label_encoder, scaler, model_loaded
    
    if not model_loaded or ml_model is None:
        return None, "ML model not available"
    
    try:
        # Convert answers to numpy array and reshape
        X = np.array(answers).reshape(1, -1)
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction_encoded = ml_model.predict(X_scaled)[0]
        prediction_proba = ml_model.predict_proba(X_scaled)[0]
        
        # Decode prediction
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence score
        confidence = max(prediction_proba) * 100
        
        return prediction, confidence
        
    except Exception as e:
        print(f"Error making ML prediction: {str(e)}")
        return None, f"Prediction error: {str(e)}"

def calculate_score_and_band(answers):
    """
    Calculate the total score and determine the risk band based on the answers.
    Based on the CSV data analysis, the scoring appears to be:
    - Low risk: 0-10
    - At-risk: 11-20  
    - Problematic use: 21-30
    - High risk: 31+
    """
    total_score = sum(answers)
    
    if total_score <= 10:
        band = "Low risk"
    elif total_score <= 20:
        band = "At-risk (brief advice/monitor)"
    elif total_score <= 30:
        band = "Problematic use likely (structured assessment)"
    else:
        band = "High risk / addictive pattern (consider referral)"
    
    return total_score, band

def save_to_csv(data):
    """Save the form data to CSV file matching the original schema"""
    # Use a different filename to avoid conflicts
    csv_file = 'new_user_responses.csv'
    
    # Check if it's a directory and remove it
    if os.path.isdir(csv_file):
        import shutil
        shutil.rmtree(csv_file)
        print(f"Removed directory {csv_file}, will create as file")
    
    file_exists = os.path.isfile(csv_file)
    
    # Try to write with retry mechanism for file locking issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(csv_file, 'a', newline='', encoding='utf-8') as file:
                fieldnames = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 
                             'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 
                             'Q20', 'Q21', 'Q22', 'ResultScore', 'ResultBand', 'Timestamp']
                
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Add timestamp to data
                data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                writer.writerow(data)
                print(f"Data saved to {csv_file}")
                break
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                import time
                time.sleep(0.1)  # Wait 100ms before retry
            else:
                print(f"Failed to save after {max_retries} attempts: {e}")
                # Create a backup filename if main file is locked
                import time
                backup_file = f'user_responses_backup_{int(time.time())}.csv'
                try:
                    with open(backup_file, 'w', newline='', encoding='utf-8') as backup:
                        fieldnames = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 
                                     'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 
                                     'Q20', 'Q21', 'Q22', 'ResultScore', 'ResultBand', 'Timestamp']
                        writer = csv.DictWriter(backup, fieldnames=fieldnames)
                        writer.writeheader()
                        data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow(data)
                        print(f"Data saved to backup file: {backup_file}")
                except Exception as backup_error:
                    print(f"Failed to save to backup file: {backup_error}")
                    raise

@app.route('/')
def index():
    """Serve the questionnaire form"""
    return render_template('questionnaire_form.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    """Handle form submission and calculate results"""
    try:
        # Extract answers from form data
        answers = []
        for i in range(1, 23):  # Q1 to Q22
            question_key = f'Q{i}'
            answer = request.form.get(question_key)
            if answer is None:
                return jsonify({'error': f'Missing answer for {question_key}'}), 400
            answers.append(int(answer))
        
        # Calculate traditional score and band
        total_score, traditional_band = calculate_score_and_band(answers)
        
        # Get ML model prediction
        ml_prediction, ml_confidence = predict_with_ml_model(answers)
        
        # Use ML prediction if available, otherwise use traditional method
        if ml_prediction is not None:
            final_band = ml_prediction
            prediction_method = "ML Model"
            confidence = ml_confidence
        else:
            final_band = traditional_band
            prediction_method = "Traditional Scoring"
            confidence = None
        
        # Prepare data for CSV
        csv_data = {}
        for i, answer in enumerate(answers, 1):
            csv_data[f'Q{i}'] = answer
        csv_data['ResultScore'] = total_score
        csv_data['ResultBand'] = final_band
        
        # Save to CSV
        save_to_csv(csv_data)
        
        # Prepare response data
        response_data = {
            'answers': answers,
            'total_score': total_score,
            'traditional_band': traditional_band,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'final_band': final_band,
            'prediction_method': prediction_method,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print values to console as requested
        print("\n" + "="*60)
        print("QUESTIONNAIRE SUBMISSION RECEIVED")
        print("="*60)
        print(f"Timestamp: {response_data['timestamp']}")
        print(f"Total Score: {total_score}")
        print(f"Traditional Band: {traditional_band}")
        if ml_prediction is not None:
            print(f"ML Prediction: {ml_prediction} (Confidence: {ml_confidence:.1f}%)")
        print(f"Final Band: {final_band} (Method: {prediction_method})")
        print("\nIndividual Answers:")
        for i, answer in enumerate(answers, 1):
            print(f"Q{i}: {answer}")
        print("="*60)
        
        return jsonify({
            'success': True,
            'message': 'Assessment completed successfully',
            'data': response_data
        })
        
    except Exception as e:
        print(f"Error processing form: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your assessment'}), 500

@app.route('/results')
def show_results():
    """Display comprehensive results with analysis and charts"""
    try:
        # Get user's assessment results from session
        user_results = None
        
        # Analyze CSV data for charts and statistics
        analysis_data = analyze_csv_data()
        
        return render_template('results.html', 
                             user_results=user_results,
                             analysis=analysis_data)
        
    except Exception as e:
        return f"Error displaying results: {str(e)}"

def analyze_csv_data():
    """Analyze CSV data and return statistics for charts"""
    try:
        # Read original training data
        if os.path.exists('user_response_data lite.csv'):
            df = pd.read_csv('user_response_data lite.csv')
        else:
            return None
        
        # Basic statistics
        total_assessments = len(df)
        
        # Risk band distribution
        risk_distribution = df['ResultBand'].value_counts().to_dict()
        
        # Score statistics
        score_stats = {
            'mean': df['ResultScore'].mean(),
            'median': df['ResultScore'].median(),
            'std': df['ResultScore'].std(),
            'min': df['ResultScore'].min(),
            'max': df['ResultScore'].max()
        }
        
        # Question-wise statistics
        question_stats = {}
        for i in range(1, 23):
            col = f'Q{i}'
            question_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'distribution': df[col].value_counts().to_dict()
            }
        
        # Score ranges for each risk band
        score_ranges = {}
        for band in df['ResultBand'].unique():
            band_scores = df[df['ResultBand'] == band]['ResultScore']
            score_ranges[band] = {
                'min': band_scores.min(),
                'max': band_scores.max(),
                'mean': band_scores.mean()
            }
        
        # Age-like analysis (if we had age data, we'd use it)
        # For now, we'll create some synthetic patterns
        
        return {
            'total_assessments': total_assessments,
            'risk_distribution': risk_distribution,
            'score_stats': score_stats,
            'question_stats': question_stats,
            'score_ranges': score_ranges
        }
        
    except Exception as e:
        print(f"Error analyzing CSV data: {str(e)}")
        return None

@app.route('/api/results')
def api_results():
    """API endpoint to get all results as JSON"""
    try:
        # Try to read from the new file first, then fallback to old file
        csv_file = 'new_user_responses.csv'
        
        # If new file doesn't exist, try the old file
        if not os.path.isfile(csv_file):
            csv_file = 'user_responses.csv'
            
            # Check if it's a directory and remove it
            if os.path.isdir(csv_file):
                import shutil
                shutil.rmtree(csv_file)
                print(f"Removed directory {csv_file}, will create as file")
        
        if not os.path.isfile(csv_file):
            return jsonify([])
        
        results = []
        
        # Try to read with retry mechanism for file locking issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(csv_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        results.append(row)
                break
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed to read {csv_file}: {e}. Retrying...")
                    import time
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    print(f"Failed to read {csv_file} after {max_retries} attempts: {e}")
                    return jsonify({'error': f'Unable to read data file: {str(e)}'})
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis')
def api_analysis():
    """API endpoint to get analysis data for charts"""
    try:
        analysis_data = analyze_csv_data()
        if analysis_data:
            return jsonify(analysis_data)
        else:
            return jsonify({'error': 'No analysis data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Move the HTML file to templates directory
    if os.path.exists('questionnaire_form.html'):
        import shutil
        shutil.move('questionnaire_form.html', 'templates/questionnaire_form.html')
    
    # Load ML model on startup
    print("Starting Flask application...")
    print("Loading ML model...")
    model_loaded_successfully = load_ml_model()
    
    if model_loaded_successfully:
        print("ML model loaded successfully! Predictions will use ML model.")
    else:
        print("ML model not loaded. Predictions will use traditional scoring method.")
    
    print("Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
