# Social Media Addiction Assessment System

A comprehensive machine learning-powered web application for assessing social media addiction risk levels through questionnaire responses.

## 🎯 Overview

This system uses advanced machine learning algorithms to predict social media addiction risk levels based on 22 questionnaire responses. The application provides real-time predictions with 98.5% accuracy using a trained Logistic Regression model.

## ✨ Features

- **ML-Powered Assessment**: 5 different algorithms tested, best model selected
- **Real-time Predictions**: Instant risk level assessment
- **Comprehensive Analysis**: Detailed visualizations and insights
- **Bias Mitigation**: SMOTE + Tomek Links for balanced training
- **Interactive Web Interface**: User-friendly questionnaire and results
- **Production Ready**: Optimized for deployment

## 🏆 Model Performance

- **Best Model**: Logistic Regression
- **Accuracy**: 98.5%
- **F1-Score**: 98.5%
- **Cross-Validation**: 98.6% ± 0.2%
- **Dataset**: 5,000 samples, perfectly balanced

## 📊 Risk Levels

1. **Low Risk**: Healthy social media usage patterns
2. **At-risk (brief advice/monitor)**: Developing concerning habits
3. **Problematic use likely (structured assessment)**: Significant issues
4. **High risk / addictive pattern (consider referral)**: Severe addiction patterns

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/dr_social_addiction.git
cd dr_social_addiction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the ML model**
```bash
python train.py
```

4. **Run the application**
```bash
python app.py
```

5. **Access the application**
Open your browser and go to: `http://localhost:5000`

## 📁 Project Structure

```
dr_social_addiction/
├── app.py                          # Main Flask application
├── train.py                        # ML training script
├── requirements.txt                # Python dependencies
├── user_response_data.csv          # Training dataset (5,000 samples)
├── user_responses.csv              # User submissions data
├── model_performance.json          # Model metadata
├── MODELS/                         # Trained ML models
│   └── best_model_logistic_regression/
├── TRAIN_Analysis/                 # Training analysis and reports
│   ├── training_report.md          # Comprehensive training report
│   ├── model_performance_comparison.png
│   ├── confusion_matrix_best_model.png
│   └── feature_importance_*.png
├── templates/                      # HTML templates
│   ├── questionnaire_form.html     # Assessment form
│   └── results.html                # Results page with visualizations
├── mlruns/                         # MLflow experiment tracking
└── README.md                       # This file
```

## 🧠 Machine Learning Pipeline

### Models Tested
1. **Random Forest Classifier** - 86.6% accuracy
2. **Gradient Boosting Classifier** - 85.0% accuracy
3. **Support Vector Machine** - 95.7% accuracy
4. **Logistic Regression** - 98.5% accuracy ⭐
5. **K-Nearest Neighbors** - 72.9% accuracy

### Training Process
- **Dataset**: 5,000 samples, 4 balanced classes
- **Features**: 22 questionnaire responses (Q1-Q22)
- **Preprocessing**: Label encoding, feature scaling
- **Validation**: 5-fold cross-validation
- **Bias Mitigation**: SMOTE + Tomek Links
- **Selection Criteria**: F1-Score optimization

## 🌐 Web Application

### Endpoints
- `GET /` - Questionnaire form
- `POST /submit` - Submit assessment
- `GET /results` - Results page with visualizations
- `GET /api/analysis` - Analysis data for charts
- `GET /api/results` - All assessment results
- `GET /health` - Health check endpoint

### Features
- **Responsive Design**: Works on desktop and mobile
- **Real-time Validation**: Form validation and error handling
- **Interactive Charts**: Chart.js visualizations
- **Session Management**: Secure data handling
- **API Endpoints**: RESTful API for data access

## 📈 Visualizations

The system generates comprehensive visualizations:

1. **Model Performance Comparison** - All models compared across metrics
2. **Confusion Matrix** - Best model performance breakdown
3. **Feature Importance** - Most influential questionnaire items
4. **Risk Distribution** - Population risk level distribution
5. **Response Patterns** - User response analysis
6. **Risk Analysis** - Individual risk assessment

## 🔧 Configuration

### Environment Variables
```bash
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key
```

### Model Configuration
- Models are automatically loaded from `MODELS/` folder
- Preprocessing objects recreated from training data
- Fallback to traditional scoring if ML model unavailable

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed AWS deployment instructions.

**Quick AWS Deployment:**
```bash
# Elastic Beanstalk
pip install awsebcli
eb init
eb create production
eb deploy
```

## 📊 API Usage

### Submit Assessment
```bash
curl -X POST http://localhost:5000/submit \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "Q1=2&Q2=1&Q3=3&..."
```

### Get Analysis Data
```bash
curl http://localhost:5000/api/analysis
```

### Health Check
```bash
curl http://localhost:5000/health
```

## 🧪 Testing

### Manual Testing
1. Complete the questionnaire form
2. Submit and verify results
3. Check visualizations load correctly
4. Test API endpoints

### Model Testing
```bash
# Run training to test model performance
python train.py

# Check model files are created
ls MODELS/
ls TRAIN_Analysis/
```

## 📚 Dependencies

- **Flask** - Web framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **mlflow** - Model tracking
- **matplotlib** - Plotting
- **seaborn** - Statistical visualization
- **imbalanced-learn** - Bias mitigation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Dr. Ashwini** - Research Supervisor
- **Development Team** - Implementation

## 🙏 Acknowledgments

- Research participants who provided questionnaire data
- Open source ML libraries and frameworks
- AWS for cloud deployment infrastructure

## 📞 Support

For questions or support:
- Create an issue in this repository
- Contact the development team
- Check the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment help

## 📈 Future Enhancements

- [ ] Real-time model retraining
- [ ] Advanced bias detection
- [ ] Multi-language support
- [ ] Mobile app development
- [ ] Integration with health systems
- [ ] Advanced analytics dashboard

---

**Last Updated**: October 2025
**Version**: 1.0.0
