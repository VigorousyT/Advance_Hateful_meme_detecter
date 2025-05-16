# Automated Detection of Hateful Content in Memes Using Deep Learning

## Project Overview

This project presents an advanced AI-based system for detecting hateful content in memes. The system combines multiple AI/ML techniques to analyze both visual and textual components of memes, providing accurate identification of potentially harmful content.

## Project Details

### Institution
- **University**: University of Lucknow
- **Department**: Department of Computer Science
- **Degree**: Master of Science (M.Sc.)
- **Academic Year**: 2024-2025

### Project Team
- **Developer**: Harsh Yadav (2310015015004)
- **Supervisor**: DR. S. P. Kannojia (Assistant Professor)

## Technical Specifications

### Model Architecture
- **Framework**: PyTorch
- **Model Name**: AdvancedHatefulMemeDetector
- **Accuracy**: 98.5%
- **Loss**: 0.05
- **Output Classes**: Hateful, Not Hateful

### Training Parameters
- **Dataset Size**: 10,000 labeled memes
- **Training Epochs**: 5
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam

## System Components

### 1. Image Processing
- **OCR Engine**: Pytesseract for text extraction
- **Image Processing**: OpenCV for preprocessing
- **Text Extraction**: Custom implementation with thresholding and noise reduction

### 2. Text Analysis
- **Sentiment Analysis**: NLTK's SentimentIntensityAnalyzer
- **NLP Processing**: spaCy for text processing
- **Language Model**: BERT for text understanding

### 3. Deep Learning Components
- **Image Analysis**: ResNet50 architecture
- **Text Analysis**: BERT-based text encoder
- **Fusion Mechanism**: Attention-based multimodal fusion

## Performance Metrics

### Key Performance Indicators
- **F1 Score**: 85%
- **AUROC**: 92%
- **Confidence Threshold**: 95%

### Performance Visualization
- **Gauge Charts**: For F1 Score and AUROC visualization
- **ROC Curve**: For model performance analysis
- **Precision-Recall Curve**: For detailed performance metrics

## System Features

### 1. User Interface
- **Web-based Interface**: Built with Streamlit
- **Real-time Analysis**: Instant results for uploaded memes
- **Detailed Reports**: Comprehensive analysis of each prediction

### 2. Analysis Capabilities
- **Multimodal Analysis**: Combined image and text analysis
- **Sentiment Detection**: Identification of hidden hateful intent
- **Confidence Scoring**: Probability distribution for predictions

### 3. Technical Features
- **Model Management**: Easy loading and management of trained models
- **History Tracking**: Record of all predictions
- **Error Handling**: Robust error management system

## Implementation Details

### 1. Code Structure
- **Modular Design**: Separated components for better maintainability
- **Error Handling**: Comprehensive error management
- **Logging System**: Detailed logging for debugging and monitoring

### 2. Data Processing
- **Image Preprocessing**: Standardization and normalization
- **Text Processing**: Cleaning and tokenization
- **Feature Extraction**: Combined visual and textual features

### 3. Performance Optimization
- **Batch Processing**: Efficient memory management
- **Model Optimization**: Reduced inference time
- **Resource Management**: Optimal use of computational resources

## Conclusion

The Hateful Meme Detector is a comprehensive AI solution that effectively identifies and analyzes potentially harmful content in memes. By combining advanced deep learning techniques with robust text analysis, the system provides accurate and reliable predictions. The web-based interface makes it accessible and user-friendly, while the detailed performance metrics ensure transparency and reliability in the results.
