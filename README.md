# Automated Detection of Hateful Content in Memes Using Deep Learning App
# Created by
# Harsh Yadav
# Student of Master of Science in Computer Science at the University of Lucknow.
# project under the guidance of Dr. S.P. Kannojia

This application uses an AI model to detect hateful content in memes. It combines image and text analysis to identify potentially harmful content.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Setup Instructions

1. Clone the repository (if using Git):
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv_new
```

3. Activate the virtual environment:
- On Windows:
```bash
venv_new\Scripts\activate
```
- On Unix/MacOS:
```bash
source venv_new/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Download the Hateful Memes dataset from Facebook AI:
- Visit: https://www.kaggle.com/datasets/harshlu/meme-data-set
- Follow the instructions to download the dataset
- Place the dataset in the `data/` directory

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the application:
- Local URL: http://localhost:8501
- Network URL: http://<your-ip-address>:8501

## Known Warnings and Solutions

### 1. Hugging Face Hub Warning
```
FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0
```
- This warning appears when downloading models from Hugging Face Hub
- It's safe to ignore as downloads will still work correctly
- In future versions, use `force_download=True` if you need to force a new download

### 2. Plotly/Pandas Warning
```
FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple
```
- This is a pandas/plotly compatibility warning
- Does not affect functionality
- Will be resolved in future versions of the libraries

## Project Structure
- `app.py`: Main Streamlit application file
- `model.py`: Model training and inference code
- `utils.py`: Utility functions
- `data/`: Directory for storing the dataset
- `models/`: Directory for storing trained models
- `checkpoints/`: Directory for model checkpoints

## Features
- Upload and analyze memes
- Real-time hateful content detection
- Confidence score for predictions
- Batch processing capability
- Interactive web interface
- Model performance visualization

## Model Information

### Training Process
- Training Dataset: 8,500 images from train.jsonl
- Validation Dataset: 500 images from dev.jsonl
- Batch Size: 32 images per batch
- Training Accuracy: 52.59%
- Validation Accuracy: 51.40%

### Training Flow
1. **Training Phase**:
   - 265 batches per epoch (8,480 images)
   - Training time per batch: ~23.20 seconds
   - Full epoch training time: ~1 hour 42 minutes

2. **Validation Phase**:
   - 16 batches (512 images)
   - Validation time per batch: ~6.29 seconds
   - Full validation time: ~1 minute 40 seconds

### Checkpoint Management
- Checkpoints are saved in the `checkpoints` directory
- Best model is saved as `best_model.pt`
- Contains:
  - Model state
  - Optimizer state
  - Epoch number
  - Validation accuracy

## Usage Guide

### 1. Training the Model
```bash
python meme_classifier.py
```

### 2. Making Predictions
```python
from meme_classifier import MemeClassifier, analyze_prediction

# Load the trained model
model = MemeClassifier()
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])

# Analyze a meme
result = analyze_prediction(model, 'path/to/meme.jpg', 'meme text')
print(result)
```

### 3. Using the Web Interface
1. Open the application in your browser
2. Upload a meme image
3. Enter the meme text (if any)
4. Click "Analyze" to get results
5. View the prediction and confidence score

## Model Output
The model provides:
- Binary classification (Hateful/Non-hateful)
- Confidence scores
- Probabilities for both classes
- Timestamp of prediction

## Troubleshooting

### Common Issues
1. **Model Loading Error**
   - Ensure the model checkpoint exists in the `checkpoints` directory
   - Verify the model file path is correct

2. **Dataset Not Found**
   - Check if the dataset is properly downloaded
   - Verify the dataset path in the configuration

3. **Dependencies Issues**
   - Ensure all requirements are installed
   - Try updating pip: `python -m pip install --upgrade pip`

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Facebook AI for the Hateful Memes dataset
- Hugging Face for the model architecture
- Streamlit for the web interface framework 
