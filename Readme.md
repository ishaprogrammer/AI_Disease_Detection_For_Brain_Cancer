# Brain Tumor Detection Using Deep Learning

## Project Overview
This project implements an AI-based system for detecting brain tumors in MRI scans using deep learning. The system uses a Convolutional Neural Network (CNN) to classify brain MRI images into two categories: with tumor (yes) and without tumor (no).

## Technologies Used
- **Python 3.8+**: Primary programming language
- **TensorFlow 2.x**: Deep learning framework for model creation and training
- **OpenCV (cv2)**: Image processing and manipulation
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and plotting training results
- **scikit-learn**: Data splitting and preprocessing utilities
- **Pandas**: Data manipulation and analysis

## Project Structure
```
brain_tumor_detection/
├── models/
│   ├── __init__.py
│   └── cnn_model.py          # CNN model architecture
├── utils/
│   ├── __init__.py
│   ├── image_preprocessing.py # Image preprocessing utilities
│   └── data_loader.py        # Data loading and processing
├── data/
│   └── brain_tumor_dataset/  # Dataset directory
│       ├── yes/             # Tumor-positive images
│       └── no/              # Tumor-negative images
├── saved_models/            # Directory for saved models
├── logs/                    # Training logs and visualizations
├── config.py               # Configuration parameters
├── train.py               # Training script
├── predict.py             # Prediction script
└── requirements.txt       # Project dependencies
```

## Dataset
The project uses the Brain MRI Images for Brain Tumor Detection dataset from Kaggle:
- Source: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- Contains MRI scans categorized into two classes:
  - With tumor (yes)
  - Without tumor (no)

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd brain_tumor_detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in the correct directory:
```bash
brain_tumor_detection/data/brain_tumor_dataset/
├── yes/  # Tumor-positive images
└── no/   # Tumor-negative images
```

## Usage

### Training the Model
```bash
python train.py
```
This will:
- Load and preprocess the dataset
- Train the CNN model
- Save the trained model
- Generate training history plots

### Making Predictions
```bash
python predict.py path/to/image.jpg path/to/saved_model.keras
```

## Model Architecture
The CNN model consists of:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Batch normalization layers
- Dropout layers for regularization
- Dense layers for final classification

## Features
- Custom image preprocessing pipeline
- Data augmentation techniques
- Training progress visualization
- Model checkpointing
- Early stopping to prevent overfitting
- Learning rate scheduling
- Detailed prediction outputs with confidence scores

## Performance Monitoring
- Training progress is logged using TensorBoard
- Training history plots are saved automatically
- Model checkpoints are saved for best performing epochs

## Error Handling
The system includes comprehensive error handling for:
- Missing files and directories
- Invalid image formats
- Model loading issues
- Prediction errors

## Future Improvements
- Support for multiple tumor types
- Integration with web interface
- Additional data augmentation techniques
- Model ensemble approaches
- DICOM format support
- Segmentation capabilities

## Notes
- The model is trained on MRI images specifically
- Performance may vary depending on image quality
- Predictions should be verified by medical professionals
- This tool is for research purposes only

