import os
import cv2
import tensorflow as tf
from models.cnn_model import DiseaseDetectionModel
from utils.image_preprocessing import ImagePreprocessor
from config import Config
import keras

def predict_image(image_path, model_path):
    """Predict tumor presence from a brain MRI image."""
    # Verify files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load and preprocess image
        preprocessor = ImagePreprocessor(Config.INPUT_SHAPE[:2])
        image = cv2.imread(image_path)
        processed_image = preprocessor.preprocess_image(image)
        
        # Load model
        model = keras.models.load_model(model_path)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Get class prediction and confidence
        class_idx = prediction.argmax()
        class_prediction = Config.CLASS_NAMES[class_idx]
        confidence = float(prediction.max())
        
        return {
            'prediction': class_prediction,
            'confidence': confidence,
            'raw_predictions': {
                'no_tumor': float(prediction[0][0]),
                'tumor': float(prediction[0][1])
            }
        }
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    # Hardcode the paths to the image and model
    image_path = r'data\brain_tumor_dataset\yes\Y1.jpg'  # specify your image path here
    model_path = r'saved_models\brain_tumor_model_20241218-140806.keras'  # specify your model path here
    
    try:
        result = predict_image(image_path, model_path)
        print("\nPrediction Results:")
        print(f"Classification: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print("\nRaw Predictions:")
        print(f"No Tumor: {result['raw_predictions']['no_tumor']*100:.2f}%")
        print(f"Tumor: {result['raw_predictions']['tumor']*100:.2f}%")
    except Exception as e:
        print(f"Error: {str(e)}")
