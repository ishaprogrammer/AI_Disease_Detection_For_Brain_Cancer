# utils/image_preprocessing.py
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def preprocess_image(self, image):
        """Preprocess a single image for the model."""
        if image is None:
            raise ValueError("Image could not be loaded")
            
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def apply_augmentation(self, image):
        """Apply data augmentation techniques specific to MRI images."""
        # Random rotation
        angle = np.random.uniform(-15, 15)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height))
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.9, 1.1)
        image = cv2.multiply(image, brightness)
        
        # Random contrast adjustment
        contrast = np.random.uniform(0.9, 1.1)
        image = cv2.multiply(image, contrast)
        
        # Clip values to valid range
        image = np.clip(image, 0, 1)
        
        return image