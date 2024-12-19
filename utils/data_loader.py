# utils/data_loader.py
import os
import cv2
import numpy as np
from keras._tf_keras.keras.utils import to_categorical
from utils.image_preprocessing import ImagePreprocessor

class BrainMRIDataLoader:
    def __init__(self, data_dir, target_size=(224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.preprocessor = ImagePreprocessor(target_size)
        
    def load_dataset(self):
        """Load and preprocess the brain MRI dataset."""
        images = []
        labels = []
        
        # Dictionary to keep track of data distribution
        class_counts = {'no': 0, 'yes': 0}
        
        # Process each class directory
        for class_name in ['no', 'yes']:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                raise ValueError(f"Directory not found: {class_path}")
                
            print(f"Processing {class_name} images...")
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    
                    try:
                        image = cv2.imread(img_path)
                        if image is not None:
                            processed_image = self.preprocessor.preprocess_image(image)
                            images.append(processed_image[0])
                            labels.append(1 if class_name == 'yes' else 0)
                            class_counts[class_name] += 1
                        else:
                            print(f"Warning: Could not load image {img_path}")
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Total images: {sum(class_counts.values())}")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} images")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = to_categorical(np.array(labels))
        
        return X, y