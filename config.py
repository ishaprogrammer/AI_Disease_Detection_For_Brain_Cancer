# config.py
import os

class Config:
    # Base directory (project root)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Model parameters
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 2
    CLASS_NAMES = ['no', 'yes']  # no tumor, yes tumor
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Data parameters
    TRAIN_TEST_SPLIT = 0.2
    VALIDATION_SPLIT = 0.2
    
    # Paths
    DATA_DIR = os.path.join(BASE_DIR, "data", "brain_tumor_dataset")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models")
    LOGS_PATH = os.path.join(BASE_DIR, "logs")
    
    # Augmentation parameters
    AUGMENTATION = True