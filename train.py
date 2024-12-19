import os
import tensorflow as tf
from models.cnn_model import DiseaseDetectionModel
from utils.data_loader import BrainMRIDataLoader
from config import Config
import matplotlib.pyplot as plt
from datetime import datetime
import keras

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [Config.MODEL_SAVE_PATH, Config.LOGS_PATH]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def train_model():
    setup_directories()
    
    # Create data loader
    print("\nInitializing data loader...")
    data_loader = BrainMRIDataLoader(Config.DATA_DIR, Config.INPUT_SHAPE[:2])
    
    try:
        print("Loading dataset...")
        X, y = data_loader.load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TRAIN_TEST_SPLIT,
        random_state=42,
        stratify=y
    )
    
    # Create and train model
    print("\nCreating model...")
    model = DiseaseDetectionModel(
        input_shape=Config.INPUT_SHAPE,
        num_classes=Config.NUM_CLASSES
    )
    model.compile_model(learning_rate=Config.LEARNING_RATE)
    
    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, f'brain_tumor_model_{timestamp}.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(Config.LOGS_PATH, timestamp)
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = os.path.join(Config.MODEL_SAVE_PATH, f'brain_tumor_model_final_{timestamp}.keras')
    model.model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Plot and save training history
    plot_training_history(history, timestamp)
    
    return history

def plot_training_history(history, timestamp):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(Config.LOGS_PATH, f'training_history_{timestamp}.png')
    plt.savefig(plot_path)
    print(f"\nTraining history plot saved as {plot_path}")

if __name__ == "__main__":
    train_model()