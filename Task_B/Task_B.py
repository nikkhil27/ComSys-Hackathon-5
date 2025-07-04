import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Configure MPS (Metal Performance Shaders) for Apple Silicon
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

# Enable MPS if available
if tf.config.list_physical_devices('GPU'):
    print("MPS GPU detected and will be used")
    # Enable memory growth to avoid allocating all GPU memory at once
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("No MPS GPU detected, using CPU")

# Set mixed precision for better performance (optional)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Alternative MPS configuration for better memory management
with tf.device('/GPU:0'):
    # Force operations to run on GPU
    pass

# Monitor GPU memory usage
def print_gpu_memory():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        print(f"GPU memory info: {tf.config.experimental.get_memory_info('GPU:0')}")

# For binary classification, we can use TensorFlow's built-in metrics
# Top-1 accuracy is equivalent to binary accuracy for binary classification
def top1_accuracy_metric():
    """Returns a top-1 accuracy metric for binary classification"""
    return tf.keras.metrics.BinaryAccuracy(name='top1_accuracy', threshold=0.5)

# Function to calculate top-1 accuracy manually
def calculate_top1_accuracy(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate top-1 accuracy for binary classification
    For binary classification, this is equivalent to regular accuracy
    """
    # Get the predicted class (0 or 1) based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate accuracy (which is top-1 accuracy for binary classification)
    top1_accuracy = accuracy_score(y_true, y_pred)
    
    return top1_accuracy

# Call this function to monitor memory usage during training
# print_gpu_memory()

def load_image(path, size=(224, 224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    return np.array(img) / 255.0

def create_balanced_pairs(data_dir, image_size=(224, 224), max_pairs_per_person=10):
    """Create balanced positive and negative pairs"""
    positive_pairs, negative_pairs = [], []
    people = os.listdir(data_dir)
    
    # Create positive pairs
    for person in people:
        person_path = os.path.join(data_dir, person)
        true_img_path = os.path.join(person_path, f"{person.lower()}.jpg")
        if not os.path.exists(true_img_path):
            continue
            
        true_img = load_image(true_img_path, image_size)
        
        # Add positive pairs from distortions
        distort_dir = os.path.join(person_path, "distortion")
        if os.path.exists(distort_dir):
            distort_files = os.listdir(distort_dir)
            # Limit positive pairs to avoid overwhelming
            for fname in distort_files[:max_pairs_per_person]:
                pos_img = load_image(os.path.join(distort_dir, fname), image_size)
                positive_pairs.append([true_img, pos_img])
    
    # Create equal number of negative pairs
    num_positive = len(positive_pairs)
    people_with_images = []
    
    # Get all people with valid images
    for person in people:
        person_path = os.path.join(data_dir, person)
        true_img_path = os.path.join(person_path, f"{person.lower()}.jpg")
        if os.path.exists(true_img_path):
            people_with_images.append(person)
    
    # Generate negative pairs
    for _ in range(num_positive):
        # Pick two different people randomly
        person1, person2 = random.sample(people_with_images, 2)
        
        img1_path = os.path.join(data_dir, person1, f"{person1.lower()}.jpg")
        img2_path = os.path.join(data_dir, person2, f"{person2.lower()}.jpg")
        
        img1 = load_image(img1_path, image_size)
        img2 = load_image(img2_path, image_size)
        
        negative_pairs.append([img1, img2])
    
    # Combine and create labels
    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    
    # Shuffle
    combined = list(zip(all_pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)
    
    print(f"Created {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
    return np.array(pairs), np.array(labels)

def build_improved_siamese_model(input_shape):
    """Enhanced Siamese model with better feature extraction"""
    def build_base_network():
        inputs = layers.Input(shape=input_shape)
        
        # Enhanced feature extraction
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)  # L2 normalize
        
        return models.Model(inputs, x)
    
    base_network = build_base_network()
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    embed_a = base_network(input_a)
    embed_b = base_network(input_b)
    
    # Multiple distance metrics
    l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([embed_a, embed_b])
    l2_distance = layers.Lambda(lambda tensors: tf.square(tensors[0] - tensors[1]))([embed_a, embed_b])
    cosine_similarity = layers.Lambda(lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1, keepdims=True))([embed_a, embed_b])
    
    # Combine distances
    combined = layers.Concatenate()([l1_distance, l2_distance, cosine_similarity])
    
    # Classification head
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    siamese_model = models.Model(inputs=[input_a, input_b], outputs=outputs)
    return siamese_model

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss to handle class imbalance"""
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        loss = -alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1) - \
               (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
        return tf.reduce_mean(loss)
    return loss_fn

# Load balanced pairs
print("Creating balanced training pairs...")
train_pairs, train_labels = create_balanced_pairs(r'Task_B/train', max_pairs_per_person=8)
print("Creating balanced validation pairs...")
val_pairs, val_labels = create_balanced_pairs(r'Task_B/val', max_pairs_per_person=5)

# Split pairs into inputs
x_train_1 = train_pairs[:, 0]
x_train_2 = train_pairs[:, 1]
x_val_1 = val_pairs[:, 0]
x_val_2 = val_pairs[:, 1]

# Build improved model
model = build_improved_siamese_model(input_shape=(224, 224, 3))

# Compute class weights for remaining imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

# Alternative: Use weighted binary crossentropy instead of focal loss
def weighted_binary_crossentropy(pos_weight=1.0):
    def loss_fn(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, 
            logits=tf.math.log(y_pred / (1 - y_pred)), 
            pos_weight=pos_weight
        )
    return loss_fn

# Compile with focal loss, higher learning rate, and Top-1 accuracy
model.compile(
    loss=focal_loss(gamma=2.0, alpha=0.25),  # Use focal loss
    # loss='binary_crossentropy',  # Alternative: try standard binary crossentropy
    # loss=weighted_binary_crossentropy(pos_weight=2.0),  # Alternative: weighted BCE
    optimizer=optimizers.Adam(learning_rate=0.001),  # Increased learning rate from 0.0001 to 0.001
    metrics=['accuracy', top1_accuracy_metric()]  # Include both regular accuracy and top-1 accuracy
)

# Print model summary to check architecture
print("\nModel Architecture:")
model.summary()

# Check data distribution
print(f"\nTraining data distribution:")
print(f"Positive pairs (same person): {np.sum(train_labels == 1)}")
print(f"Negative pairs (different person): {np.sum(train_labels == 0)}")
print(f"Class balance ratio: {np.sum(train_labels == 1) / len(train_labels):.3f}")

print(f"\nValidation data distribution:")
print(f"Positive pairs (same person): {np.sum(val_labels == 1)}")
print(f"Negative pairs (different person): {np.sum(val_labels == 0)}")
print(f"Class balance ratio: {np.sum(val_labels == 1) / len(val_labels):.3f}")

# Check if images are loaded correctly
print(f"\nImage data shapes:")
print(f"Training images shape: {x_train_1.shape}, {x_train_2.shape}")
print(f"Validation images shape: {x_val_1.shape}, {x_val_2.shape}")
print(f"Image value range: [{np.min(x_train_1):.3f}, {np.max(x_train_1):.3f}]")

# Enhanced callbacks for better training
callbacks = [
    # Early stopping - stops training when validation loss stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,  # Increased patience to 10 epochs
        restore_best_weights=True,  # Restore the best weights
        verbose=1,
        mode='min'  # We want to minimize validation loss
    ),
    
    # Reduce learning rate when validation loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation loss
        factor=0.3,  # Reduce LR more aggressively (0.3 instead of 0.5)
        patience=5,  # Increased patience to 5 epochs
        min_lr=1e-6,  # Higher minimum learning rate
        verbose=1,
        mode='min'
    ),
    
    # Save the best model based on validation loss (more reliable than accuracy when model is struggling)
    tf.keras.callbacks.ModelCheckpoint(
        'best_siamese_model.h5', 
        monitor='val_loss',  # Monitor validation loss instead of accuracy
        save_best_only=True,
        verbose=1,
        mode='min'  # We want to minimize loss
    ),
    
    # Additional callback to monitor accuracy
    tf.keras.callbacks.ModelCheckpoint(
        'best_siamese_model_acc.h5', 
        monitor='val_top1_accuracy',  # Monitor validation accuracy
        save_best_only=True,
        verbose=0,
        mode='max'
    )
]

# Train with class weights
print("Training improved model on MPS with enhanced callbacks...")
print_gpu_memory()  # Check initial GPU memory

# Try training with a simpler loss function first if focal loss isn't working
print("\nStarting training...")
print("If the model continues to perform poorly (around 50% accuracy),")
print("consider switching to binary_crossentropy loss function.")

history = model.fit(
    [x_train_1, x_train_2], train_labels,
    validation_data=([x_val_1, x_val_2], val_labels),
    epochs=25,
    batch_size=16,  # Increased batch size for more stable gradients
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print_gpu_memory()  # Check GPU memory after training

# If the model is still performing poorly, let's try a different approach
if max(history.history['val_top1_accuracy']) < 0.6:
    print("\n" + "="*50)
    print("Model performance is poor. Trying alternative configuration...")
    print("="*50)
    
    # Create a new model with simpler loss
    model_alt = build_improved_siamese_model(input_shape=(224, 224, 3))
    
    # Compile with standard binary crossentropy
    model_alt.compile(
        loss='binary_crossentropy',  # Standard binary crossentropy
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy', top1_accuracy_metric()]
    )
    
    print("Retraining with binary crossentropy loss...")
    
    # Shorter training for the alternative model
    history_alt = model_alt.fit(
        [x_train_1, x_train_2], train_labels,
        validation_data=([x_val_1, x_val_2], val_labels),
        epochs=15,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Use the better performing model
    if max(history_alt.history['val_top1_accuracy']) > max(history.history['val_top1_accuracy']):
        print("Alternative model performs better. Using it for evaluation.")
        model = model_alt
        history = history_alt
    else:
        print("Original model performs better. Using it for evaluation.")

# Evaluate with threshold optimization for Top-1 accuracy
print("Optimizing threshold for Top-1 accuracy...")
y_pred_raw = model.predict([x_val_1, x_val_2])

best_thresh = 0.5
best_f1 = 0
best_top1_accuracy = 0

print("\nThreshold optimization results:")
print("Threshold | Top-1 Accuracy | Regular Accuracy | F1 Score")
print("-" * 60)

for thresh in np.arange(0.1, 0.9, 0.05):
    # Calculate Top-1 accuracy
    top1_accuracy = calculate_top1_accuracy(val_labels, y_pred_raw, threshold=thresh)
    
    # Calculate regular accuracy (should be the same for binary classification)
    y_pred = (y_pred_raw >= thresh).astype(int)
    regular_accuracy = accuracy_score(val_labels, y_pred)
    f1 = f1_score(val_labels, y_pred, average='macro')
    
    print(f"{thresh:.2f}      | {top1_accuracy:.4f}        | {regular_accuracy:.4f}          | {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_top1_accuracy = top1_accuracy
        best_thresh = thresh

print(f"\nBest Results:")
print(f"Threshold: {best_thresh:.2f}")
print(f"Top-1 Accuracy: {best_top1_accuracy:.4f}")
print(f"Macro F1: {best_f1:.4f}")

# Final evaluation with best threshold
y_pred_final = (y_pred_raw >= best_thresh).astype(int)

print("\nFinal Top-1 Accuracy Results:")
final_top1_accuracy = calculate_top1_accuracy(val_labels, y_pred_raw, threshold=best_thresh)
print(f"Top-1 Accuracy: {final_top1_accuracy:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(val_labels, y_pred_final, target_names=['Different Person', 'Same Person']))

print("\nConfusion Matrix:")
cm = confusion_matrix(val_labels, y_pred_final)
print(cm)

print(f"\nClass distribution in validation set:")
print(f"Different Person (0): {np.sum(val_labels == 0)}")
print(f"Same Person (1): {np.sum(val_labels == 1)}")

# Print training history with Top-1 accuracy
print("\nTraining History Summary:")
print(f"Best Training Top-1 Accuracy: {max(history.history['top1_accuracy']):.4f}")
print(f"Best Validation Top-1 Accuracy: {max(history.history['val_top1_accuracy']):.4f}")
print(f"Final Training Top-1 Accuracy: {history.history['top1_accuracy'][-1]:.4f}")
print(f"Final Validation Top-1 Accuracy: {history.history['val_top1_accuracy'][-1]:.4f}")