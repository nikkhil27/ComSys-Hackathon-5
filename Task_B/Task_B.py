import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seed(42)

augmenter = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1
)

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

from PIL import Image
import numpy as np
import tensorflow as tf

def load_image(path, size=(224, 224), augment=False):
    """Loads and optionally augments an image."""
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize(size)
        img = np.array(img) / 255.0  # Normalize to [0, 1]

        if augment:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
            img = data_gen.random_transform(img)
        return img
    except Exception as e:
        print(f"Error loading image: {path} - {e}")
        return np.zeros((*size, 3))  # fallback to blank image


import os
import random
import numpy as np

def create_balanced_pairs(data_dir, image_size=(224, 224), max_pairs_per_person=10, augment=False, allowed_distortions=None):
    """
    Create balanced positive and negative face pairs with optional augmentation and distortion filtering.
    - `allowed_distortions`: list of keywords (e.g. ['fog', 'lowlight']) to include only mild distortions.
    """
    positive_pairs, negative_pairs = [], []
    people = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]

    # For fast access later
    valid_people = []

    for person in people:
        person_path = os.path.join(data_dir, person)
        true_img_path = os.path.join(person_path, f"{person.lower()}.jpg")
        if not os.path.exists(true_img_path):
            continue

        true_img = load_image(true_img_path, image_size, augment=False)
        distort_dir = os.path.join(person_path, "distortion")
        if not os.path.exists(distort_dir):
            continue

        # Apply distortion filtering if specified
        distort_files = sorted(os.listdir(distort_dir))
        if allowed_distortions:
            distort_files = [f for f in distort_files if any(key in f.lower() for key in allowed_distortions)]

        # Limit number of distortions
        distort_files = distort_files[:max_pairs_per_person]
        for fname in distort_files:
            dist_path = os.path.join(distort_dir, fname)
            if not os.path.exists(dist_path):
                continue
            distorted_img = load_image(dist_path, image_size, augment=augment)
            positive_pairs.append([true_img, distorted_img])

        valid_people.append(person)

    # Negative pairs
    num_positive = len(positive_pairs)
    for _ in range(num_positive):
        person1, person2 = random.sample(valid_people, 2)

        path1 = os.path.join(data_dir, person1, f"{person1.lower()}.jpg")
        path2 = os.path.join(data_dir, person2, f"{person2.lower()}.jpg")
        if not os.path.exists(path1) or not os.path.exists(path2):
            continue

        img1 = load_image(path1, image_size)
        img2 = load_image(path2, image_size)
        negative_pairs.append([img1, img2])

    # Combine & shuffle
    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    combined = list(zip(all_pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)

    print(f"✅ Created {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")
    return np.array(pairs), np.array(labels)

def build_improved_siamese_model(input_shape):
    """Enhanced Siamese model with better feature extraction - MODEL 1"""
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
train_pairs, train_labels = create_balanced_pairs(r'Task_B/train', max_pairs_per_person=8,augment = False)
print("Creating balanced validation pairs...")
val_pairs, val_labels = create_balanced_pairs(r'Task_B/val', max_pairs_per_person=5,augment = False)

# Split pairs into inputs
x_train_1 = train_pairs[:, 0]
x_train_2 = train_pairs[:, 1]
x_val_1 = val_pairs[:, 0]
x_val_2 = val_pairs[:, 1]

# Build Model 1 (Improved Siamese Model)
print("\n" + "="*60)
print("BUILDING MODEL 1: IMPROVED SIAMESE MODEL")
print("="*60)

model = build_improved_siamese_model(input_shape=(224, 224, 3))

# Compute class weights for remaining imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

# Compile model
model.compile(
    loss=focal_loss(gamma=2.0, alpha=0.25),  # Use focal loss
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', top1_accuracy_metric()]
)

# Print model summary
print("\nImproved Siamese Model Architecture:")
model.summary()

# Enhanced callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-6,
        verbose=1,
        mode='min'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_siamese_model_improved.h5', 
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    )
]

# Train model
print("\nTraining Improved Siamese Model...")
history = model.fit(
    [x_train_1, x_train_2], train_labels,
    validation_data=([x_val_1, x_val_2], val_labels),
    epochs=20,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Print model performance
best_val_acc = max(history.history['val_top1_accuracy'])
print(f"\nModel Best Validation Top-1 Accuracy: {best_val_acc:.4f}")

# Evaluate model
print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)

# Get model predictions
model_pred = model.predict([x_val_1, x_val_2])

print("\nThreshold Optimization:")
print("Threshold | Top-1 Accuracy | F1 Score")
print("-" * 40)

best_thresh = 0.5
best_acc = 0
best_f1 = 0

for thresh in np.arange(0.1, 0.9, 0.05):
    acc = calculate_top1_accuracy(val_labels, model_pred, threshold=thresh)
    y_pred = (model_pred >= thresh).astype(int)
    f1 = f1_score(val_labels, y_pred, average='macro')
    
    print(f"{thresh:.2f}      | {acc:.4f}        | {f1:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh
        best_f1 = f1

print(f"\nBest Results:")
print(f"Threshold: {best_thresh:.2f}")
print(f"Top-1 Accuracy: {best_acc:.4f}")
print(f"F1 Score: {best_f1:.4f}")

# Final evaluation
y_pred_final = (model_pred >= best_thresh).astype(int)

print("\nFinal Classification Report:")
print(classification_report(val_labels, y_pred_final, target_names=['Different Person', 'Same Person']))

print("\nConfusion Matrix:")
cm = confusion_matrix(val_labels, y_pred_final)
print(cm)

# Save model
model.save('improved_siamese_model.h5')
print("\nModel saved as 'improved_siamese_model.h5'")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Final Top-1 Accuracy: {best_acc:.4f}")
print(f"Model saved as: improved_siamese_model.h5")
print("="*60)

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