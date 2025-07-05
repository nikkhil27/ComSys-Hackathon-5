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

# Data loading and augmentation setup omitted for brevity, assumed same as original

# Load data
train_pairs, train_labels = create_balanced_pairs(r'Task_B/train', max_pairs_per_person=8, augment=False)
val_pairs, val_labels = create_balanced_pairs(r'Task_B/val', max_pairs_per_person=5, augment=False)

x_train_1 = train_pairs[:, 0]
x_train_2 = train_pairs[:, 1]
x_val_1 = val_pairs[:, 0]
x_val_2 = val_pairs[:, 1]

# Build best model only
model = build_improved_siamese_model(input_shape=(224, 224, 3))

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

# Compile
model.compile(
    loss=focal_loss(gamma=2.0, alpha=0.25),
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', top1_accuracy_metric()]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('best_siamese_model_improved.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# Train
history = model.fit(
    [x_train_1, x_train_2], train_labels,
    validation_data=([x_val_1, x_val_2], val_labels),
    epochs=20,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
pred = model.predict([x_val_1, x_val_2])
best_thresh = 0.5
best_acc = 0
for thresh in np.arange(0.1, 0.9, 0.1):
    acc = calculate_top1_accuracy(val_labels, pred, threshold=thresh)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"Best Top-1 Accuracy: {best_acc:.4f} at threshold {best_thresh:.2f}")

final_pred = (pred >= best_thresh).astype(int)
print("\nClassification Report:")
print(classification_report(val_labels, final_pred, target_names=['Different', 'Same']))

print("\nConfusion Matrix:")
print(confusion_matrix(val_labels, final_pred))

print("\nDone. Model saved as best_siamese_model_improved.h5")
