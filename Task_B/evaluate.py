import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# Set seeds for reproducibility
# -------------------------
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# -------------------------
# Load test data
# -------------------------
test_dir = 'path/to/test/'  # update this

img_size = (224, 224)  # or whatever your training size was
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for correct label matching
)

# -------------------------
# Load model
# -------------------------
model = load_model('improved_siamese_model.h5')

# -------------------------
# Evaluate
# -------------------------
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# -------------------------
# Print Results
# -------------------------
print("Confusion Matrix")
print(confusion_matrix(true_classes, predicted_classes))

print("\nClassification Report")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
