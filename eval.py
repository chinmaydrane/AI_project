import os
import numpy as np
import tensorflow as tf
from dsi import TBNetDSI, NUM_CLASSES
from sklearn.metrics import confusion_matrix

# Paths
DATA_PATH = 'data/'
MODEL_PATH = 'models/tbnet_best.h5'  # or 'tbnet_final.h5'

# Load dataset
dsi = TBNetDSI(data_path=DATA_PATH)
test_dataset, test_size, batch_size = dsi.get_test_dataset()
test_dataset = test_dataset.map(lambda x, y: (x, y))

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Predict on test dataset
y_true = []
y_pred = []

for images, labels in test_dataset.take(test_size // batch_size):
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix
matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(matrix)

# Sensitivity / Recall
sens = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print(f"Sensitivity Normal: {sens[0]:.3f}, Tuberculosis: {sens[1]:.3f}")

# Positive Predictive Value / Precision
ppv = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
print(f"PPV Normal: {ppv[0]:.3f}, Tuberculosis: {ppv[1]:.3f}")

# Overall accuracy
accuracy = np.sum(y_true == y_pred) / len(y_true)
print(f"Overall Test Accuracy: {accuracy:.3f}")
