import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from dsi import TBNetDSI
import os

# ==========================
# Configuration
# ==========================
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 2
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 1
EPOCHS = 10  # reduced to prevent overfitting
MODEL_SAVE_PATH = 'tbnet_best_model.h5'

# ==========================
# Load datasets
# ==========================
dsi = TBNetDSI(data_path='data/')
train_dataset, train_size, train_batch = dsi.get_train_dataset()
val_dataset, val_size, val_batch = dsi.get_validation_dataset()
test_dataset, test_size, test_batch = dsi.get_test_dataset()

# ==========================
# Map datasets to proper format
# ==========================
train_dataset = train_dataset.map(lambda x: (x['image'], x['label/one_hot'])).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(lambda x: (x['image'], x['label/one_hot'])).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(lambda x: (x['image'], x['label/one_hot'])).prefetch(tf.data.AUTOTUNE)

# ==========================
# Define the model
# ==========================
def build_tbnet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # dropout to prevent overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = build_tbnet_model()
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# ==========================
# Callbacks
# ==========================
checkpoint_cb = callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1
)
earlystop_cb = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)
csv_logger_cb = callbacks.CSVLogger('tbnet_training_log.csv', append=True)

# ==========================
# Train the model
# ==========================
history = model.fit(
    train_dataset,
    steps_per_epoch=train_size // BATCH_SIZE_TRAIN,
    validation_data=val_dataset,
    validation_steps=val_size // BATCH_SIZE_VAL,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb, csv_logger_cb],
    verbose=1
)

# ==========================
# Evaluate on test dataset
# ==========================
test_results = model.evaluate(test_dataset, steps=test_size // BATCH_SIZE_TEST)
print("\nTest Results:")
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

# ==========================
# Save final model
# ==========================
final_model_path = 'tbnet_final_model.h5'
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")
