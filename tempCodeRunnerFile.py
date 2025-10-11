import os
import argparse
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from dsi import *
from sklearn.metrics import confusion_matrix

import json

tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_TENSOR = "image:0"
LABEL_TENSOR = "classification/label:0"
LOSS_TENSOR = "add:0"
PREDICTION_TENSOR = "ArgMax:0"

parser = argparse.ArgumentParser(description='TB-Net Training')
parser.add_argument('--weightspath', default='TB-Net', type=str, help='Path to checkpoint folder')
parser.add_argument('--metaname', default='model_train.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpt')
parser.add_argument('--datapath', default='data/', type=str, help='Root folder containing the dataset')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--savepath', default='models/', type=str, help='Folder for models to be saved in')
args = parser.parse_args()

LEARNING_RATE = args.lr
OUTPUT_PATH = args.savepath
EPOCHS = args.epochs
VALIDATE_EVERY = 5

# Create folders
os.makedirs(OUTPUT_PATH, exist_ok=True)
LOGS_PATH = os.path.join(OUTPUT_PATH, "logs")
os.makedirs(LOGS_PATH, exist_ok=True)

def eval_and_log(sess, graph, val_or_test, dataset, image_tensor, label_tensor, pred_tensor, loss_tensor):
    y_true = []
    y_pred = []
    total_loss = 0
    num_evaled = 0

    iterator = dataset.make_initializable_iterator()
    datasets = {val_or_test: {'dataset': dataset, 'iterator': iterator, 'gn_op': iterator.get_next()}}
    sess.run(datasets[val_or_test]['iterator'].initializer)

    while True:
        try:
            data_dict = sess.run(datasets[val_or_test]['gn_op'])
            images = data_dict['image']
            labels = data_dict['label/one_hot'].argmax(axis=1)
            pred = sess.run(pred_tensor, feed_dict={image_tensor: images})

            y_true.extend(labels)
            y_pred.extend(pred)
            num_evaled += len(pred)

            if val_or_test == "val":
                total_loss += sess.run(loss_tensor, feed_dict={image_tensor: images, label_tensor: labels})
        except tf.errors.OutOfRangeError:
            break

    # Compute metrics
    matrix = confusion_matrix(np.array(y_true), np.array(y_pred))
    matrix = matrix.astype('float')
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]

    metrics = {
        'num_images': num_evaled,
        'confusion_matrix': matrix.tolist(),
        'class_acc': class_acc,
        'ppv': ppvs,
        'total_loss': float(total_loss)
    }

    # Save metrics log
    log_file = os.path.join(LOGS_PATH, f"{val_or_test}_metrics.json")
    with open(log_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ {val_or_test.upper()} evaluation done: {num_evaled} images.")
    print("Confusion Matrix:\n", matrix)
    print('Sens Normal: {:.3f}, Tuberculosis: {:.3f}'.format(*class_acc))
    print('PPV Normal: {:.3f}, Tuberculosis: {:.3f}'.format(*ppvs))
    return metrics

# Load dataset
dsi = TBNetDSI(data_path=args.datapath)
train_dataset, train_dataset_size, train_batch_size = dsi.get_train_dataset()
val_dataset, _, _ = dsi.get_validation_dataset()
test_dataset, _, _ = dsi.get_test_dataset()

sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
graph = tf.get_default_graph()

image_tensor = graph.get_tensor_by_name(INPUT_TENSOR)
label_tensor = graph.get_tensor_by_name(LABEL_TENSOR)
pred_tensor = graph.get_tensor_by_name(PREDICTION_TENSOR)
loss_tensor = graph.get_tensor_by_name(LOSS_TENSOR)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_tensor)

sess.run(tf.global_variables_initializer())
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

# Save base model
base_model_path = os.path.join(OUTPUT_PATH, "Baseline/TB-Net")
os.makedirs(base_model_path, exist_ok=True)
saver.save(sess, base_model_path)
print(f"Baseline checkpoint saved to {base_model_path}")

# Eval before training
eval_and_log(sess, graph, "test", test_dataset, image_tensor, label_tensor, pred_tensor, loss_tensor)

# Training loop
num_batches = train_dataset_size // train_batch_size
progbar = tf.keras.utils.Progbar(num_batches)
for epoch in range(EPOCHS):
    iterator = train_dataset.make_initializable_iterator()
    sess.run(iterator.initializer)
    datasets = {'train': {'dataset': train_dataset, 'iterator': iterator, 'gn_op': iterator.get_next()}}
    
    for i in range(num_batches):
        data_dict = sess.run(datasets['train']['gn_op'])
        batch_x = data_dict['image']
        batch_y = data_dict['label/one_hot'].argmax(axis=1)
        sess.run(train_op, feed_dict={image_tensor: batch_x, label_tensor: batch_y})
        progbar.update(i+1)

    if epoch % VALIDATE_EVERY == 0 or epoch == EPOCHS-1:
        val_metrics = eval_and_log(sess, graph, "val", val_dataset, image_tensor, label_tensor, pred_tensor, loss_tensor)
        epoch_model_path = os.path.join(OUTPUT_PATH, f"Epoch_{epoch+1}/TB-Net")
        os.makedirs(epoch_model_path, exist_ok=True)
        saver.save(sess, epoch_model_path)
        print(f"Checkpoint saved for epoch {epoch+1}")

# Final evaluation
eval_and_log(sess, graph, "test", test_dataset, image_tensor, label_tensor, pred_tensor, loss_tensor)
print("🎉 Training Complete! All logs and checkpoints saved in:", OUTPUT_PATH)
