import os, re
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

# مسارات مجلدات  التدريب و الفحص
TRAIN_DIR = r"C:\Users\Lenovo\Desktop\BIg academy files\IP\ip\Train Images 13440x32x32\train"
TEST_DIR  = r"C:\Users\Lenovo\Desktop\BIg academy files\IP\ip\Test Images 3360x32x32\test"

IMG_H, IMG_W = 32, 32
CHANNELS = 1  # grayscale
BATCH_SIZE = 64
EPOCHS = 15

label_re = re.compile(r"label_(\d+)")

def load_images(folder):
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for f in files:
        m = label_re.search(f)
        if not m:
            continue
        label = int(m.group(1))
        path = os.path.join(folder, f)

        img = Image.open(path).convert("L").resize((IMG_W, IMG_H))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr[..., None]  # (32,32,1)

        X.append(arr)
        y.append(label)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)
    return X, y

print("Loading data...")
X_train, y_train = load_images(TRAIN_DIR)
X_test, y_test = load_images(TEST_DIR)

# مهم: labels عندك تبدأ من 1 غالبًا -> نحولها إلى 0..C-1
# ونضمن وجود mapping ثابت
unique_labels = np.unique(np.concatenate([y_train, y_test]))
label_to_index = {lab:i for i, lab in enumerate(unique_labels)}
index_to_label = {i:lab for lab, i in label_to_index.items()}

y_train_i = np.array([label_to_index[v] for v in y_train], dtype=np.int32)
y_test_i  = np.array([label_to_index[v] for v in y_test], dtype=np.int32)

num_classes = len(unique_labels)
print("Train:", X_train.shape, "Test:", X_test.shape, "Classes:", num_classes)
print("Label mapping:", label_to_index)

# Dataset pipeline
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_i)).shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test_i)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# CNN model (مناسب لـ32x32)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_H, IMG_W, CHANNELS)),

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluation
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test_i, y_pred)
print("\nTest Accuracy:", acc)

print("\nClassification report:")
print(classification_report(y_test_i, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test_i, y_pred))

# Save model + label mapping
os.makedirs("models", exist_ok=True)
model.save("models/arabic_letters_cnn.keras")

import json

# حوّلي المفاتيح/القيم إلى types عادية (Python int / str)
label_to_index_clean = {str(int(k)): int(v) for k, v in label_to_index.items()}
index_to_label_clean = {str(int(k)): int(v) for k, v in index_to_label.items()}

with open("models/label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {"label_to_index": label_to_index_clean, "index_to_label": index_to_label_clean},
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSaved: models/arabic_letters_cnn.keras and models/label_mapping.json")