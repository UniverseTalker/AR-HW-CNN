import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/arabic_letters_cnn.keras"
MAP_PATH   = "models/label_mapping.json"
IMG_SIZE   = (32, 32)

# Standard Arabic alphabet (28 letters)
LABEL_TO_ARABIC = {
    1: "ا",  2: "ب",  3: "ت",  4: "ث",  5: "ج",  6: "ح",  7: "خ",
    8: "د",  9: "ذ", 10: "ر", 11: "ز", 12: "س", 13: "ش", 14: "ص",
   15: "ض", 16: "ط", 17: "ظ", 18: "ع", 19: "غ", 20: "ف", 21: "ق",
   22: "ك", 23: "ل", 24: "م", 25: "ن", 26: "ه", 27: "و", 28: "ي",
}
# --- Load model and mapping once ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # keys were saved as strings sometimes -> convert safely
    index_to_label = {int(k): int(v) for k, v in mapping["index_to_label"].items()}
    return model, index_to_label

def preprocess_image(img: Image.Image):
    img = img.convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[..., None]          # (32,32,1)
    arr = np.expand_dims(arr, 0)  # (1,32,32,1)
    return arr

st.set_page_config(page_title="Arabic Letter Classifier", layout="centered")
st.title("Arabic Letter Recognizer (CNN)")
st.write("Upload a letter image (32x32 recommended). The model will predict the letter.")

model, index_to_label = load_assets()

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    pred_label = index_to_label[pred_idx]          # رقم من 1..28
    pred_letter = LABEL_TO_ARABIC.get(pred_label, "؟")

    st.metric("Predicted letter", pred_letter)
    st.write(f"Label: {pred_label}")

    st.subheader("Prediction")
   
    st.metric("Confidence", f"{confidence*100:.2f}%")

    # Top-3
    st.subheader("Top 3 predictions")
    topk = probs.argsort()[-3:][::-1]
    for rank, i in enumerate(topk, start=1):
        lab = index_to_label[int(i)]
        letter = LABEL_TO_ARABIC.get(lab, "؟")
        st.write(f"{rank}) {letter} (label {lab}) — {probs[int(i)]*100:.2f}%")

#st.caption("Note: labels are numeric (1..28). If you want Arabic letters names, we can map them.")