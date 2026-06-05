from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


IMG_SIZE = 128
CNN_PATH = Path("artifacts/deepfake_cnn.keras")
MOBILENET_PATH = Path("artifacts/deepfake_mobilenetv2.keras")


@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    batch = np.asarray(resized, dtype="float32")[None, ...]
    return image, batch


def predict(model, batch):
    prob_real = float(model.predict(batch, verbose=0).ravel()[0])
    label = "REAL" if prob_real >= 0.5 else "FAKE"
    confidence = prob_real if label == "REAL" else 1.0 - prob_real
    return label, prob_real, confidence


st.set_page_config(
    page_title="Comparaison modèles",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("CNN vs MobileNetV2")
st.caption("Même image, deux modèles, verdicts côte à côte.")

cnn_ok = CNN_PATH.exists()
mobilenet_ok = MOBILENET_PATH.exists()

if not cnn_ok and not mobilenet_ok:
    st.warning("Aucun modèle trouvé. Générez d'abord les artefacts depuis le notebook.")
    st.stop()

uploaded = st.file_uploader("Image à comparer", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Ajoutez une image pour comparer les prédictions.")
    st.stop()

image, batch = prepare_image(uploaded)
st.image(image, caption="Image analysée", use_container_width=True)

models = []
if cnn_ok:
    models.append(("CNN", CNN_PATH))
if mobilenet_ok:
    models.append(("MobileNetV2", MOBILENET_PATH))

cols = st.columns(len(models))
results = []

for col, (name, path) in zip(cols, models):
    with col:
        st.subheader(name)
        model = load_model(str(path))
        with st.spinner("Analyse..."):
            label, prob_real, confidence = predict(model, batch)

        results.append((name, label, confidence, prob_real))
        st.metric("Prédiction", label)
        st.metric("Confiance", f"{confidence:.1%}")
        st.progress(confidence)
        st.caption(f"P(REAL) = {prob_real:.3f}")

if len(results) == 2:
    first, second = results
    if first[1] == second[1]:
        st.success(f"Consensus : {first[1]}")
    else:
        st.warning(
            f"Désaccord : {first[0]} prédit {first[1]}, "
            f"{second[0]} prédit {second[1]}."
        )
