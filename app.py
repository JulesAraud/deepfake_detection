from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


IMG_SIZE = 128
MODEL_PATH = Path("artifacts/deepfake_mobilenetv2.keras")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(resized, dtype="float32")
    return image, array[None, ...]


def main():
    st.set_page_config(page_title="Deepfake Detection", layout="centered")
    st.title("Détection de deepfakes")

    if not MODEL_PATH.exists():
        st.warning(
            "Modèle introuvable. Exécutez d'abord le notebook complet pour générer "
            "`artifacts/deepfake_mobilenetv2.keras`."
        )
        st.stop()

    model = load_model()
    uploaded = st.file_uploader("Image de visage", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Ajoutez une image pour obtenir une prédiction.")
        st.stop()

    image, batch = prepare_image(uploaded)
    prob_real = float(model.predict(batch, verbose=0).ravel()[0])
    label = "REAL" if prob_real >= 0.5 else "FAKE"
    confidence = prob_real if label == "REAL" else 1.0 - prob_real

    st.image(image, caption="Image analysée", use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Classe prédite", label)
    col2.metric("Confiance", f"{confidence:.1%}")

    st.progress(min(max(confidence, 0.0), 1.0))
    st.caption(f"Probabilité REAL brute : {prob_real:.3f}")


if __name__ == "__main__":
    main()
