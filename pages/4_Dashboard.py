from pathlib import Path

import pandas as pd
import streamlit as st


CNN_PATH = Path("artifacts/deepfake_cnn.keras")
MOBILENET_PATH = Path("artifacts/deepfake_mobilenetv2.keras")


st.set_page_config(
    page_title="Dashboard projet",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Dashboard projet")
st.caption("Vue synthétique des résultats et artefacts du projet.")

scores = pd.DataFrame(
    [
        {
            "Modèle": "Baseline Lasso",
            "Famille": "ML classique",
            "Accuracy": 0.745,
            "AUC": 0.831,
            "Temps (s)": 8.4,
        },
        {
            "Modèle": "CNN",
            "Famille": "Deep learning",
            "Accuracy": 0.677,
            "AUC": 0.744,
            "Temps (s)": 2630.8,
        },
        {
            "Modèle": "CNN + TTA",
            "Famille": "Deep learning",
            "Accuracy": 0.670,
            "AUC": 0.742,
            "Temps (s)": 2630.8,
        },
        {
            "Modèle": "MobileNetV2",
            "Famille": "Transfer learning",
            "Accuracy": 0.705,
            "AUC": 0.804,
            "Temps (s)": 751.1,
        },
    ]
)

recalls = pd.DataFrame(
    [
        {"Modèle": "Baseline Lasso", "FAKE": 0.77, "REAL": 0.72},
        {"Modèle": "CNN", "FAKE": 0.73, "REAL": 0.62},
        {"Modèle": "CNN + TTA", "FAKE": 0.73, "REAL": 0.61},
        {"Modèle": "MobileNetV2", "FAKE": 0.84, "REAL": 0.57},
    ]
).set_index("Modèle")

best_auc = scores.loc[scores["AUC"].idxmax()]
best_dl = scores[scores["Famille"] != "ML classique"].sort_values("AUC", ascending=False).iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Meilleure AUC", f"{best_auc['AUC']:.3f}", best_auc["Modèle"])
with col2:
    st.metric("Meilleur DL", f"{best_dl['AUC']:.3f}", best_dl["Modèle"])
with col3:
    st.metric("Train le plus rapide", "8.4 s", "Baseline Lasso")
with col4:
    st.metric("Dataset test", "600", "300 REAL / 300 FAKE")

st.divider()

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Performance des modèles")
    chart_df = scores.set_index("Modèle")[["Accuracy", "AUC"]]
    st.bar_chart(chart_df)

with right:
    st.subheader("Temps d'entraînement")
    time_df = scores.set_index("Modèle")[["Temps (s)"]]
    st.bar_chart(time_df)

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Rappel par classe")
    st.bar_chart(recalls)

with col_b:
    st.subheader("Artefacts modèles")
    st.write("État des fichiers utilisés par l'application.")

    cnn_status = "Disponible" if CNN_PATH.exists() else "Manquant"
    mobile_status = "Disponible" if MOBILENET_PATH.exists() else "Manquant"

    artifact_df = pd.DataFrame(
        [
            {"Artefact": "CNN", "Chemin": str(CNN_PATH), "Statut": cnn_status},
            {
                "Artefact": "MobileNetV2",
                "Chemin": str(MOBILENET_PATH),
                "Statut": mobile_status,
            },
        ]
    )
    st.dataframe(artifact_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Lecture rapide")

st.markdown(
    """
    - **Lasso** reste la meilleure référence globale en AUC.
    - **MobileNetV2** est le meilleur modèle deep learning du projet.
    - **CNN + TTA** n'améliore pas le CNN sur ce run.
    - **MobileNetV2** détecte bien les faux (`recall FAKE = 0.84`) mais confond encore des vrais.
    """
)
