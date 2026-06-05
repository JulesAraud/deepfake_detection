import streamlit as st


st.set_page_config(
    page_title="Méthodologie",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Méthodologie")
st.caption("Comment le projet passe des données au prototype de détection.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Baseline Lasso", "0.831", "AUC test")
with col2:
    st.metric("CNN", "0.744", "AUC test")
with col3:
    st.metric("MobileNetV2", "0.804", "AUC test")

st.divider()

st.subheader("1. Données")
st.markdown(
    """
    Le projet utilise le dataset **140k Real and Fake Faces** :

    - **REAL** : visages réels issus de FFHQ ;
    - **FAKE** : visages synthétiques générés par StyleGAN ;
    - splits séparés en entraînement, validation et test ;
    - dataset équilibré entre vrais et faux visages.

    Ce choix évite une tâche trop simple où le modèle distinguerait seulement des photos
    et des dessins. Ici, les deux classes sont des visages réalistes.
    """
)

st.subheader("2. Analyse exploratoire")
eda1, eda2 = st.columns(2)
with eda1:
    st.markdown(
        """
        **Contrôles réalisés**

        - répartition REAL/FAKE ;
        - intensité moyenne ;
        - contraste ;
        - canaux RGB ;
        - doublons exacts.
        """
    )
with eda2:
    st.markdown(
        """
        **Objectif**

        Vérifier que le dataset ne contient pas d'anomalies évidentes et que la
        classification ne repose pas sur une fuite trop grossière.
        """
    )

st.subheader("3. Baseline machine learning")
st.markdown(
    """
    Les images sont réduites en **32×32**, aplaties en vecteurs, puis utilisées par une
    régression logistique régularisée.
    """
)

ml1, ml2, ml3 = st.columns(3)
with ml1:
    st.markdown("**Ridge / L2**  \nStabilise tous les coefficients.")
with ml2:
    st.markdown("**Lasso / L1**  \nSélectionne les pixels utiles.")
with ml3:
    st.markdown("**ElasticNet**  \nCompromis entre L1 et L2.")

st.info(
    "La baseline Lasso obtient la meilleure AUC du projet : 0.831. "
    "Elle reste donc une référence forte malgré sa simplicité."
)

st.subheader("4. CNN entraîné de zéro")
st.markdown(
    """
    Le CNN exploite la structure spatiale des images, contrairement à la baseline qui
    aplatit les pixels. Il utilise :

    - blocs `Conv2D → BatchNorm → MaxPooling` ;
    - `ReLU` pour limiter la saturation des gradients ;
    - `Dropout` pour réduire le surapprentissage ;
    - augmentation d'images pour améliorer la robustesse.
    """
)

st.warning(
    "Sur ce run, le CNN reste inférieur à la baseline : 0.744 d'AUC. "
    "Il apprend un signal utile, mais l'entraînement from scratch est coûteux et sensible."
)

st.subheader("5. Transfer learning MobileNetV2")
st.markdown(
    """
    MobileNetV2 réutilise des représentations visuelles apprises sur ImageNet. La base
    est gelée, puis une tête de classification binaire apprend à distinguer REAL/FAKE.

    Cette approche améliore le deep learning du projet :

    - AUC MobileNetV2 : **0.804** ;
    - entraînement plus rapide que le CNN compact ;
    - meilleur rappel sur la classe FAKE.
    """
)

st.subheader("6. Inférence dans le dashboard")
inf1, inf2 = st.columns(2)
with inf1:
    st.markdown(
        """
        **Prétraitement**

        1. image convertie en RGB ;
        2. redimensionnement en 128×128 ;
        3. passage au modèle `.keras`.
        """
    )
with inf2:
    st.markdown(
        """
        **Sortie**

        - sigmoid = `P(REAL)` ;
        - seuil par défaut : 0.5 ;
        - affichage du label et de la confiance.
        """
    )

st.subheader("7. Limites")
st.markdown(
    """
    Le dataset contient des faux issus de **StyleGAN uniquement**. Le modèle peut donc
    apprendre des artefacts propres à ce générateur et moins bien généraliser à :

    - des images produites par diffusion ;
    - du face swap vidéo ;
    - des images compressées ou fortement retouchées ;
    - des visages hors distribution.
    """
)

st.caption(
    "Le dashboard est un prototype académique : il aide à analyser une image, "
    "mais ne remplace pas une expertise forensique."
)
