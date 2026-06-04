# Detection de deepfakes - Projet complet

## Auteurs

- Paul Sode
- Jules Araud
- Mohamed Hamza Laamarti

## Objectif

Ce projet construit une chaine complete de detection de deepfakes sur images de visages : choix du dataset, analyse exploratoire, baseline de machine learning, reseaux de neurones, transfer learning et dashboard Streamlit.

## Dataset

Dataset utilise : [`xhlulu/140k-real-and-fake-faces`](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).

Le dataset contient des visages reels issus de FFHQ et des visages generes par StyleGAN. Il n'est pas inclus dans ce depot afin d'eviter de versionner des donnees lourdes. Le notebook le telecharge via `kagglehub`.

## Structure

- `deepfake_detection_2026.ipynb` : notebook complet du rendu final.
- `models.py` : architectures Keras, CNN, MLP et MobileNetV2 par transfer learning.
- `app.py` : dashboard Streamlit pour exploiter le modele sauvegarde.
- `requirements.txt` : dependances Python.

## Installation

```bash
pip install -r requirements.txt
```

Si le telechargement Kaggle demande une authentification, configurer les identifiants Kaggle avant d'executer le notebook.

## Execution du notebook

```bash
jupyter notebook deepfake_detection_2026.ipynb
```

Le notebook complet couvre :

- Partie 1 : donnees, EDA, augmentation et baseline Ridge/Lasso/ElasticNet.
- Partie 2 : CNN entraine de zero, optimisation et evaluation.
- Partie 3 : transfer learning MobileNetV2, comparaison finale et sauvegarde du modele.

Le modele final est sauvegarde dans `artifacts/deepfake_mobilenetv2.keras`.

## Dashboard Streamlit

Apres avoir execute la Partie 3 du notebook et genere le modele :

```bash
streamlit run app.py
```

Le dashboard permet d'uploader une image `.jpg`, `.jpeg` ou `.png`, puis affiche la classe predite (`REAL` ou `FAKE`) et la confiance associee.

## Resultats principaux

La baseline de regression logistique regularisee compare Ridge/L2, Lasso/L1 et ElasticNet sur des images reduites a 32x32. Dans l'execution fournie, Lasso obtient environ `0.745` d'accuracy et `0.831` d'AUC sur le jeu de test.

Le CNN entraine de zero illustre l'apport des convolutions, mais sa version reduite reste sensible au reglage et au temps de calcul. La Partie 3 ajoute MobileNetV2 pre-entraine ImageNet pour exploiter le transfer learning et obtenir une approche avancee plus robuste.

## Labels Git recommandes

Les jalons peuvent etre marques avec les labels/commits suivants :

```bash
git tag data
git tag eda
git tag ml
git tag eval-ml
git tag dl
git tag opti-dl
git tag eval-dl
```

## CI/CD

Le workflow GitHub Actions dans `.github/workflows/ci.yml` verifie les fichiers Python et les imports principaux sans telecharger le dataset ni entrainer les modeles.

## Transparence IA

Des outils d'assistance IA ont ete utilises pour aider a structurer le projet, reformuler certaines explications, organiser le code, preparer le dashboard et mettre en place la CI. Les resultats, limites et choix techniques restent verifies dans le notebook.

## Limites

Les faux visages du dataset viennent d'un seul generateur, StyleGAN. Meme avec de bons scores, le modele peut apprendre des traces propres a ce generateur et ne garantit pas une generalisation parfaite a d'autres types de deepfakes, comme la diffusion ou l'echange de visages en video.
