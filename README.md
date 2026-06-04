# Detection de deepfakes - Data Science & Baseline ML

## Auteurs

- Paul Sode
- Jules Araud
- Mohamed Hamza Laamarti

## Objectif

Ce projet construit une premiere solution de detection de deepfakes a partir d'images de visages. La demarche couvre le choix du dataset, l'analyse exploratoire, la preparation des donnees, une baseline de machine learning classique et l'analyse biais/variance.

## Dataset

Dataset utilise : [`xhlulu/140k-real-and-fake-faces`](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).

Le dataset contient des visages reels issus de FFHQ et des visages generes par StyleGAN. Il n'est pas inclus dans ce depot afin d'eviter de versionner des donnees lourdes. Le notebook le telecharge via `kagglehub`.

## Fichiers

- `deepfake_detection_partie1.ipynb` : notebook principal de la v1.
- `partie1_utils.py` : fonctions metier reutilisables pour le chargement, l'EDA, l'augmentation et les features ML.
- `requirements.txt` : dependances Python necessaires.
- `.gitignore` : exclusions pour les donnees, caches et fichiers locaux.

## Installation

```bash
pip install -r requirements.txt
```

Si le telechargement Kaggle demande une authentification, configurer les identifiants Kaggle avant d'executer le notebook.

## Execution

Lancer Jupyter, ouvrir `deepfake_detection_partie1.ipynb`, puis executer les cellules dans l'ordre.

```bash
jupyter notebook
```

## Resultats principaux

La baseline de regression logistique regularisee compare Ridge/L2, Lasso/L1 et ElasticNet sur des images reduites a 32x32. Dans l'execution fournie, Lasso obtient environ `0.745` d'accuracy et `0.831` d'AUC sur le jeu de test.

L'analyse biais/variance montre un surapprentissage net : l'accuracy train est proche de `0.979`, contre environ `0.745` sur le test. La regularisation autour de `C = 0.1` donne le meilleur compromis observe.

## Transparence IA

Des outils d'assistance IA ont ete utilises pour aider a structurer le projet, reformuler certaines explications et organiser le code. Les resultats, limites et choix techniques restent verifies dans le notebook.

## Limites

Les faux visages du dataset viennent d'un seul generateur, StyleGAN. Le modele peut donc apprendre des traces propres a ce generateur et ne garantit pas une generalisation parfaite a d'autres types de deepfakes.
