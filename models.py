"""Architectures des réseaux de neurones — module externe (Jalon 5 : modularité).

On sort la définition des modèles Keras du notebook pour :
- garder le notebook lisible (orchestration uniquement) ;
- pouvoir réutiliser / tester les architectures indépendamment ;
- versionner proprement l'architecture (label Git `dl`).

Fonctions :
- make_augmentation() : couche d'augmentation géométrique à la volée.
- build_cnn(...)       : CNN convolutif (modèle principal, DL).
- build_mlp(...)       : perceptron multicouche dense (comparaison d'architecture).
- build_transfer_mobilenet(...) : transfer learning MobileNetV2 (DL avancé).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


def _make_optimizer(name, lr):
    """Fabrique un optimiseur Keras à partir d'un nom (pour la recherche d'hyperparamètres)."""
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(lr, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(lr)
    raise ValueError(f"Optimiseur inconnu : {name}")


def make_augmentation():
    """Augmentation géométrique appliquée à la volée (active à l'entraînement uniquement)."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ], name="augmentation")


def build_cnn(img_size=128, lr=1e-3, optimizer="adam"):
    """CNN entraine de zero : 3 blocs (Conv2D -> BatchNorm -> MaxPooling) + tete dense.

    Version volontairement legere (16 -> 32 -> 64 filtres, tete dense de 64 neurones,
    ~28 000 parametres). On a divise par deux le nombre de filtres de chaque bloc par rapport
    a une config 32/64/128 : sur ce probleme les classes restent bien separables, et on gagne
    un facteur ~3-4 sur le temps de calcul par epoque, ce qui rend l'entrainement praticable sur
    CPU sans perte notable de performance.

    Idees d'architecture (Jalon 5) :
    - Conv2D : exploite la structure spatiale 2D via des filtres locaux a poids partages
      (invariance par translation, peu de parametres compare a une couche dense).
    - 3 blocs empiles : hierarchie de motifs (contours -> textures -> structures du visage).
    - GlobalAveragePooling2D : agrege spatialement et reduit fortement le nombre de parametres.

    Idees d'optimisation (Jalon 6), contre le vanishing/exploding gradient :
    - ReLU : gradient = 1 pour x>0, donc pas de saturation (contrairement a sigmoide/tanh).
    - BatchNormalization : recentre/normalise les activations -> gradients stables, convergence
      plus rapide, moins sensible a l'initialisation.
    - Initialisation He (defaut Keras pour ReLU) : variance adaptee a la profondeur.
    - Dropout : regularisation contre le surapprentissage.
    """
    model = models.Sequential([
        layers.Input((img_size, img_size, 3)),
        make_augmentation(),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="cnn_deepfake")
    model.compile(optimizer=_make_optimizer(optimizer, lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def build_mlp(img_size=128, lr=1e-3, optimizer="adam"):
    """MLP dense **sans convolution** : sert de comparaison d'architecture (Jalon 5/7).

    Limites attendues vs CNN :
    - `Flatten` détruit la structure spatiale 2D de l'image ;
    - aucune invariance par translation ;
    - énormément de paramètres (chaque pixel relié à chaque neurone).
    -> illustre *pourquoi* le CNN est mieux adapté aux images.
    """
    model = models.Sequential([
        layers.Input((img_size, img_size, 3)),
        layers.Rescaling(1. / 255),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="mlp_baseline")
    model.compile(optimizer=_make_optimizer(optimizer, lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def build_transfer_mobilenet(img_size=128, lr=1e-4, train_base=False):
    """Modele avance par transfer learning avec MobileNetV2 pre-entraine ImageNet.

    Pourquoi MobileNetV2 (Jalon 8) :
    - les premieres couches d'un modele ImageNet apprennent deja des motifs visuels generiques
      utiles aux visages : bords, textures, contrastes locaux ;
    - la base pre-entrainee donne une meilleure initialisation qu'un CNN entraine de zero ;
    - MobileNetV2 reste leger, donc compatible avec un entrainement notebook et un dashboard.

    `train_base=False` gele la base pour entrainer seulement la tete de classification. Passer
    `train_base=True` permet un fine-tuning ulterieur si le temps de calcul le permet.
    """
    inputs = layers.Input((img_size, img_size, 3), name="image")
    x = make_augmentation()(inputs)
    # Equivalent du preprocess_input MobileNetV2 : pixels [0, 255] -> [-1, 1].
    x = layers.Rescaling(1. / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = train_base

    x = base(x, training=train_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.35, name="dropout_head")(x)
    x = layers.Dense(128, activation="relu", name="dense_head")(x)
    x = layers.Dropout(0.25, name="dropout_out")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_real")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_deepfake")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
