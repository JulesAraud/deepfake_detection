"""Fonctions metier pour la v1 Data Science / baseline ML.

Le notebook garde l'orchestration et les commentaires d'analyse, tandis que les
operations reutilisables restent dans ce module pour faciliter la maintenance.
"""

import io
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.utils.class_weight import compute_class_weight

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - repli simple pour les environnements minimaux
    def tqdm(iterable=None, total=None, desc=""):
        seq = list(iterable) if iterable is not None else []
        total = total or len(seq)
        step = max(1, total // 20)
        for i, x in enumerate(seq, 1):
            if i % step == 0 or i == total:
                print(f"\r{desc} {100 * i // total:3d}%", end="", flush=True)
            yield x
        print()


IMG_EXT = (".jpg", ".jpeg", ".png")


def find_split_dirs(root):
    """Repere les dossiers train/val/test contenant des sous-dossiers real/fake."""
    aliases = {
        "train": "train",
        "training": "train",
        "valid": "val",
        "val": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    splits = {}
    for dirpath, dirnames, _ in os.walk(root):
        sub = {d.lower() for d in dirnames}
        if {"real", "fake"}.issubset(sub):
            base = os.path.basename(dirpath).lower()
            if base in aliases:
                splits[aliases[base]] = dirpath
    return splits


def class_dir(split_dir, cls):
    """Retourne le sous-dossier d'une classe, independamment de la casse."""
    for d in os.listdir(split_dir):
        if d.lower() == cls:
            return os.path.join(split_dir, d)
    raise FileNotFoundError(f"Sous-dossier '{cls}' introuvable dans {split_dir}")


def count_images(directory):
    """Compte les images supportees dans un dossier."""
    return sum(1 for f in os.listdir(directory) if f.lower().endswith(IMG_EXT))


def load_split(split_dir, n_per_class, img_size=128, seed=42):
    """Charge jusqu'a n_per_class images par classe depuis un split local."""
    x_values, y_values = [], []
    for cls_name, label in [("real", 1), ("fake", 0)]:
        cls_dir = class_dir(split_dir, cls_name)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXT)]
        random.Random(seed).shuffle(files)
        files = files[:n_per_class]
        for filename in tqdm(files, total=len(files), desc=f"{os.path.basename(split_dir)}/{cls_name}"):
            try:
                img = Image.open(os.path.join(cls_dir, filename)).convert("RGB")
                x_values.append(np.asarray(img.resize((img_size, img_size)), dtype="uint8"))
                y_values.append(label)
            except Exception:
                pass
    return np.array(x_values), np.array(y_values)


def image_feature_table(x_values, y_values):
    """Construit une table de descripteurs simples par image."""
    x_float = x_values.astype("float32")
    return pd.DataFrame({
        "intensite_moy": x_float.mean(axis=(1, 2, 3)),
        "contraste": x_float.std(axis=(1, 2, 3)),
        "R": x_float[..., 0].mean(axis=(1, 2)),
        "G": x_float[..., 1].mean(axis=(1, 2)),
        "B": x_float[..., 2].mean(axis=(1, 2)),
        "label": np.where(y_values == 1, "REAL", "FAKE"),
    })


def show_grid(x_values, y_values, label_int, label_str, n=5, plt_module=None):
    """Affiche une ligne d'exemples pour une classe donnee."""
    if plt_module is None:
        import matplotlib.pyplot as plt_module

    idx = np.where(y_values == label_int)[0][:n]
    fig, axes = plt_module.subplots(1, n, figsize=(3 * n, 3))
    fig.suptitle(f"Exemples - {label_str}", fontsize=14, y=1.05)
    for ax, i in zip(axes, idx):
        ax.imshow(x_values[i])
        ax.axis("off")
    plt_module.tight_layout()
    plt_module.show()


def _to_img(arr):
    return Image.fromarray(arr.astype("uint8"))


def _to_arr(img, img_size):
    return np.asarray(img.resize((img_size, img_size)), dtype="uint8")


def _mirror(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))


def _contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))


def _color(img):
    return ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.5))


def _rotate(img):
    return img.rotate(random.uniform(-12, 12), resample=Image.BILINEAR)


def _jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(15, 45))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _noise(img):
    arr = np.asarray(img, dtype="int16")
    noise = np.random.normal(0, random.uniform(8, 22), arr.shape)
    return _to_img(np.clip(arr + noise, 0, 255))


def _blur(img):
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.6, 1.6)))


TRANSFORMS = [_mirror, _brightness, _contrast, _color, _rotate, _jpeg, _noise, _blur]


def augment_training_set(x_values, y_values, aug_per_image, img_size=128):
    """Ajoute des variantes augmentees par image et renvoie un train melange."""
    aug_x, aug_y = [], []
    k = min(aug_per_image, len(TRANSFORMS))
    for i in tqdm(range(len(x_values)), total=len(x_values), desc="Augmentation"):
        base = _to_img(x_values[i])
        for transform in random.sample(TRANSFORMS, k):
            try:
                aug_x.append(_to_arr(transform(base), img_size))
                aug_y.append(y_values[i])
            except Exception:
                pass
    x_out = np.concatenate([x_values, np.array(aug_x, dtype="uint8")], axis=0)
    y_out = np.concatenate([y_values, np.array(aug_y)], axis=0)
    order = np.random.permutation(len(x_out))
    return x_out[order], y_out[order]


def compute_balanced_class_weight(y_values):
    """Calcule les poids de classe equilibres au format attendu par scikit-learn/Keras."""
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_values)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def to_flat_features(x_values, size=32):
    """Reduit chaque image et l'aplatit en vecteur normalise dans [0, 1]."""
    out = np.empty((len(x_values), size * size * 3), dtype="float32")
    for i in tqdm(range(len(x_values)), total=len(x_values), desc=f"Features {size}x{size}"):
        img = Image.fromarray(x_values[i]).resize((size, size))
        out[i] = np.asarray(img, dtype="float32").reshape(-1) / 255.0
    return out
