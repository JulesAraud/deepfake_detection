import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------
# CONFIGURATION & CONSTANTS
# ----------------------------------------------------
IMG_SIZE = 128
MODEL_PATH = Path("artifacts/deepfake_mobilenetv2.keras")
DATASET_BASE = Path(r"C:\Users\Hamza\Downloads\Projet Deep learning\deepfake_detection-main")
KAGGLE_DATASET_BASE = Path(r"C:\Users\Hamza\.cache\kagglehub\datasets\xhlulu\140k-real-and-fake-faces\versions\2\real_vs_fake\real-vs-fake\test")

# French prize money tree ladder
PRIZE_LADDER_FR = [
    (10, "1 000 000 €", True),
    (9, "500 000 €", False),
    (8, "250 000 €", False),
    (7, "125 000 €", False),
    (6, "64 000 €", False),
    (5, "32 000 €", True),
    (4, "16 000 €", False),
    (3, "8 000 €", False),
    (2, "4 000 €", False),
    (1, "1 000 €", True),
    (0, "0 €", False)
]

# Set Streamlit Page Config
st.set_page_config(
    page_title="Le Millionnaire du Deepfake",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Custom CSS for Millionaire TV Studio Aesthetics
st.markdown("""
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at center, #111a36 0%, #050812 100%) !important;
        font-family: 'Outfit', sans-serif !important;
        color: #e2e8f0;
    }

    /* Main Title Styling */
    .game-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ffd700, #ff8c00, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        text-shadow: 0px 4px 20px rgba(255, 215, 0, 0.3);
    }
    .game-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Glassmorphic Panel/Card Design */
    div.st-key-rules_box,
    div.st-key-play_box,
    div.st-key-ladder_box,
    div.st-key-scanning_box,
    div.st-key-game_over_box {
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 170, 0, 0.35) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), 0 0 10px rgba(255, 170, 0, 0.1) !important;
    }

    div.st-key-ladder_box {
        padding-top: 12px !important;
        padding-bottom: 32px !important;
    }

    div.st-key-rules_box {
        padding-top: 12px !important;
    }

    /* Prevent double borders and hide native Streamlit border */
    div.st-key-rules_box > div,
    div.st-key-play_box > div,
    div.st-key-ladder_box > div,
    div.st-key-scanning_box > div,
    div.st-key-game_over_box > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Responsive Equal-Height Cards */
    div[data-testid="stHorizontalBlock"] > div > div > div[data-testid="stVerticalBlock"] {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    div.st-key-play_box,
    div.st-key-ladder_box {
        flex-grow: 1 !important;
    }

    /* Money Ladder Panel */
    .ladder-container {
        background: transparent;
        border: none;
        padding: 0;
        box-shadow: none;
    }
    
    .ladder-row {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 6px 12px;
        margin: 4.5px 0;
        border-radius: 12px;
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 600;
        text-align: center;
        border: 1px solid rgba(255, 215, 0, 0.25);
        background: rgba(15, 23, 42, 0.4);
        transition: all 0.2s ease;
    }
    
    .ladder-row.milestone {
        color: #38bdf8;
        border: 1.5px solid #38bdf8;
    }
    
    .ladder-row.active {
        background: linear-gradient(90deg, #ff8c00, #ffd700);
        color: #050812 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.8);
        transform: scale(1.03);
    }
    
    .ladder-row.past {
        color: #64748b;
        text-decoration: line-through;
        border-color: rgba(100, 116, 139, 0.15);
        background: transparent;
    }

    /* Option Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%) !important;
        color: #ffd700 !important;
        border: 2px solid #ffaa00 !important;
        border-radius: 30px !important;
        padding: 12px 30px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        width: 100% !important;
        box-shadow: 0 0 12px rgba(255, 215, 0, 0.15) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #ff8c00 0%, #ffd700 100%) !important;
        color: #050812 !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 25px rgba(255, 215, 0, 0.7) !important;
        transform: translateY(-2px) !important;
    }

    /* Chat bubble for friend prediction */
    .chat-bubble {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #38bdf8;
        border-radius: 15px;
        padding: 15px;
        margin-top: 15px;
        color: #f1f5f9;
        font-style: italic;
        position: relative;
    }
    
    .chat-bubble::before {
        content: '';
        position: absolute;
        top: -10px;
        left: 30px;
        border-width: 0 10px 10px;
        border-style: solid;
        border-color: transparent transparent rgba(30, 41, 59, 0.8);
        display: block;
        width: 0;
    }

    /* Score badges */
    .result-badge {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-win {
        background-color: rgba(34, 197, 94, 0.2);
        border: 2px solid #22c55e;
        color: #4ade80;
    }
    .result-lose {
        background-color: rgba(239, 68, 68, 0.2);
        border: 2px solid #ef4444;
        color: #f87171;
    }
    .result-info {
        background-color: rgba(56, 189, 248, 0.2);
        border: 2px solid #38bdf8;
        color: #7dd3fc;
    }

    /* Clickable Lifeline styling for native Streamlit buttons styled as horizontal ovals */
    .st-key-joker_50, .st-key-joker_friend, .st-key-joker_audience {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }

    .st-key-joker_50 button, .st-key-joker_friend button, .st-key-joker_audience button {
        border-radius: 25px !important;
        padding: 10px 20px !important;
        font-size: 1rem !important;
        font-weight: bold !important;
        width: 100% !important;
        max-width: 150px !important;
        height: 45px !important;
        text-align: center !important;
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%) !important;
        color: #ffd700 !important;
        border: 2px solid #ffaa00 !important;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.2) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }

    .st-key-joker_50 button:hover, .st-key-joker_friend button:hover, .st-key-joker_audience button:hover {
        background: linear-gradient(135deg, #ff8c00 0%, #ffd700 100%) !important;
        color: #050812 !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    .st-key-joker_50 button:disabled, .st-key-joker_friend button:disabled, .st-key-joker_audience button:disabled {
        filter: grayscale(100%) brightness(0.4) !important;
        border-color: #64748b !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Green buttons (Yes) */
    div.st-key-btn_human_std button,
    div.st-key-btn_yes_1 button,
    div.st-key-btn_yes_2 button,
    div.st-key-btn_yes_3 button,
    div.st-key-btn_yes_4 button,
    div.st-key-btn_yes_5 button,
    div[class*="st-key-btn_yes_"] button {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
        color: #ffffff !important;
        border: 2px solid #22c55e !important;
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.2) !important;
    }
    div.st-key-btn_human_std button:hover,
    div.st-key-btn_yes_1 button:hover,
    div.st-key-btn_yes_2 button:hover,
    div.st-key-btn_yes_3 button:hover,
    div.st-key-btn_yes_4 button:hover,
    div.st-key-btn_yes_5 button:hover,
    div[class*="st-key-btn_yes_"] button:hover {
        background: linear-gradient(135deg, #22c55e 0%, #4ade80 100%) !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.7) !important;
    }

    /* Red buttons (No) */
    div.st-key-btn_ai_std button,
    div.st-key-btn_no_1 button,
    div.st-key-btn_no_2 button,
    div.st-key-btn_no_3 button,
    div.st-key-btn_no_4 button,
    div.st-key-btn_no_5 button,
    div[class*="st-key-btn_no_"] button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: #ffffff !important;
        border: 2px solid #ef4444 !important;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.2) !important;
    }
    div.st-key-btn_ai_std button:hover,
    div.st-key-btn_no_1 button:hover,
    div.st-key-btn_no_2 button:hover,
    div.st-key-btn_no_3 button:hover,
    div.st-key-btn_no_4 button:hover,
    div.st-key-btn_no_5 button:hover,
    div[class*="st-key-btn_no_"] button:hover {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%) !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.7) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# DATASET LOADING (CACHED)
# ----------------------------------------------------
@st.cache_data
def load_dataset_image_paths():
    """Scan and cache the paths to all test images from the dataset directories."""
    paths = [KAGGLE_DATASET_BASE, DATASET_BASE / "test", Path("test")]
    detected_path = None
    
    for p in paths:
        if p.exists() and (p / "real").exists() and (p / "fake").exists():
            detected_path = p
            break
            
    if detected_path is None:
        return None
        
    real_dir = detected_path / "real"
    fake_dir = detected_path / "fake"
    
    real_images = [str(real_dir / f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_images = [str(fake_dir / f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    return {"REAL": real_images, "FAKE": fake_images}

# ----------------------------------------------------
# MODEL LOADING (CACHED RESOURCE)
# ----------------------------------------------------
@st.cache_resource
def get_strong_model():
    """Load and cache the pre-trained MobileNetV2 keras model."""
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def get_weak_model():
    """Instantiate and compile the CNN model architecture from models.py."""
    from models import build_cnn
    return build_cnn(IMG_SIZE)

def prepare_inference_image(image_path):
    """Load, convert to RGB, resize, and convert to numpy batch array for Keras."""
    img = Image.open(image_path).convert("RGB")
    resized = img.resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(resized, dtype="float32")
    return img, array[None, ...]

# ----------------------------------------------------
# GAME STATE CONFIGURATION
# ----------------------------------------------------
if "score" not in st.session_state:
    st.session_state.score = 0
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "has_won" not in st.session_state:
    st.session_state.has_won = False
if "current_image" not in st.session_state:
    st.session_state.current_image = "50-50_2018.webp" if not st.runtime.exists() else None
if "current_label" not in st.session_state:
    st.session_state.current_label = "REAL" if not st.runtime.exists() else None
if "stage" not in st.session_state:
    st.session_state.stage = "start"  # "start", "playing", "scanning", "result"
if "jokers_used" not in st.session_state:
    st.session_state.jokers_used = {
        "phone_a_friend": False,
        "ask_the_audience": False
    }
if "jokers_active" not in st.session_state:
    st.session_state.jokers_active = {
        "fifty_fifty": False
    }
if "fifty_fifty_presses" not in st.session_state:
    st.session_state.fifty_fifty_presses = 0
if "fifty_fifty_broken" not in st.session_state:
    st.session_state.fifty_fifty_broken = False
if "fifty_fifty_just_broke" not in st.session_state:
    st.session_state.fifty_fifty_just_broke = False
if "friend_prediction" not in st.session_state:
    st.session_state.friend_prediction = None
if "audience_votes" not in st.session_state:
    st.session_state.audience_votes = None
if "user_guess" not in st.session_state:
    st.session_state.user_guess = None
if "ai_label" not in st.session_state:
    st.session_state.ai_label = None
if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = 0.0
if "points_gained" not in st.session_state:
    st.session_state.points_gained = 0
if "current_visage_caption" not in st.session_state:
    st.session_state.current_visage_caption = "Analysez attentivement ce visage..."

# ----------------------------------------------------
# GAME STATE TRANSITION ACTIONS
# ----------------------------------------------------
def start_game():
    st.session_state.score = 0
    st.session_state.game_over = False
    st.session_state.has_won = False
    st.session_state.fifty_fifty_broken = False
    st.session_state.fifty_fifty_just_broke = False
    st.session_state.fifty_fifty_presses = 0
    st.session_state.jokers_used = {
        "phone_a_friend": False,
        "ask_the_audience": False
    }
    st.session_state.jokers_active = {
        "fifty_fifty": False
    }
    setup_next_round()

def setup_next_round():
    images = load_dataset_image_paths()
    if not images or len(images["REAL"]) == 0 or len(images["FAKE"]) == 0:
        st.error("Erreur : Impossible de charger les images du dataset. Veuillez vérifier le dossier de test.")
        st.stop()
        
    # Pick randomly between REAL and FAKE
    label = random.choice(["REAL", "FAKE"])
    img_path = random.choice(images[label])
    
    st.session_state.current_image = img_path
    st.session_state.current_label = label
    st.session_state.stage = "playing"
    st.session_state.jokers_active["fifty_fifty"] = False
    st.session_state.fifty_fifty_just_broke = False
        
    st.session_state.friend_prediction = None
    st.session_state.audience_votes = None
    st.session_state.user_guess = None
    st.session_state.ai_label = None
    st.session_state.ai_confidence = 0.0
    
    # Choose a random visage synonym for the active round
    synonyms = [
        ("ce", "faciès"),
        ("ce", "portrait"),
        ("ce", "minois"),
        ("cette", "frimousse"),
        ("cette", "tête"),
        ("cette", "bobine"),
        ("cette", "trombine"),
        ("cette", "physionomie")
    ]
    det, syn = random.choice(synonyms)
    st.session_state.current_visage_caption = f"Analysez attentivement {det} {syn}..."

def trigger_guess(guess_val):
    st.session_state.user_guess = guess_val
    st.session_state.stage = "scanning"

# ----------------------------------------------------
# MAIN UI LAYOUT
# ----------------------------------------------------
st.markdown("<h1 class='game-title'>🎭 QUI VEUT ÊTRE MEILLEUR QUE L'IA ? 💰</h1>", unsafe_allow_html=True)
st.markdown("<p class='game-subtitle'>Pouvez-vous distinguer un visage humain d'une création de l'IA (StyleGAN) et battre le modèle MobileNetV2 ? Tentez de gagner 1 000 000 € !</p>", unsafe_allow_html=True)

# Safety check for model and dataset before playing
if not MODEL_PATH.exists():
    st.warning("⚠️ Modèle entraîné `deepfake_mobilenetv2.keras` introuvable dans `artifacts/`. Exécutez d'abord le notebook ou copiez le fichier.")
    st.stop()

images_dict = load_dataset_image_paths()
if not images_dict:
    st.warning("⚠️ Les images du dataset n'ont pas pu être chargées. Veuillez vous assurer que le dataset `140k-real-and-fake-faces` est présent dans le dossier de cache kagglehub.")
    st.stop()

# Initialize if in start state
if st.session_state.stage == "start":
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        with st.container(key="rules_box", border=True):
            st.markdown("""
            <div class='glass-anchor'></div>
            <div style='text-align: center;'>
                <h2 style='margin-top: 0;'>Prêt à tester votre supériorité humaine ?</h2>
                <p style='color: #94a3b8;'>Battez la machine, marquez des points et tentez de remporter le grand prix de 1 000 000 €.</p>
                <h4 style='color: #ffd700; margin-top: 20px;'>Règles du jeu :</h4>
                <ul style='text-align: left; max-width: 500px; margin: 0 auto; line-height: 1.7; color: #cbd5e1;'>
                    <li>On vous présente un visage. Devinez s'il est <b>Humain</b> ou <b>Généré par IA</b>.</li>
                    <li>Notre IA (MobileNetV2) analyse l'image de son côté en même temps.</li>
                    <li><b>Vous avez tout les deux juste :</b> Vous gagnez <b>1 point</b>.</li>
                    <li><b>Vous avez tout les deux faux :</b> Vous gagnez <b>0 point</b> (le jeu continue).</li>
                    <li><b>Vous correct, l'IA fausse :</b> Vous gagnez <b>2 points</b> !</li>
                    <li><b>Vous faux, l'IA correcte :</b> <b>PERDU !</b></li>
                    <li>Atteignez <b>1 000 000 d'euros</b> pour remporter la victoire !</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🚀 Commencer la partie", key="start_btn"):
            start_game()
            st.rerun()
    st.stop()

# Set up column layout for playing and result stages
col_left, col_right = st.columns([2.2, 1])

# ----------------------------------------------------
# RIGHT COLUMN: THE CASH LADDER (FRENCH)
# ----------------------------------------------------
with col_right:
    st.markdown("<h3 class='game-title' style='text-align: center; color: #ffd700;'>💰 ÉCHELLE DES GAINS</h3>", unsafe_allow_html=True)
    with st.container(key="ladder_box", border=True):
        st.markdown("<div class='glass-anchor'></div>", unsafe_allow_html=True)
        
        for level_score, price, is_milestone in PRIZE_LADDER_FR:
            row_class = "ladder-row"
            if is_milestone:
                row_class += " milestone"
            
            # Determine status relative to current score
            if st.session_state.score == level_score:
                row_class += " active"
                prefix = "▶ "
            elif st.session_state.score > level_score:
                row_class += " past"
                prefix = "✓ "
            else:
                prefix = ""
                
            st.markdown(f"""
            <div class='{row_class}'>
                <span><b>{prefix}{price}</b></span>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------
# LEFT COLUMN: THE GAMEPLAY AREA
# ----------------------------------------------------
with col_left:
    # ------------------------------------------------
    # STAGE: GAME OVER / WON SCREEN
    # ------------------------------------------------
    if st.session_state.game_over:
        with st.container(key="game_over_box", border=True):
            st.markdown("<div class='glass-anchor'></div>", unsafe_allow_html=True)
            if st.session_state.has_won:
                st.balloons()
                st.markdown("""
                <div class='result-badge result-win'>
                    🎉 FÉLICITATIONS ! VOUS AVEZ GAGNÉ LE MILLION ! 🎉
                </div>
                """, unsafe_allow_html=True)
                st.subheader("Vous avez atteint les 10 points requis et surpassé notre IA !")
            else:
                st.markdown("""
                <div class='result-badge result-lose'>
                    💥 GAME OVER ! L'IA VOUS A BATTU ! 💥
                </div>
                """, unsafe_allow_html=True)
                st.write("Votre mauvaise réponse a mis fin à la partie.")

            col_over_img, col_over_info = st.columns([1, 1.2])
            
            with col_over_img:
                # Displays the face image
                img, _ = prepare_inference_image(st.session_state.current_image)
                caption_text = "Votre syndrome de l'imposteur n'est PAS justifié !" if st.session_state.has_won else "Votre syndrome de l'imposteur est justifié..."
                st.image(img, width="stretch", caption=caption_text)
                
            with col_over_info:
                # Displays the summary on the right of the image
                st.write("#### 📝 Résultats de la dernière manche")
                user_lbl = "Humain" if st.session_state.user_guess == "REAL" else "Généré par IA"
                ai_lbl = "Humain" if st.session_state.ai_label == "REAL" else "Généré par IA"
                true_lbl = "Humain" if st.session_state.current_label == "REAL" else "Généré par IA"
                
                st.markdown(f"""
                - 👤 **Votre réponse :** {user_lbl}
                - 🤖 **Réponse de l'IA MobileNetV2 :** {ai_lbl} (Confiance : {st.session_state.ai_confidence:.1%})
                - 🌟 **La vérité :** **{true_lbl}**
                """)
                
                st.write("---")
                # Calculate milestone fallback cash
                final_cash = "0 €"
                if st.session_state.score >= 5:
                    final_cash = "32 000 €"
                elif st.session_state.score >= 1:
                    final_cash = "1 000 €"
                    
                st.markdown(f"### 💰 Gains finaux : **{final_cash if not st.session_state.has_won else '1 000 000 €'}**")
                
                if st.button("🔄 Recommencer une partie", key="game_over_restart"):
                    start_game()
                    st.rerun()
        st.stop()

    # ------------------------------------------------
    # STAGE: SCANNING IN GLASS PANEL (AI RUNNING)
    # ------------------------------------------------
    if st.session_state.stage == "scanning":
        st.write("### 🔍 Analyse du visage en cours...")
        with st.container(key="scanning_box", border=True):
            st.markdown("<div class='glass-anchor'></div>", unsafe_allow_html=True)
            
            col_img_scan, col_progress_scan = st.columns([1, 1.2])
            with col_img_scan:
                img, _ = prepare_inference_image(st.session_state.current_image)
                st.image(img, width="stretch", caption="Scan de la structure faciale...")
                
            with col_progress_scan:
                st.write("#### ⚙️ Analyse des modèles d'IA")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Animate progress bar visually
                for p in range(0, 100, 2):
                    time.sleep(0.015)
                    progress_bar.progress(p + 2)
                    if p < 40:
                        status_text.text("Extraction des caractéristiques locales (yeux, peau, contrastes)...")
                    elif p < 80:
                        status_text.text("Exécution du réseau de neurones MobileNetV2...")
                    else:
                        status_text.text("Finalisation des calculs de probabilité...")
                
                # Compute actual predictions
                model = get_strong_model()
                if model is not None:
                    _, batch = prepare_inference_image(st.session_state.current_image)
                    prob_real = float(model.predict(batch, verbose=0).ravel()[0])
                    ai_lbl = "REAL" if prob_real >= 0.5 else "FAKE"
                    ai_conf = prob_real if ai_lbl == "REAL" else 1.0 - prob_real
                    
                    st.session_state.ai_label = ai_lbl
                    st.session_state.ai_confidence = ai_conf
                    
                    # Check correctness
                    guess_val = st.session_state.user_guess
                    true_label = st.session_state.current_label
                    user_correct = (guess_val == true_label)
                    ai_correct = (ai_lbl == true_label)
                    
                    if user_correct and ai_correct:
                        st.session_state.score += 1
                        st.session_state.points_gained = 1
                    elif not user_correct and not ai_correct:
                        st.session_state.points_gained = 0
                    elif user_correct and not ai_correct:
                        st.session_state.score += 2
                        st.session_state.points_gained = 2
                    else:
                        st.session_state.points_gained = -1
                        st.session_state.game_over = True
                        st.session_state.has_won = False
                        
                # Check win condition
                if st.session_state.score >= 10:
                    st.session_state.score = 10
                    st.session_state.game_over = True
                    st.session_state.has_won = True
                    
                st.session_state.stage = "result"
                st.rerun()
        st.stop()

    # ------------------------------------------------
    # ACTIVE ROUND PLAYING & RESULT LAYOUT
    # ------------------------------------------------
    st.markdown("<h3 class='game-title' style='text-align: left; color: #ffd700;'>🎭 LE VISAGE DE LA MANCHE</h3>", unsafe_allow_html=True)
    with st.container(key="play_box", border=True):
        st.markdown("<div class='glass-anchor'></div>", unsafe_allow_html=True)
        
        col_img, col_info = st.columns([1, 1.2])
        
        with col_img:
            # Load and render the current active face
            img, _ = prepare_inference_image(st.session_state.current_image)
            st.image(img, width="stretch", caption=st.session_state.current_visage_caption)
            
        with col_info:
            # Show lifelines (Jokers) if we are in the playing phase
            if st.session_state.stage == "playing":
                st.write("#### 🛡️ JOKERS")
                j_col1, j_col2, j_col3 = st.columns(3)
                
                with j_col1:
                    # 50:50 - Reusable joke (adds buttons until breaking on 4th press)
                    is_50_disabled = st.session_state.fifty_fifty_broken
                    if st.button("50:50", key="joker_50", disabled=is_50_disabled):
                        st.session_state.fifty_fifty_presses += 1
                        st.session_state.jokers_active["fifty_fifty"] = True
                        if st.session_state.fifty_fifty_presses >= 5:
                            st.session_state.fifty_fifty_broken = True
                            st.session_state.fifty_fifty_just_broke = True
                            st.session_state.jokers_active["fifty_fifty"] = False
                            st.toast("Le joker 50:50 a surchauffé et s'est CASSÉ !", icon="💥")
                        else:
                            st.toast(f"Joker 50:50 activé ! (Appui {st.session_state.fifty_fifty_presses}/3)", icon="🎭")
                        st.rerun()
                        
                with j_col2:
                    # Phone a Friend - Weaker AI (One-time) - Loads ACTUAL CNN MODEL!
                    paf_used = st.session_state.jokers_used["phone_a_friend"]
                    if st.button("Appel Ami", key="joker_friend", disabled=paf_used):
                        with st.spinner("Voici le message du seul proche qui a bien voulu vous aider (un modèle CNN que nous avons entraîné) :"):
                            # Load and execute the ACTUAL CNN model!
                            weak_model = get_weak_model()
                            _, batch_cnn = prepare_inference_image(st.session_state.current_image)
                            prob_real_cnn = float(weak_model.predict(batch_cnn, verbose=0).ravel()[0])
                            friend_lbl = "REAL" if prob_real_cnn >= 0.5 else "FAKE"
                            friend_conf = prob_real_cnn if friend_lbl == "REAL" else 1.0 - prob_real_cnn
                            
                            st.session_state.friend_prediction = {
                                "label": friend_lbl,
                                "confidence": friend_conf
                            }
                        st.session_state.jokers_used["phone_a_friend"] = True
                        st.toast("Votre ami le modèle CNN a répondu !", icon="📞")
                        st.rerun()
                        
                with j_col3:
                    # Ask the Audience - One-time
                    ata_used = st.session_state.jokers_used["ask_the_audience"]
                    if st.button("Avis Public", key="joker_audience", disabled=ata_used):
                        # Simulate 100 people with 62% accuracy
                        true_val = st.session_state.current_label
                        correct_v = sum(np.random.choice([1, 0], size=100, p=[0.62, 0.38]))
                        incorrect_v = 100 - correct_v
                        
                        if true_val == "REAL":
                            votes = {"Humain": correct_v, "IA": incorrect_v}
                        else:
                            votes = {"IA": correct_v, "Humain": incorrect_v}
                            
                        st.session_state.audience_votes = votes
                        st.session_state.jokers_used["ask_the_audience"] = True
                        st.toast("Le public s'est prononcé !", icon="👥")
                        st.rerun()

                # Render active joker responses
                if st.session_state.friend_prediction:
                    friend = st.session_state.friend_prediction
                    label_fr = "Humain" if friend["label"] == "REAL" else "Généré par IA"
                    st.markdown(f"""
                    <div class='chat-bubble'>
                        📞 <b>Voici le message du seul proche qui a bien voulu vous aider (un modèle CNN que nous avons entraîné) :</b><br/>
                        "Salut ! J'ai analysé l'image avec mon réseau de neurones. Je suis sûr à <b>{friend['confidence']:.1%}</b> que c'est un <b>{label_fr}</b>. J'espère que ça t'aidera !"
                    </div>
                    """, unsafe_allow_html=True)
                    
                if st.session_state.audience_votes:
                    votes = st.session_state.audience_votes
                    st.write("📊 **Sondage du public :**")
                    df_votes = pd.DataFrame({
                        "Choix": list(votes.keys()),
                        "Votes (%)": list(votes.values())
                    }).set_index("Choix")
                    st.bar_chart(df_votes, height=180)

            # ------------------------------------------------
            # INTERACTION AREA: GUESS BUTTONS OR ROUND RESULTS
            # ------------------------------------------------
            st.write("---")
            
            if st.session_state.stage == "playing":
                st.write("#### 🔮 Est-ce un visage humain réel ?")
                
                if st.session_state.fifty_fifty_just_broke:
                    st.warning("💥 MAIS ÇA VA PAS DE CLIQUER COMME ÇA, VOUS AVEZ CASSÉ LE BOUTON !")
                elif 1 <= st.session_state.fifty_fifty_presses <= 4:
                    st.info(f"Ah, le 50/50 a buggé, ça a ajouté des options au lieu d'en enlever. Le joker a été remboursé pour la {st.session_state.fifty_fifty_presses}e fois.")
                    
                opt_col1, opt_col2 = st.columns(2)
                with opt_col1:
                    if st.button("Oui", key="btn_human_std"):
                        trigger_guess("REAL")
                        st.rerun()
                with opt_col2:
                    if st.button("Non", key="btn_ai_std"):
                        trigger_guess("FAKE")
                        st.rerun()

                # Additional choices based on fifty_fifty_presses
                presses = st.session_state.fifty_fifty_presses
                if presses > 0:
                    real_opts = [
                        ("Absolument", "btn_yes_2"),
                        ("Tout à fait", "btn_yes_3"),
                        ("C'est certain", "btn_yes_4"),
                        ("Sans aucun doute", "btn_yes_5")
                    ]
                    fake_opts = [
                        ("Pas du tout", "btn_no_2"),
                        ("Impossible", "btn_no_3"),
                        ("Que nenni", "btn_no_4"),
                        ("Aucunement", "btn_no_5")
                    ]
                    
                    num_sets = min(presses, 4)
                    
                    col_real, col_fake = st.columns(2)
                    with col_real:
                        for i in range(num_sets):
                            label, key = real_opts[i]
                            if st.button(label, key=key):
                                trigger_guess("REAL")
                                st.rerun()
                    with col_fake:
                        for i in range(num_sets):
                            label, key = fake_opts[i]
                            if st.button(label, key=key):
                                trigger_guess("FAKE")
                                st.rerun()
                            
            elif st.session_state.stage == "result":
                st.write("#### 📝 Résultats de la manche")
                
                # Show verdicts
                user_lbl = "Humain" if st.session_state.user_guess == "REAL" else "Généré par IA"
                ai_lbl = "Humain" if st.session_state.ai_label == "REAL" else "Généré par IA"
                true_lbl = "Humain" if st.session_state.current_label == "REAL" else "Généré par IA"
                
                st.markdown(f"""
                - 👤 **Votre réponse :** {user_lbl}
                - 🤖 **Réponse de l'IA MobileNetV2 :** {ai_lbl} (Confiance : {st.session_state.ai_confidence:.1%})
                - 🌟 **Vérité :** **{true_lbl}**
                """)
                
                # Feedback on points gained
                gained = st.session_state.points_gained
                if gained == 2:
                    st.markdown("""
                    <div class='result-badge result-win'>
                        🏆 Vous avez battu l'IA ! +2 points ! 🏆
                    </div>
                    """, unsafe_allow_html=True)
                elif gained == 1:
                    st.markdown("""
                    <div class='result-badge result-info'>
                        🤝 Consensus ! Les deux corrects. +1 point.
                    </div>
                    """, unsafe_allow_html=True)
                elif gained == 0:
                    st.markdown("""
                    <div class='result-badge result-info' style='background-color: rgba(100, 116, 139, 0.2); border: 2px solid #64748b; color: #cbd5e1;'>
                        🥱 Les deux faux ! Aucun point. Le jeu continue.
                    </div>
                    """, unsafe_allow_html=True)
                    
                if st.button("⏭ Question suivante", key="next_round_btn"):
                    setup_next_round()
                    st.rerun()
