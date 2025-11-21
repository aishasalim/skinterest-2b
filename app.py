# app_pipeline.py
# 3-model skin pipeline
# 1) Kaggle model  -> 10 diseases (center-cropped)
# 2) Merged-12 Ensemble (Xception ⊕ ResNet50V2) -> 12 canonical classes
# 3) Merged-12 EfficientNetB0 + Custom CNN -> 12 canonical classes
# + Lighting/Undertone ResNet model
#
# NOTE: Research prototype only – NOT medical advice.

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet50  
import requests   
import os

WEIGHTS_DIR = "weights_local"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# -----------------------------
# 0) Paths & image sizes
# -----------------------------

# Google Drive *file IDs* (NOT folder URL)
GDRIVE_FILES = { 
    "kaggle_classifier_model.keras": "1QyK-mwjCw30g1gINNnn5srQiz4kEPZt3", 
    "merged12_xception_finetuned.keras": "1CL06tnH4X8ogJ1NXxx2wrRh1avX_ac9s", 
    "merged12_resnet50v2_finetuned.keras":"12Q9jz4Nkg_sgDdBLo3WptC1M-aV7R4Z8", 
    "merged12_effnet_custom_finetuned.keras": "1SvThqmZpa34bEG0byX8bIEtL3TOkqwDU", 
    "undertone_classifier_model.keras": "19g4Yxw8UbmUkrzqDrIj8RhjHzba7-ht5", 
    "kaggle_label_classes.npy": "1niHVRZDezHX0ssxt2eFdoYKfn1UyOHbS", 
    "merged12_label_classes.npy": "115FixjBHgp9LylFwMlGWSumVEAwdielc", 
}

# --- Paths (edit if your files are elsewhere) ---
KAGGLE_MODEL_PATH      = "weights_local/kaggle_classifier_model.keras"
KAGGLE_LABELS_PATH     = "weights_local/kaggle_label_classes.npy"

# Merged-12 ensemble parts + labels
MERGED12_XCEPTION_PATH = "weights_local/merged12_xception_finetuned.keras"
MERGED12_RESNET_PATH   = "weights_local/merged12_resnet50v2_finetuned.keras"

# EfficientNet+CustomCNN 12-class model (finetuned)
MERGED12_EFFNET_CUSTOM_PATH = "weights_local/merged12_effnet_custom_finetuned.keras"

# Lighting + undertone model
UNDERTONE_MODEL_PATH   = "weights_local/undertone_classifier_model.keras"

# Canonical 12-class labels
MERGED12_LABELS_PATH   = "weights_local/merged12_label_classes.npy"

# Labels for lighting / undertone heads
LIGHTING_LABELS  = ["good", "poor_contrast", "overexposed"]  # adjust to your training labels
UNDERTONE_LABELS = ["cool", "neutral", "warm"]

# --- Image sizes ---
KAGGLE_IMG_SIZE  = (224, 224)         # Kaggle 10
MERGED12_X_IMG   = (299, 299)         # Xception branch
MERGED12_R_IMG   = (224, 224)         # ResNet50V2 branch + EffNetB0 branch


def _gdrive_download_from_id(file_id: str, dest_path: str, chunk_size: int = 1 << 20):
    """
    Minimal Google Drive downloader.
    Works if the file is shared as 'Anyone with the link'.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

def ensure_weights_present():
    """
    Ensure all weight files exist in weights_local/.
    Download from Google Drive if missing.
    """
    for fname, file_id in GDRIVE_FILES.items():
        local_path = os.path.join(WEIGHTS_DIR, fname)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            continue  # already here

        print(f"[weights] Downloading {fname}...")
        _gdrive_download_from_id(file_id, local_path)
        print(f"[weights] Saved to {local_path}")

# Run this before loading any models
ensure_weights_present()


# -----------------------------
# 1) Custom layers for Kaggle model
# -----------------------------

class ColorCalibration(layers.Layer):
    def __init__(self, reg_lambda=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.reg_lambda = reg_lambda
        self.I = tf.constant(np.eye(3, dtype=np.float32))

    def build(self, input_shape):
        self.M = self.add_weight(
            name="ccm",
            shape=(3, 3),
            dtype="float32",
            initializer=tf.keras.initializers.Identity(),
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(3,),
            dtype="float32",
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        x32 = tf.cast(x, tf.float32)
        x_corr = tf.einsum("bhwc,cd->bhwd", x32, self.M) + self.b
        x_corr = tf.clip_by_value(x_corr, 0.0, 1.0)
        self.add_loss(self.reg_lambda * tf.reduce_sum(tf.square(self.M - self.I)))
        return tf.cast(x_corr, x.dtype)


class ResNetV2Preprocess(layers.Layer):
    def call(self, x):
        z = tf.cast(x, tf.float32)
        return tf.keras.applications.resnet_v2.preprocess_input(z * 255.0)

# -----------------------------
# 2) Load models & label arrays
# -----------------------------

@st.cache_resource(show_spinner="Loading models…")
def load_models_and_labels():
    # --- Base Kaggle model ---
    kaggle_model = load_model(
        KAGGLE_MODEL_PATH,
        custom_objects={
            "ColorCalibration": ColorCalibration,
            "ResNetV2Preprocess": ResNetV2Preprocess,
        },
    )

    # --- Merged-12 ensemble parts (Xception + ResNet50V2) ---
    merged12_x = load_model(MERGED12_XCEPTION_PATH)
    merged12_r = load_model(MERGED12_RESNET_PATH)

    # --- Merged-12 EfficientNetB0 + Custom CNN ensemble ---
    merged12_effnet_custom = load_model(MERGED12_EFFNET_CUSTOM_PATH)

    # --- Lighting + undertone ResNet model (2-head) ---
    undertone_model = load_model(
        UNDERTONE_MODEL_PATH,
        custom_objects={"ResNetV2Preprocess": ResNetV2Preprocess},
    )

    # --- labels (exact training order) ---
    kaggle_classes   = np.load(KAGGLE_LABELS_PATH,    allow_pickle=True)
    merged12_labels  = np.load(MERGED12_LABELS_PATH,  allow_pickle=True)

    # Optional keywords (not used yet)
    KAGGLE_SCIN_KEYWORDS = {
        "1. Eczema 1677": ["eczema"],
        "3. Atopic Dermatitis - 1.25k": ["atopic", "dermatitis", "eczema"],
        "7. Psoriasis pictures Lichen Planus and related diseases - 2k": ["psoriasis", "lichen"],
        "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": ["tinea", "ringworm", "candidiasis", "fungal"],
        "10. Warts Molluscum and other Viral Infections - 2103": ["wart", "verruca", "molluscum", "viral"],
    }

    return (
        kaggle_model,
        merged12_x, merged12_r,
        merged12_effnet_custom,
        undertone_model,
        kaggle_classes, merged12_labels,
        KAGGLE_SCIN_KEYWORDS,
    )

(
    kaggle_model,
    merged12_x, merged12_r,
    merged12_effnet_custom,
    undertone_model,
    kaggle_classes, merged12_labels,
    KAGGLE_SCIN_KEYWORDS,
) = load_models_and_labels()

# -----------------------------
# 3) Small helpers
# -----------------------------

def confidence_label(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.50:
        return "medium"
    return "low"

def center_crop_and_resize(pil_img: Image.Image, crop_scale: float, target_size=(224, 224)):
    """
    Center crop by crop_scale of the shorter side, then resize.
    Returns (arr_float01, cropped_pil).
    """
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    side = int(min(w, h) * crop_scale)
    cx, cy = w // 2, h // 2
    x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x2, y2 = min(x1 + side, w), min(y1 + side, h)
    crop = pil_img.crop((x1, y1, x2, y2))
    resized = crop.resize(target_size)
    arr = np.array(resized).astype("float32") / 255.0
    return arr, crop

# -----------------------------
# 4) Inference wrappers (Kaggle)
# -----------------------------

def run_kaggle(pil_img: Image.Image, crop_scale: float = 0.7, top_k: int = 3):
    arr, crop = center_crop_and_resize(pil_img, crop_scale, target_size=KAGGLE_IMG_SIZE)
    batch = np.expand_dims(arr, axis=0)
    probs = kaggle_model.predict(batch, verbose=0)[0]
    top_ids = probs.argsort()[-top_k:][::-1]
    results = [
        {"id": int(i), "label": str(kaggle_classes[i]), "prob": float(probs[i])}
        for i in top_ids
    ]
    return results, probs, crop, arr

# -----------------------------
# 5) Merged-12 Ensemble helpers (Xception + ResNet)
# -----------------------------

def preprocess_for_merged12_x(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(MERGED12_X_IMG)
    arr = np.array(img).astype("float32") / 255.0
    return tf.keras.applications.xception.preprocess_input(arr * 255.0)

def preprocess_for_merged12_r(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(MERGED12_R_IMG)
    arr = np.array(img).astype("float32") / 255.0
    return tf.keras.applications.resnet_v2.preprocess_input(arr * 255.0)

def run_merged12_ensemble(pil_img: Image.Image, top_k: int = 3, method: str = "geometric"):
    """
    Xception ⊕ ResNet50V2 ensemble.
    Returns: (top list, final_probs, xception_arr_for_cam)
    """
    arr_x = preprocess_for_merged12_x(pil_img)
    arr_r = preprocess_for_merged12_r(pil_img)

    px = merged12_x.predict(np.expand_dims(arr_x, 0), verbose=0)[0]
    pr = merged12_r.predict(np.expand_dims(arr_r, 0), verbose=0)[0]

    if method == "geometric":
        eps = 1e-8
        p = np.exp((np.log(px + eps) + np.log(pr + eps)) / 2.0)
        p = p / p.sum()
    else:
        p = 0.5 * (px + pr)

    top_ids = p.argsort()[-top_k:][::-1]
    results = [
        {"id": int(i), "label": str(merged12_labels[i]), "prob": float(p[i])}
        for i in top_ids
    ]
    return results, p, arr_x

# -----------------------------
# 6) Merged-12 EfficientNet+CustomCNN helpers
# -----------------------------

def preprocess_for_effnet_custom(pil_img: Image.Image):
    """
    EfficientNetB0+CustomCNN model was trained on [0,1] RGB 224x224.
    It contains its own EfficientNet preprocessing internally.
    """
    img = pil_img.convert("RGB").resize(MERGED12_R_IMG)
    arr = np.array(img).astype("float32") / 255.0  # [0,1]
    return arr

def run_effnet_custom(pil_img: Image.Image, top_k: int = 3):
    arr = preprocess_for_effnet_custom(pil_img)
    probs = merged12_effnet_custom.predict(np.expand_dims(arr, 0), verbose=0)[0]
    top_ids = probs.argsort()[-top_k:][::-1]
    results = [
        {"id": int(i), "label": str(merged12_labels[i]), "prob": float(probs[i])}
        for i in top_ids
    ]
    return results, probs, arr

# -----------------------------
# 7) Lighting + undertone model helpers
# -----------------------------

def preprocess_for_undertone_model(pil_img: Image.Image):
    """
    Resizes to 224x224 and applies ResNet50 preprocessing.
    If, in your training code, you already used preprocess_input in tf.data,
    comment out the line with resnet50.preprocess_input.
    """
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = resnet50.preprocess_input(arr)
    return arr

def run_lighting_undertone_model(pil_img: Image.Image):
    """
    Returns:
      lighting_label, lighting_prob,
      undertone_label, undertone_prob
    """
    arr = preprocess_for_undertone_model(pil_img)
    batch = np.expand_dims(arr, 0)

    # model has two heads: [lighting_probs, undertone_probs]
    lighting_probs, undertone_probs = undertone_model.predict(batch, verbose=0)

    lighting_probs  = lighting_probs[0]
    undertone_probs = undertone_probs[0]

    light_idx   = int(np.argmax(lighting_probs))
    undert_idx  = int(np.argmax(undertone_probs))

    light_label   = LIGHTING_LABELS[light_idx]
    light_prob    = float(lighting_probs[light_idx])

    undert_label  = UNDERTONE_LABELS[undert_idx]
    undert_prob   = float(undertone_probs[undert_idx])

    return light_label, light_prob, undert_label, undert_prob

# -----------------------------
# 8) Streamlit UI (3 models + lighting/undertone model)
# -----------------------------

st.set_page_config(
    page_title="Skin classifier (Kaggle • Merged-12)",
    layout="wide"
)
st.title("Skin photo classifier — Kaggle • Merged-12 Ensembles")
st.caption("Prototype only — **not** medical advice.")

uploaded = st.file_uploader(
    "Upload a phone-style skin photo (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    pil_img.thumbnail((1600, 1600))  # avoid massive images slowing things down

    st.image(
        pil_img,
        caption="Uploaded image",
        width=420,
        use_container_width=False,
    )

    with st.spinner("Running models…"):
        # Kaggle 10-class head
        kaggle_top, kaggle_probs, crop_pil, kaggle_arr = run_kaggle(
            pil_img, crop_scale=0.7, top_k=3
        )
        # Xception ⊕ ResNet50V2 12-class ensemble
        ens_top, ens_probs, ens_arr_x = run_merged12_ensemble(
            pil_img, top_k=3, method="geometric"
        )
        # EfficientNetB0 + Custom CNN 12-class model
        eff_top, eff_probs, eff_arr = run_effnet_custom(
            pil_img, top_k=3
        )
        # Lighting + undertone model
        light_label, light_prob, undert_label, undert_prob = (
            run_lighting_undertone_model(pil_img)
        )

    kaggle_main = kaggle_top[0]
    kaggle_conf = confidence_label(kaggle_main["prob"])

    ens_main    = ens_top[0]
    ens_conf    = confidence_label(ens_main["prob"])

    eff_main    = eff_top[0]
    eff_conf    = confidence_label(eff_main["prob"])

    col1, col2, col3 = st.columns(3)

    # ---------- Kaggle ----------
    with col1:
        st.subheader("Kaggle Skin Diseases Dataset based model (10 diseases)")
        st.caption("Center crop sent to Kaggle:")
        st.markdown(
            f"**Top:** {kaggle_main['label']} "
            f"({kaggle_main['prob']:.2f}, {kaggle_conf})"
        )
        st.markdown("**Top-3:**")
        for r in kaggle_top:
            st.write(f"- {r['label']}: {r['prob']:.2f}")

    # ---------- Merged-12 Ensemble (Xception ⊕ ResNet) ----------
    with col2:
        st.subheader("Ensemble of Xception + ResNet (12 categories)")
        st.markdown(
            f"**Top:** {ens_main['label']} "
            f"({ens_main['prob']:.2f}, {ens_conf})"
        )
        st.markdown("**Top-3:**")
        for r in ens_top:
            st.write(f"- {r['label']}: {r['prob']:.2f}")

    # ---------- Merged-12 EfficientNetB0 + Custom CNN ----------
    with col3:
        st.subheader("Ensemble of EffNetB0 + Custom CNN (12 categories)")
        st.markdown(
            f"**Top:** {eff_main['label']} "
            f"({eff_main['prob']:.2f}, {eff_conf})"
        )
        st.markdown("**Top-3:**")
        for r in eff_top:
            st.write(f"- {r['label']}: {r['prob']:.2f}")

    # ---------- Photo quality + undertone MODEL ----------
    st.markdown("---")
    st.subheader("Photo quality, lighting, and undertone model")

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown("**Is this photo good enough for AI?**")
        st.write(f"- Lighting condition: **{light_label}** (p = {light_prob:.2f})")
        if light_label == "good" and light_prob >= 0.70:
            st.success("This photo is likely good enough for the disease classifiers.")
        else:
            st.warning(
                "Lighting may reduce model reliability. "
                "Try taking a new photo with more even lighting."
            )

    with col_q2:
        st.markdown("**Predicted skin undertone**")
        st.write(f"- Undertone: **{undert_label}** (p = {undert_prob:.2f})")
        st.caption(
            "Undertone model trained on labeled images (cool / neutral / warm); "
            "used to check that our disease models behave consistently across tones."
        )

    st.markdown(
        "> This tool cannot confirm or rule out any diagnosis. It is a research "
        "prototype and must not be used as a substitute for a professional medical evaluation."
    )

else:
    st.markdown(
        "⬆️ Upload a photo above to see what **three** models predict."
    )

st.markdown("---")
st.caption("TensorFlow + Streamlit • Prototype only • Not for diagnosis.")
