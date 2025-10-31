# app.py
# Streamlit demo for multitask Lighting + Undertone model
# - Loads a .keras file with custom layers
# - Performs TTA (test-time augmentation)
# - Uncertainty/abstain gate
# - Optional Grad-CAM visualization for undertone head

import os
import io
import warnings
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_pre

# -----------------------------
# 0) Config (EDIT THESE)
# -----------------------------
MODEL_PATH = "resnet152v2_lighting_undertone_full_model.keras"  # <-- your .keras file

# EXACT order you saw from `le_under.classes_` in training:
UNDERTONE_CLASSES = [
    "1. eczema 1677",
    "10. warts molluscum and other viral infections - 2103",
    "2. melanoma 15.75k",
    "3. atopic dermatitis - 1.25k",
    "4. basal cell carcinoma (bcc) 3323",
    "5. melanocytic nevi (nv) - 7970",
    "6. benign keratosis-like lesions (bkl) 2624",
    "7. psoriasis pictures lichen planus and related diseases - 2k",
    "8. seborrheic keratoses and other benign tumors - 1.8k",
    "9. tinea ringworm candidiasis and other fungal infections - 1.7k",
]

IMG_SIZE = (224, 224)
CENTER_CROP_FRAC = 0.6  # same as training

# Optional per-class thresholds (tune offline; fallback to global if missing)
CLASS_THRESH = {i: 0.65 for i in range(len(UNDERTONE_CLASSES))}

# -----------------------------
# 1) Custom layers (must match training names/behavior)
# -----------------------------
class ColorCalibration(tf.keras.layers.Layer):
    def __init__(self, reg_lambda=1e-4, **kw):
        # Do NOT force dtype here; let Keras pass what was saved.
        super().__init__(**kw)
        self.reg_lambda = reg_lambda
        self.I = tf.constant(np.eye(3, dtype=np.float32))

    def build(self, input_shape):
        self.M = self.add_weight(
            name="ccm", shape=(3, 3), dtype="float32",
            initializer=tf.keras.initializers.Identity(), trainable=True
        )
        self.b = self.add_weight(
            name="bias", shape=(3,), dtype="float32",
            initializer="zeros", trainable=True
        )

    def call(self, x):
        x32 = tf.cast(x, tf.float32)
        x_corr = tf.einsum("bhwc,cd->bhwd", x32, self.M) + self.b
        x_corr = tf.clip_by_value(x_corr, 0., 1.)
        self.add_loss(self.reg_lambda * tf.reduce_sum(tf.square(self.M - self.I)))
        return tf.cast(x_corr, x.dtype)

class ResNetV2Preprocess(tf.keras.layers.Layer):
    def call(self, x):
        z = tf.cast(x, tf.float32)
        return resnet_pre(z * 255.0)

# -----------------------------
# 2) Image helpers
# -----------------------------
def center_crop_and_resize(pil_img, frac=CENTER_CROP_FRAC, size=IMG_SIZE):
    w, h = pil_img.size
    new_w = int(w * frac)
    new_h = int(h * frac)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    right  = left + new_w
    bottom = top + new_h
    pil_cropped = pil_img.crop((left, top, right, bottom))
    pil_resized = pil_cropped.resize(size, Image.BILINEAR)
    arr = np.asarray(pil_resized).astype("float32") / 255.0  # [0,1]
    return arr  # HWC

def augment_np(img224):
    """Simple stochastic aug used for TTA: H-flip + 0/90/180/270 rotate."""
    x = img224.copy()
    if np.random.rand() < 0.5:
        x = np.flip(x, axis=1)  # horizontal flip
    k = np.random.choice([0, 1, 2, 3])
    x = np.rot90(x, k, axes=(0, 1))
    return x

# -----------------------------
# 3) Load model (cached)
# -----------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model_safely(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "ColorCalibration": ColorCalibration,
                "ResNetV2Preprocess": ResNetV2Preprocess,
            },
            compile=False,  # inference only
        )
    # Build once for fixed input shape
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype="float32")
    _ = model(dummy, training=False)
    return model

model = load_model_safely(MODEL_PATH)

# -----------------------------
# 4) Inference (single + TTA)
# -----------------------------
def predict_single(arr224):
    """arr224: (224,224,3) float32 [0,1]"""
    batch = np.expand_dims(arr224, 0)
    light_prob, undertone_probs = model.predict(batch, verbose=0)
    return float(light_prob[0, 0]), undertone_probs[0]

def predict_tta(arr224, n_aug=8):
    """Average predictions over TTA augs."""
    lights, conds = [], []
    for _ in range(max(1, n_aug)):
        x = augment_np(arr224)
        l, c = predict_single(x)
        lights.append(l); conds.append(c)
    return float(np.mean(lights)), np.mean(conds, axis=0)

# -----------------------------
# 5) Grad-CAM for undertone head (best-effort)
# -----------------------------
def _find_last_conv_4d(m):
    last = None
    for layer in m.layers:
        try:
            out = layer.output
            if len(out.shape) == 4:
                last = layer
        except Exception:
            pass
    return last

def gradcam_undertone(arr224, class_index):
    """
    Returns a heatmap (H,W) in [0,1] or None if we couldn't compute it.
    """
    try:
        last_conv = _find_last_conv_4d(model)
        if last_conv is None:
            return None

        # Build a sub-model from inputs to [last_conv_output, undertone_head]
        undertone_layer = model.get_layer("undertone_out")
        grad_model = tf.keras.Model(
            [model.inputs],
            [last_conv.output, undertone_layer.output]
        )

        x = np.expand_dims(arr224, 0)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x, training=False)
            target = preds[:, class_index]  # (1,)
        grads = tape.gradient(target, conv_out)  # (1,H,W,C)
        if grads is None:
            return None

        conv_out = conv_out[0].numpy()
        grads = grads[0].numpy()
        weights = np.mean(grads, axis=(0, 1))  # GAP over H,W
        cam = np.maximum(np.tensordot(conv_out, weights, axes=([2], [0])), 0)
        cam -= cam.min()
        cam = cam / (cam.max() + 1e-8)
        # resize to 224x224
        cam_img = Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BILINEAR)
        return np.asarray(cam_img).astype("float32") / 255.0
    except Exception:
        return None

def overlay_heatmap(pil_img, heatmap, alpha=0.35):
    """Overlay heatmap (H,W) on pil_img (224x224)."""
    if heatmap is None:
        return pil_img
    hm = np.uint8(255 * heatmap)
    hm_rgb = ImageOps.colorize(Image.fromarray(hm, mode="L"), black="black", white="red")
    hm_rgb = hm_rgb.convert("RGBA")
    base = pil_img.convert("RGBA")
    blended = Image.blend(base, hm_rgb, alpha=alpha)
    return blended.convert("RGB")

# -----------------------------
# 6) UI
# -----------------------------
st.set_page_config(page_title="Skin Condition + Lighting (Prototype)", layout="centered")
st.title("Skin Condition + Lighting Classifier (Prototype)")
st.caption("Research demo. Not medical advice.")

with st.sidebar:
    st.header("Inference options")
    tta = st.slider("TTA samples", 1, 16, 8, help="More TTA = more stable but slower")
    uncertain_global = st.slider("Uncertain threshold (global)", 0.50, 0.90, 0.65, 0.01)
    show_cam = st.checkbox("Show Grad-CAM (undertone)", value=True)

uploaded = st.file_uploader("Upload a close-up skin image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(pil, caption="Uploaded image", use_container_width=True)

    arr = center_crop_and_resize(pil, CENTER_CROP_FRAC, IMG_SIZE)
    light_score, probs = predict_tta(arr, n_aug=tta)

    # Lighting label
    lighting_label = "well-lit" if light_score >= 0.5 else "poor lighting"

    # Undertone top-k
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top_name = UNDERTONE_CLASSES[top_idx] if top_idx < len(UNDERTONE_CLASSES) else f"class {top_idx}"

    st.subheader("Results")
    st.write(f"**Lighting guess:** {lighting_label}  (score = {light_score:.3f})")
    st.write(f"**Predicted class:** {top_name}")
    st.write(f"**Confidence:** {top_conf*100:.1f}%")

    # Uncertainty/abstain gate (per-class, falling back to global)
    per_class_thresh = CLASS_THRESH.get(top_idx, uncertain_global)
    is_uncertain = top_conf < per_class_thresh or top_conf < uncertain_global
    if is_uncertain:
        st.warning("Model **uncertain**. Treat this as exploratory only.")

    # Top-3 list
    st.markdown("**Top-3 classes:**")
    top3 = np.argsort(-probs)[:3]
    for r, idx in enumerate(top3, 1):
        name_i = UNDERTONE_CLASSES[idx] if idx < len(UNDERTONE_CLASSES) else f"class {idx}"
        st.write(f"{r}. {name_i} ({probs[idx]*100:.1f}%)")

    # Grad-CAM
    if show_cam:
        hm = gradcam_undertone(arr, class_index=top_idx)
        pil224 = Image.fromarray(np.uint8(arr * 255))
        overlay = overlay_heatmap(pil224, hm, alpha=0.35)
        st.markdown("**Grad-CAM (undertone head, top class)**")
        st.image(overlay, use_container_width=False, width=320)

st.markdown("---")
st.caption("Built with TensorFlow + Streamlit. © Your Team • Not for diagnosis.")
