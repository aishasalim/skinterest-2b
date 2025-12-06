# Skinterest-2B — Multi-Modal Skin Condition Classification (Break Through Tech AI × Skinterest Tech)

> **Disclaimer:** This project is a research prototype for educational purposes only and **is not medical advice**. Do not use it to diagnose or treat any condition.

---

## 👥 Team Members

| Name                    | GitHub Handle        | Role / Contribution                                                            |
| ----------------------- | -------------------- | ------------------------------------------------------------------------------ |
| **Aisha Salimgereyeva** | `@aishasalim`        | **ResNet-152V2** pipeline; training/eval scripts; Streamlit demo; docs         |
| **Wanying Xu**          | `@OliviaCoding`      | **MobileNetV3** baselines; EDA & visuals; documentation                     |
| **Ayleen Jimenez**      | `@ayleenjim`         | **EfficientNet-B7** experiments; error analysis                                |
| **Hoang Do**            | `@hoangggdo`         | **MaxViT** experiments; augmentation/regularization ablations                  |
| **Alexis Amadi**        | `@aalexis123`        | **ResNet50** baseline; optimization & speed profiling                          |
| **Susan Qu**            | `@susan-q`           | **ResNet50** experiments; lighting and skin tone analysis                      |
| **Nandini**             | `@albatrosspreacher` | Reviewer (Write access); PM support; meeting notes                             |

---

## 🎯 Project Highlights

- Developed a **multimodal CNN** by using various deep learning models(ResNet-152V2, MobileNetV3, etc) in order to process and classify a wide range of skin conditions, such as Eczema/Atopic Dermatitis, Lupus, and Pigmentation disorders.
- Achived a testing accuracy of **over 80%**, demonstrating that this model is suitable for image for AI analysis and directly contributing to Skinterest's goal of fostering inclusivity within the dermatology field.
- Implemented **(1) lighting harshness** and **(2) skin undertones** analysis of the data so that the model is able to classify images with different lighting and color tones.
- Created a **Streamlit demo** for qualitative testing and stakeholder feedback.
  
---
# 👩🏽‍💻 Setup & Installation

## Prerequisites
- **Python:** 3.9–3.11  
- **(Recommended)** GPU runtime for training (Colab T4/A100 or local NVIDIA)  
- **Kaggle API credentials** (for Kaggle dataset download)

---

## Repo layout + outputs

```text
.
├── app.py
├── requirements.txt
├── configs/
│   └── resnet152v2_baseline.yaml
├── experiments/                 # auto-created: one folder per run (small text+png only)
│   └── <run_name>/
│       ├── report.json
│       ├── metrics.csv          # optional training history
│       ├── weights.txt          # how to fetch large .keras/.h5 from cloud storage
│       └── figures/             # optional PNGs (cm, grad-cam, slice metrics, etc.)
├── notebooks/
│   └── aisha/
│       ├── 01_eda_scins_kaggle.ipynb
│       ├── 02_training_multitask_resnet152v2.ipynb
│       └── 03_error_analysis_fairness.ipynb
├── scripts/
│   ├── prepare_kaggle_meta.py   # builds meta CSV with labels + lighting + ITA
│   └── train_abc.py             # trains Phase A/B/C from a config
└── src/
    └── ... (data / layers / models / training / eval / utils)
```

### Outputs contract (important)
- Training + evaluation artifacts should go to `experiments/<run_name>/`
- Large model binaries (`.keras`, `.h5`) should **NOT** be committed; store externally and put the download command/link in `experiments/<run_name>/weights.txt`

---

## Quickstart (choose one)

### A) One-click (Google Colab)

1. Open: `notebooks/aisha/02_training_multitask_resnet152v2.ipynb` in Colab.
2. Install dependencies (top cell):

   ```bash
   !pip -q install kaggle==1.6.17 tensorflow==2.20.0 tensorflow-addons==0.23.0 opencv-python==4.10.0.84
   ```

3. Kaggle credentials:
   - Upload `kaggle.json` to `/root/.kaggle/kaggle.json`
   - Set permissions:

     ```bash
     !chmod 600 /root/.kaggle/kaggle.json
     ```

4. Run the notebook cells to:
   - Download dataset(s)
   - Generate metadata (lighting + ITA)
   - Train Phase A/B/C
   - Write artifacts into `experiments/<run_name>/`

**Expected outputs (example):**
- `experiments/<run_name>/report.json`
- `experiments/<run_name>/metrics.csv` *(optional)*
- `experiments/<run_name>/weights.txt` *(download instructions for `.keras/.h5`)*
- `experiments/<run_name>/figures/*.png` *(optional)*

> Note: If you currently save figures to `docs/figures/`, consider redirecting them into `experiments/<run_name>/figures/` so every run is self-contained.

---

### B) Local (macOS, Apple Silicon)

> Tested on Python **3.9–3.11**.

Create venv + install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

If you use TensorFlow on Apple Silicon (recommended pins):

```bash
pip install tensorflow-macos==2.16.1 tensorflow-metal==1.1.0 keras==3.3.3
```

Train from config (recommended reproducible entrypoint):

```bash
python scripts/train_abc.py --config configs/resnet152v2_baseline.yaml --run_name resnet152v2_baseline_seed42 --seed 42
```

Run the demo:

```bash
streamlit run app.py
```

> If you see model deserialization issues, make sure custom layer class names (e.g., `ColorCalibration`, `ResNetV2Preprocess`) are imported and **match exactly** what was used during training.

---

## Data access & expected layout

### SCIN (Google Research)
- Follow the official SCIN access instructions (see References).
- Recommended: place SCIN under `data/scin/` *(not committed).*

### Kaggle dataset (e.g., “Skin Diseases Image Dataset”)
Download via Kaggle CLI (example):

```bash
kaggle datasets download -d <kaggle-dataset-slug> -p data/kaggle --unzip
```

### Metadata generation (lighting + ITA)
Create metadata CSV used by `tf.data` pipelines:

```bash
python scripts/prepare_kaggle_meta.py --input_dir data/kaggle --out_csv data/meta/kaggle_meta.csv
```

> Ensure your training config / notebook points to the generated `meta.csv` path.

---

## 🏗️ Project Overview
**About the Program:**
* The Break Through Tech AI program is an experiential learning opportunity that allows students to gain **hands-on technical experience** in the competitive AI/ML industry. This program connected us to our project's challenge advisors from Skinterest Tech and through this program, we learned how to work as a team, **preprocess and clean data, train AI models**, and **fine-tune parameters** for better results. These learned skills set us apart from other applicants when applying to jobs.
**Our Goals, Objectives, and the Company:**
* Skinterest Tech is a **skincare startup** whose goal is to **diversify skincare** and help patients find the right product based on their skin quality, texture, tone, and more. The objective our AI Studio project with Skinterest Tech is to develop a **reliable and usable machine learning model** that detects poor lighting and classifies images of **common dermatologic conditions** across **diverse** skin tones, to be used for clinical review.
**Business Relevance:**
* The problem that our machine learning model solves is significant because training data used by today's dermatology industry is often heavily skewed toward lighter skin tones, **neglecting representation for people with darker complexions**. This issue can impact skin condition diagnosis and product awareness. Skin condition diagnosis on people with deeper skin tones may be incorrectly classified and patients may be recommended unnecessary, or even harmful products. Our model specifically addresses this by **accounting for various skin tones and image lighting** of their pictures.

---

## 📊 Data Exploration

**Datasets**

- **SCIN**: large dermatology corpus emphasizing representation across skin tones. Used for primary training and evaluation splits.
- **Kaggle: Skin Diseases Image Dataset (ismailpromus)**: used for stress testing and additional qualitative validation.

**Preprocessing & assumptions**

- **Lighting features** (HSV/V/contrast/specular) generate a binary label (_well-lit vs poor lighting_) via conservative thresholds.
- **Skin-tone bucket** is computed from **ITA** (LAB space): `light / medium / dark` (median over a simple skin mask).
- **Center-crop** (default `0.8`) + resize to 224×224 to reduce background bias and normalize scale.
- **Label encoding** for 10-class diagnosis head; consistent class order is stored in `demo/class_index.json`.

**EDA Insights**

- Class imbalance is significant (e.g., **nevi** >> **eczema/psoriasis**).
- Lighting quality and tone distribution are skewed—necessitating **class-weights** and **fairness slices**.
- Basic augmentations (flip/rotate/zoom + color jitter) help reduce overfitting without harming calibration.

---

## 🧠 Model Development

**Architecture (multitask)**

- **Input**: 224×224×3 float [0,1] → **ColorCalibration (CCM)** → **ResNetV2Preprocess** → **Backbone** (e.g., ResNet-152V2, ResNet50, EfficientNet-B7, MobileNetV2/V3, MaxViT) →

  - **Head 1 (lighting)**: Dense(128, ReLU) → Dropout → Dense(1, Sigmoid)
  - **Head 2 (diagnosis)**: Dense(128, ReLU) → Dropout → Dense(10, Softmax)

**Training schedule**

- **Phase A (heads only):** backbone + CCM frozen; LR=1e-3 (AdamW).
- **Phase B (CCM only):** unfreeze CCM; LR=5e-4.
- **Phase C (partial backbone):** unfreeze top 40%; LR=5e-5.
- **Phase D (optional):** full unfreeze at tiny LR (1e-5 → 5e-6) with strong regularization + early stop.

**Imbalance handling**

- **Default:** **class-weights** (preferred).
- **Ablation:** capped oversampling by `(diagnosis × tone_bucket)` to check fairness trade-offs.

**Loss / Metrics**

- Lighting: Binary Cross-Entropy (+ label smoothing 0.05), **Accuracy**, **AUC**.
- Diagnosis: Sparse Categorical Cross-Entropy, **Top-1 Accuracy**, **Macro-Avg Accuracy**, **Per-Class Accuracy**.
- Fairness: accuracy by `tone_bucket`.

---

## 🧩 Code Highlights

- `src/model_multitask.py`

  - `ColorCalibration`: learnable 3×3 color transform + bias with L2 prior to identity.
  - `build_multitask(backbone=..., drop_rate=...)`: returns Keras model with two heads.

- `src/data_prep.py`

  - ITA computation + simple skin mask; metadata CSV; stratified splits; `tf.data` pipelines with center-crop and augmentations.

- `src/train.py`

  - Implements Phases A/B/C; class-weights; callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau).

- `src/eval.py`

  - Confusion matrix, per-class tables, fairness slices, and Grad-CAM utilities.

- `app.py`

  - Streamlit demo; loads `.keras` with custom layers; top-k predictions; optional Grad-CAM.

---

## 📈 Results & Key Findings

> Numbers below are from a representative **ResNet-152V2 + CCM** run (Phases A/B/C), single seed 42.

**Test set (diagnosis head)**

- **Overall accuracy:** ~**0.80**
- **Macro-avg accuracy:** ~**0.75**
- Notable strong classes: **BCC** (~0.94), **Nevi** (~0.92)
- Weaker classes: **Eczema / Psoriasis** (0.55–0.65); confusions often symmetric.

**Lighting head**

- **Accuracy:** ~**0.86**; **AUC:** high-0.88/0.89 range.

**Fairness slice (diagnosis by tone_bucket)**

- **light:** ~0.82
- **medium:** ~0.71
- **dark:** ~0.86 _(very small n; wide CI)_

**Figures (saved under `docs/figures/`)**

- `confusion_matrix_diagnosis.png`
- `pr_curves_lighting.png`
- `gradcam_examples/…`
- `fairness_bars_tone_bucket.png`

**Takeaways**

- **CCM** + **center-crop** reduce color/illumination drift.
- **Class-weights** outperform heavy oversampling for generalization.
- **Full unfreeze (Phase D)** risks overfitting unless combined with stronger regularization and early stopping.


---

## 💬 Discussion & Reflection

**Summary:**
Our model can be highly susceptable to overfitting while training because of domain shifts and high variance in minority classes of the data due to class imbalance and subtle visual traits. It also sometimes struggles to differentiate Eczema and Psoriasis, likely due to visual overlap and labeling noise factors. Introducing external images outside of data set may also effect model accuracy; Grad-CAM helps audit failure modes.

**What worked**

- Multitask formulation stabilized training and improved robustness to lighting.
- Lightweight CCM provided consistent gains with negligible compute cost.
- Clear phase schedule (A/B/C) improved convergence and prevented catastrophic forgetting.

**What didn’t**

- Phase D full unfreeze frequently **overfit** (val↓ while train↑).
- Eczema/Psoriasis remain challenging—visual overlap + labeling noise likely factors.
- External images (distribution shift) can degrade accuracy; Grad-CAM helps audit failure modes.

**Why**

- Class imbalance + subtle visual traits → higher variance in minority classes.
- Domain shift (camera, distance, compression) → emphasize data standardization at inference.

---

## 🚀 Next Steps

**Procedural:** With more time and resources, we may consider other project approach options such as having the team focus on one project step and one model at a time rather than all at once. This may encourage even more teamwork and learning opportunities.
Some additional data we may want to emplore are additional skin images from either the company or online to increase our dataset size fix imbalances in the data. We may even want to add images of normal skin of different lighting.

**Technical:** 
1. **Detector→Classifier**: Use YOLO lesion crops instead of global center-crop.
2. **Calibration**: Temperature scaling / Dirichlet calibration for better confidence estimates.
3. **Data curation**: Add cleaner eczema/psoriasis samples; augment under-represented tones.
4. **Fairness**: Track per-tone **ECE** and per-class **macro-F1**; evaluate with bootstrapped CIs.
5. **Light-quality feedback**: Turn lighting head into a user tip (“move closer”, “avoid flash glare”).
6. **Distillation**: Compress best model to MobileNetV3-Small for on-device triage.

---

## 📝 License

This project is licensed under the MIT License.

---

## 📄 References

* **SCIN: A New Resource for Representative Dermatology Images** (Dataset, Blog, and GitHub provided by Google Research).
* **Kaggle: Skin Diseases Image Dataset** by _ismailpromus_.
* **Deep Residual Learning for Image Recognition** (ResNet) - He et al. (2016).
* **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** - Tan & Le (2019).
* **MobileNetV2/V3: Searching for MobileNetV3** - Howard et al. (2019/2020).
* **MaxViT: Multi-Axis Attention for Vision Transformers** - Tu et al. (2022).
* **Skin Tone Representation in Dermatologist Social Media Accounts** - Paradkar & Kaffenberger (2022).

---

## 🙏 **Acknowledgements**

Many thanks to our Skinterest Tech challenge advisors Ashley Abid and Thandiwe-Kesi Robins and Break Through Tech Coach Nandini Proothi for guiding us and answering our questions!
