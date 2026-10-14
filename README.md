# NFL Draft Prediction — ML Pipeline

An end-to-end machine learning pipeline built to predict whether a college athlete will be selected in the NFL Draft. This architecture achieves a **0.856 OOF ROC-AUC** score utilizing a highly optimized, multi-seed gradient boosting ensemble.

## 🛠️ Tech Stack
* **Languages & Core:** Python, NumPy, Pandas
* **Machine Learning:** XGBoost, LightGBM, Scikit-Learn

---

## 🧠 What I Built

### 1. Feature Engineering
* **Position-Relative Z-Scores:** Computed physical metrics relative to position-group medians (e.g., standardizing a 4.5s 40-yard dash differently for a Wide Receiver vs. an Offensive Tackle).
* **Custom Athletic Scores:** Formulated custom Speed Scores and an aggregate, weighted Combine Score across all 6 physical drills.
* **Temporal Signals:** Extracted year-level and position-by-year historical draft rates to capture changing scouting trends over time.
* **Missing Data As Signal:** Implemented position-median imputation tied to dedicated missingness flags, tracking instances where athletes strategically skipped drills.

### 2. Modeling & Validation
* **Leakage-Free Target Encoding:** Encoded school, position, player type, and interaction features safely out-of-fold across 5 stratified folds.
* **Multi-Seed Ensemble:** Trained XGBoost and LightGBM models across 5 distinct random seeds using 10-fold Cross-Validation, blending the averaged predictions to heavily minimize variance.
* **Semi-Supervised Pseudo-Labeling:** Extracted high-confidence test predictions (thresholded at >85% and <15%) to safely augment the training data for a secondary optimization pass.

---

## 📊 Results & Performance

| Metric | Score |
| :--- | :--- |
| **XGBoost OOF AUC** | ~0.843 |
| **LightGBM OOF AUC** | ~0.848 |
| **Final Blend OOF AUC** | **0.856** |

> **💡 Key Takeaway:** The single largest performance gains came from engineering position-relative context rather than raw model complexity. Giving the model the ability to evaluate an athlete's physical capability *relative to their specific position peers* drastically improved signal extraction.
