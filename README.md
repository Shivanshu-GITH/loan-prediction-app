# Loan Approval Prediction — Decision Tree Classifier

**Author:** Shivanshu Tiwari
**Roll Number:** 2400270130173
**Program:** AI & DS Internship and Training Program

A practical machine learning project demonstrating **data preprocessing, model
training, evaluation, and deployment** of a loan prediction model using a
Decision Tree Classifier.

---

## Project Structure

| File | Purpose |
|---|---|
| `loan_data.csv` | Dataset (45,000 loan records) |
| `train_model.py` | Preprocessing, training, evaluation — saves the model |
| `app.py` | Deployment: Streamlit web application |
| `loan_model.pkl` | Trained Decision Tree model |
| `scaler.pkl` | Fitted StandardScaler |
| `confusion_matrix.png` | Evaluation: confusion matrix plot |
| `feature_importance.png` | Evaluation: feature importance plot |
| `decision_tree.png` | Visualization of the trained tree (top 3 levels) |
| `requirements.txt` | Python dependencies |

## Dataset

4 input features and 1 binary target:

- **Age** — applicant age in years
- **Income** — annual income
- **LoanAmount** — requested loan amount
- **CreditScore** — credit score (390–850)
- **Approved** — target: 1 = approved, 0 = rejected (imbalanced: ~22% approved)

## Workflow

### 1. Data Preprocessing
- Checked for missing values (none found)
- Removed 1 duplicate row
- Reviewed statistics for outliers
- Stratified 80/20 train-test split (preserves class ratio)
- Feature scaling with `StandardScaler` (fit on training set only)

### 2. Model Training
- `DecisionTreeClassifier` tuned with **GridSearchCV** (5-fold cross-validation, F1 scoring)
- Grid covered `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, and `class_weight`
- Best parameters: `class_weight='balanced'`, `criterion='entropy'`, `max_depth=10`, `min_samples_leaf=20`, `min_samples_split=50`
- `class_weight='balanced'` compensates for the imbalanced target

### 3. Evaluation (held-out test set, 9,000 samples)

| Metric | Score |
|---|---|
| Accuracy | 0.77 |
| Precision (Approved) | 0.49 |
| Recall (Approved) | 0.71 |
| F1 Score (Approved) | 0.58 |

Most important features: **Income (65%)** and **LoanAmount (30%)**.

### 4. Deployment
Interactive Streamlit web app: enter applicant details, get an
Approved/Rejected prediction with a confidence score.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates loan_model.pkl, scaler.pkl, and plots)
python train_model.py

# 3. Launch the web application
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.
