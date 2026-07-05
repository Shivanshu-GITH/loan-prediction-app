"""
Loan Approval Prediction - Deployment App (Flask)
Author : Shivanshu Tiwari (Roll No: 2400270130173)

An explainable loan approval predictor built on a Decision Tree classifier.
Besides the Approve/Reject decision, the app explains the reasoning: the exact
rules the tree applied, how the applicant compares to 45,000 real applicants,
and the data behind the confidence percentage.

Run with:  python app.py     (then open http://127.0.0.1:5000)
"""

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

FEATURES = ["Age", "Income", "LoanAmount", "CreditScore"]
PRETTY = {
    "Age": "Age",
    "Income": "Annual Income",
    "LoanAmount": "Loan Amount",
    "CreditScore": "Credit Score",
}
MONEY = {"Income", "LoanAmount"}

# ----------------------------------------------------------------------
# Load model, scaler and reference data once at startup
# ----------------------------------------------------------------------
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
ref = pd.read_csv("loan_data.csv").drop_duplicates()

DATASET_OVERVIEW = {
    "total": len(ref),
    "approved": int(ref["Approved"].sum()),
    "approval_rate": round(ref["Approved"].mean() * 100),
    "median_income": f"{ref['Income'].median():,.0f}",
    "median_credit": f"{ref['CreditScore'].median():.0f}",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def fmt(feature, value):
    if feature in MONEY:
        return f"Rs {value:,.0f}"
    return f"{value:,.0f}"


def percentile_of(feature, value):
    return float((ref[feature] < value).mean() * 100)


def decision_path_reasons(input_df):
    """Translate the applicant's path through the tree into plain-language rules."""
    x_scaled = scaler.transform(input_df)
    tree = model.tree_
    node_indicator = model.decision_path(x_scaled)
    leaf_id = model.apply(x_scaled)[0]
    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    reasons = []
    for node_id in node_index:
        if tree.children_left[node_id] == tree.children_right[node_id]:
            continue
        feat_idx = tree.feature[node_id]
        feat = FEATURES[feat_idx]
        threshold_scaled = tree.threshold[node_id]
        threshold = scaler.mean_[feat_idx] + threshold_scaled * scaler.scale_[feat_idx]
        applicant_value = input_df.iloc[0][feat]
        comparison = "at or below" if applicant_value <= threshold else "above"
        reasons.append(
            f"{PRETTY[feat]} of {fmt(feat, applicant_value)} is {comparison} "
            f"the threshold of {fmt(feat, threshold)}"
        )
    return reasons, leaf_id


def leaf_stats(leaf_id):
    tree = model.tree_
    n_samples = int(tree.n_node_samples[leaf_id])
    values = tree.value[leaf_id][0]
    approve_share = values[1] / values.sum()
    return n_samples, approve_share


def build_result(age, income, loan_amount, credit_score):
    input_df = pd.DataFrame(
        [[age, income, loan_amount, credit_score]], columns=FEATURES
    )
    x_scaled = scaler.transform(input_df)
    pred = int(model.predict(x_scaled)[0])
    prob_approve = float(model.predict_proba(x_scaled)[0][1])

    reasons, leaf_id = decision_path_reasons(input_df)
    n_similar, leaf_share = leaf_stats(leaf_id)

    # Percentiles for the comparison chart
    comparisons = [
        {
            "label": PRETTY[f],
            "pct": round(percentile_of(f, input_df.iloc[0][f])),
        }
        for f in FEATURES
    ]

    # Plain-language narrative
    income_pct = percentile_of("Income", income)
    credit_pct = percentile_of("CreditScore", credit_score)
    loan_pct = percentile_of("LoanAmount", loan_amount)
    bits = []
    if income_pct >= 60:
        bits.append(f"a strong income (higher than {income_pct:.0f}% of applicants)")
    elif income_pct <= 40:
        bits.append(f"a modest income (lower than {100 - income_pct:.0f}% of applicants)")
    if credit_pct >= 60:
        bits.append(f"a healthy credit score (top {100 - credit_pct:.0f}%)")
    elif credit_pct <= 40:
        bits.append(f"a below-average credit score (bottom {credit_pct:.0f}%)")
    if loan_pct >= 70:
        bits.append(f"a large requested amount (higher than {loan_pct:.0f}% of applicants)")

    verdict_word = "approve" if pred == 1 else "reject"
    summary = (
        f"The model chose to {verdict_word} this loan after applying "
        f"{len(reasons)} sequential decision rules to the applicant's profile."
    )
    if bits:
        summary += " Notable characteristics of this profile: " + ", ".join(bits) + "."

    # Feature importance (global)
    importances = sorted(
        [
            {"label": PRETTY[f], "pct": round(imp * 100)}
            for f, imp in zip(FEATURES, model.feature_importances_)
        ],
        key=lambda d: d["pct"],
        reverse=True,
    )

    return {
        "approved": pred == 1,
        "prob_approve": round(prob_approve * 100, 1),
        "confidence": round(max(prob_approve, 1 - prob_approve) * 100, 1),
        "reasons": reasons,
        "n_similar": f"{n_similar:,}",
        "leaf_share": round(leaf_share * 100, 1),
        "comparisons": comparisons,
        "summary": summary,
        "importances": importances,
        "inputs": {
            "Age": age,
            "Income": f"Rs {income:,.0f}",
            "LoanAmount": f"Rs {loan_amount:,.0f}",
            "CreditScore": credit_score,
        },
    }


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # Defaults keep the form populated on first load
    form = {"age": 28, "income": 60000, "loan_amount": 20000, "credit_score": 650}
    result = None
    error = None

    if request.method == "POST":
        try:
            form["age"] = int(request.form["age"])
            form["income"] = int(request.form["income"])
            form["loan_amount"] = int(request.form["loan_amount"])
            form["credit_score"] = int(request.form["credit_score"])
            result = build_result(
                form["age"], form["income"], form["loan_amount"], form["credit_score"]
            )
        except (ValueError, KeyError):
            error = "Please enter valid numeric values for all fields."

    return render_template(
        "index.html",
        form=form,
        result=result,
        error=error,
        overview=DATASET_OVERVIEW,
    )


if __name__ == "__main__":
    app.run(debug=True)
