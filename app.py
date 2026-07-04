"""
Loan Approval Prediction - Deployment App (Streamlit)
Author : Shivanshu Tiwari (Roll No: 2400270130173)

An explainable loan approval predictor built on a Decision Tree classifier.
Besides the Approve/Reject decision, the app explains WHY: the exact rules the
tree applied, how the applicant compares to 45,000 real applicants, and the
data behind the confidence percentage.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ["Age", "Income", "LoanAmount", "CreditScore"]
PRETTY = {
    "Age": "Age",
    "Income": "Annual Income",
    "LoanAmount": "Loan Amount",
    "CreditScore": "Credit Score",
}
MONEY = {"Income", "LoanAmount"}


# ----------------------------------------------------------------------
# Load artifacts and reference data
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


@st.cache_data
def load_reference():
    df = pd.read_csv("loan_data.csv").drop_duplicates()
    return df


model, scaler = load_artifacts()
ref = load_reference()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def fmt(feature, value):
    """Human-friendly formatting for a feature value."""
    if feature in MONEY:
        return f"₹{value:,.0f}"
    return f"{value:,.0f}"


def percentile_of(feature, value):
    """What fraction of real applicants have a lower value for this feature."""
    return float((ref[feature] < value).mean() * 100)


def decision_path_reasons(input_df):
    """
    Walk the exact path this applicant takes through the decision tree and
    translate each split (which is on scaled features) back into plain,
    original-unit statements.
    """
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
            continue  # leaf node, no rule

        feat_idx = tree.feature[node_id]
        feat = FEATURES[feat_idx]
        # Convert the scaled threshold back to the original unit
        threshold_scaled = tree.threshold[node_id]
        threshold = scaler.mean_[feat_idx] + threshold_scaled * scaler.scale_[feat_idx]

        applicant_value = input_df.iloc[0][feat]
        went_left = applicant_value <= threshold

        comparison = "at or below" if went_left else "above"
        reasons.append(
            {
                "feature": feat,
                "text": f"**{PRETTY[feat]}** of {fmt(feat, applicant_value)} is "
                f"{comparison} the threshold of {fmt(feat, threshold)}",
            }
        )

    return reasons, leaf_id


def leaf_stats(leaf_id):
    """How many real training applicants share this applicant's 'profile group'."""
    tree = model.tree_
    n_samples = int(tree.n_node_samples[leaf_id])
    # weighted class distribution at the leaf (class_weight='balanced' was used)
    values = tree.value[leaf_id][0]
    approve_share = values[1] / values.sum()
    return n_samples, approve_share


def gauge(prob_approve):
    color = "#2ca02c" if prob_approve >= 0.5 else "#d62728"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_approve * 100,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#f8d7da"},
                    {"range": [50, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": "Approval Probability"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def comparison_chart(input_df):
    """Bar chart: applicant value vs dataset median for each feature (percentiles)."""
    pcts = [percentile_of(f, input_df.iloc[0][f]) for f in FEATURES]
    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=[PRETTY[f] for f in FEATURES],
            orientation="h",
            text=[f"{p:.0f}th percentile" for p in pcts],
            textposition="auto",
            marker_color=["#4c78a8" if p >= 50 else "#e45756" for p in pcts],
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color="gray",
                  annotation_text="median applicant")
    fig.update_layout(
        height=280,
        xaxis_title="Percentile vs 45,000 applicants",
        xaxis_range=[0, 100],
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


# ----------------------------------------------------------------------
# Sidebar — applicant inputs
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Applicant Details")
    age = st.slider("Age (years)", 18, 80, 28)
    income = st.number_input("Annual Income (₹)", 0, 2_000_000, 60_000, step=1_000)
    loan_amount = st.number_input("Loan Amount (₹)", 500, 1_000_000, 20_000, step=500)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    predict = st.button("🔍 Predict", type="primary", use_container_width=True)
    st.divider()
    st.caption("Decision Tree Classifier · trained on 45,000 records")
    st.caption("Shivanshu Tiwari — 2400270130173")


# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
st.title("🏦 Loan Approval Prediction")
st.markdown(
    "An **explainable** loan predictor. It doesn't just decide — it shows the "
    "exact rules behind the decision and the real data driving the confidence."
)

if not predict:
    st.info("👈 Enter the applicant's details in the sidebar and click **Predict**.")

    # Show dataset context while idle
    st.subheader("About the training data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total applicants", f"{len(ref):,}")
    c2.metric("Approved", f"{int(ref['Approved'].sum()):,}",
              f"{ref['Approved'].mean()*100:.0f}% of all")
    c3.metric("Median income", f"₹{ref['Income'].median():,.0f}")
    c4.metric("Median credit score", f"{ref['CreditScore'].median():.0f}")
    st.stop()


# --- Run prediction ---------------------------------------------------
input_df = pd.DataFrame(
    [[age, income, loan_amount, credit_score]], columns=FEATURES
)
x_scaled = scaler.transform(input_df)
pred = int(model.predict(x_scaled)[0])
proba = model.predict_proba(x_scaled)[0]
prob_approve = float(proba[1])

reasons, leaf_id = decision_path_reasons(input_df)
n_similar, leaf_approve_share = leaf_stats(leaf_id)

# --- Headline decision ------------------------------------------------
left, right = st.columns([1, 1])
with left:
    if pred == 1:
        st.success("## ✅ Loan APPROVED")
    else:
        st.error("## ❌ Loan REJECTED")
    st.markdown(
        f"The model is **{max(prob_approve, 1-prob_approve)*100:.1f}% confident** "
        f"in this decision."
    )
    st.plotly_chart(gauge(prob_approve), use_container_width=True)

with right:
    st.markdown("#### How this applicant compares")
    st.plotly_chart(comparison_chart(input_df), use_container_width=True)

st.divider()

# --- Plain-English summary -------------------------------------------
st.subheader("📋 Why this decision?")

verdict_word = "approve" if pred == 1 else "reject"
summary = (
    f"The model chose to **{verdict_word}** this loan after applying "
    f"**{len(reasons)} decision rule(s)** to the applicant's profile. "
)

# Identify the strongest comparative signals for the narrative
credit_pct = percentile_of("CreditScore", credit_score)
income_pct = percentile_of("Income", income)
loan_pct = percentile_of("LoanAmount", loan_amount)

narrative_bits = []
if income_pct >= 60:
    narrative_bits.append(f"a **strong income** (higher than {income_pct:.0f}% of applicants)")
elif income_pct <= 40:
    narrative_bits.append(f"a **modest income** (lower than {100-income_pct:.0f}% of applicants)")

if credit_pct >= 60:
    narrative_bits.append(f"a **healthy credit score** (top {100-credit_pct:.0f}%)")
elif credit_pct <= 40:
    narrative_bits.append(f"a **below-average credit score** (bottom {credit_pct:.0f}%)")

if loan_pct >= 70:
    narrative_bits.append(f"a **large requested amount** (higher than {loan_pct:.0f}% of applicants)")

if narrative_bits:
    summary += "Key factors in this profile: " + ", ".join(narrative_bits) + "."

st.markdown(summary)

# --- The exact rules the tree applied --------------------------------
st.markdown("#### The exact rules applied (in order)")
for i, r in enumerate(reasons, 1):
    st.markdown(f"{i}. {r['text']}")

# --- Where the percentage comes from ---------------------------------
st.markdown("#### Where the confidence percentage comes from")
st.info(
    f"This applicant lands in a group of **{n_similar:,} similar training applicants** "
    f"(same answers to all the rules above). Within that group, the approval rate is "
    f"**{leaf_approve_share*100:.1f}%** — which is exactly the probability the model reports. "
    f"The percentage is not arbitrary; it is the historical approval rate for this profile."
)

# --- Feature importance (global context) -----------------------------
with st.expander("🌳 Which features matter most to the model overall?"):
    imp = pd.Series(model.feature_importances_, index=[PRETTY[f] for f in FEATURES])
    imp = imp.sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker_color="#2b7bba",
                           text=[f"{v*100:.0f}%" for v in imp.values],
                           textposition="auto"))
    fig.update_layout(height=250, xaxis_title="Relative importance",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Income and Loan Amount dominate the model's decisions; Age and Credit "
        "Score play smaller roles in this dataset."
    )

with st.expander("🔢 Applicant input summary"):
    show = input_df.copy()
    show["Income"] = show["Income"].map(lambda v: f"₹{v:,.0f}")
    show["LoanAmount"] = show["LoanAmount"].map(lambda v: f"₹{v:,.0f}")
    st.dataframe(show, hide_index=True, use_container_width=True)
