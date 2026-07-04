"""
Loan Approval Prediction - Deployment App (Streamlit)
Author : Shivanshu Tiwari (Roll No: 2400270130173)

An explainable loan approval predictor built on a Decision Tree classifier.
Besides the Approve/Reject decision, the app explains the reasoning: the exact
rules the tree applied, how the applicant compares to 45,000 real applicants,
and the data behind the confidence percentage.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------
INK = "#1f2a37"       # primary text / headings
SLATE = "#64748b"     # secondary text
ACCENT = "#1d4ed8"    # primary accent (blue)
GREEN = "#15803d"     # approve
RED = "#b91c1c"       # reject
LINE = "#e2e8f0"      # borders

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
    h1, h2, h3, h4 {{ color: {INK}; font-weight: 600; letter-spacing: -0.01em; }}
    .app-title {{ font-size: 1.9rem; font-weight: 650; color: {INK}; margin-bottom: .1rem; }}
    .app-sub {{ color: {SLATE}; font-size: 1rem; margin-bottom: 1.4rem; }}
    .card {{
        border: 1px solid {LINE}; border-radius: 12px; padding: 1.15rem 1.3rem;
        background: #ffffff; margin-bottom: 1rem;
    }}
    .verdict {{
        border-radius: 12px; padding: 1.2rem 1.4rem; color: #fff;
        font-size: 1.4rem; font-weight: 600; margin-bottom: .6rem;
    }}
    .verdict-sub {{ font-size: .95rem; font-weight: 400; opacity: .9; margin-top: .2rem; }}
    .rule {{
        border-left: 3px solid {ACCENT}; padding: .45rem .8rem; margin-bottom: .4rem;
        background: #f8fafc; border-radius: 0 6px 6px 0; font-size: .93rem; color: {INK};
    }}
    .note {{
        border: 1px solid {LINE}; border-left: 3px solid {SLATE};
        border-radius: 0 8px 8px 0; padding: .9rem 1.1rem; background: #f8fafc;
        color: {INK}; font-size: .95rem; line-height: 1.55;
    }}
    .section-label {{
        text-transform: uppercase; letter-spacing: .06em; font-size: .78rem;
        color: {SLATE}; font-weight: 600; margin: 1.1rem 0 .5rem 0;
    }}
    .stButton > button {{
        background: {ACCENT}; color: #fff; border: none; border-radius: 8px;
        font-weight: 600; padding: .55rem 0;
    }}
    .stButton > button:hover {{ background: #1e40af; color: #fff; }}
    [data-testid="stMetricValue"] {{ font-size: 1.4rem; color: {INK}; }}
    footer, #MainMenu {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
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
    return pd.read_csv("loan_data.csv").drop_duplicates()


model, scaler = load_artifacts()
ref = load_reference()


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
        went_left = applicant_value <= threshold
        comparison = "at or below" if went_left else "above"
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


def gauge(prob_approve):
    color = GREEN if prob_approve >= 0.5 else RED
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_approve * 100,
            number={"suffix": "%", "font": {"size": 34, "color": INK}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": SLATE},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#fbeaea"},
                    {"range": [50, 100], "color": "#e8f2ec"},
                ],
                "threshold": {
                    "line": {"color": SLATE, "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=10, b=10),
        font=dict(color=INK, family="Inter, system-ui, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def comparison_chart(input_df):
    pcts = [percentile_of(f, input_df.iloc[0][f]) for f in FEATURES]
    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=[PRETTY[f] for f in FEATURES],
            orientation="h",
            text=[f"{p:.0f}th pct" for p in pcts],
            textposition="auto",
            marker_color=[ACCENT if p >= 50 else "#94a3b8" for p in pcts],
            marker_line_width=0,
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color=SLATE,
                  annotation_text="median", annotation_font_color=SLATE)
    fig.update_layout(
        height=240,
        xaxis_title="Percentile vs 45,000 applicants",
        xaxis_range=[0, 100],
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color=INK, family="Inter, system-ui, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=LINE),
    )
    return fig


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Applicant Details")
    age = st.slider("Age (years)", 18, 80, 28)
    income = st.number_input("Annual Income (Rs)", 0, 2_000_000, 60_000, step=1_000)
    loan_amount = st.number_input("Loan Amount (Rs)", 500, 1_000_000, 20_000, step=500)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    predict = st.button("Predict", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Decision Tree Classifier · trained on 45,000 records")
    st.caption("Shivanshu Tiwari — 2400270130173")


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="app-title">Loan Approval Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">An explainable credit-decision model. Every prediction is '
    'accompanied by the rules applied and the data behind its confidence.</div>',
    unsafe_allow_html=True,
)

if not predict:
    st.markdown('<div class="section-label">Training data overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total applicants", f"{len(ref):,}")
    c2.metric("Approval rate", f"{ref['Approved'].mean()*100:.0f}%",
              f"{int(ref['Approved'].sum()):,} approved")
    c3.metric("Median income", f"Rs {ref['Income'].median():,.0f}")
    c4.metric("Median credit score", f"{ref['CreditScore'].median():.0f}")
    st.markdown(
        '<div class="note" style="margin-top:1rem">Enter the applicant\'s details in '
        'the panel on the left and select <b>Predict</b> to generate a decision.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
input_df = pd.DataFrame([[age, income, loan_amount, credit_score]], columns=FEATURES)
x_scaled = scaler.transform(input_df)
pred = int(model.predict(x_scaled)[0])
proba = model.predict_proba(x_scaled)[0]
prob_approve = float(proba[1])

reasons, leaf_id = decision_path_reasons(input_df)
n_similar, leaf_approve_share = leaf_stats(leaf_id)

left, right = st.columns([1, 1])
with left:
    if pred == 1:
        st.markdown(
            f'<div class="verdict" style="background:{GREEN}">Loan Approved'
            f'<div class="verdict-sub">Model confidence '
            f'{max(prob_approve, 1-prob_approve)*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict" style="background:{RED}">Loan Rejected'
            f'<div class="verdict-sub">Model confidence '
            f'{max(prob_approve, 1-prob_approve)*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-label">Approval probability</div>', unsafe_allow_html=True)
    st.plotly_chart(gauge(prob_approve), use_container_width=True, config={"displayModeBar": False})

with right:
    st.markdown('<div class="section-label">Applicant vs. population</div>', unsafe_allow_html=True)
    st.plotly_chart(comparison_chart(input_df), use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# --- Reasoning --------------------------------------------------------
st.markdown("#### Basis for the decision")

verdict_word = "approve" if pred == 1 else "reject"
credit_pct = percentile_of("CreditScore", credit_score)
income_pct = percentile_of("Income", income)
loan_pct = percentile_of("LoanAmount", loan_amount)

bits = []
if income_pct >= 60:
    bits.append(f"a strong income (higher than {income_pct:.0f}% of applicants)")
elif income_pct <= 40:
    bits.append(f"a modest income (lower than {100-income_pct:.0f}% of applicants)")
if credit_pct >= 60:
    bits.append(f"a healthy credit score (top {100-credit_pct:.0f}%)")
elif credit_pct <= 40:
    bits.append(f"a below-average credit score (bottom {credit_pct:.0f}%)")
if loan_pct >= 70:
    bits.append(f"a large requested amount (higher than {loan_pct:.0f}% of applicants)")

summary = (
    f"The model chose to <b>{verdict_word}</b> this loan after applying "
    f"<b>{len(reasons)}</b> sequential decision rules to the applicant's profile."
)
if bits:
    summary += " Notable characteristics of this profile: " + ", ".join(bits) + "."
st.markdown(f'<div class="note">{summary}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Rules applied, in order</div>', unsafe_allow_html=True)
for i, r in enumerate(reasons, 1):
    st.markdown(f'<div class="rule"><b>{i}.</b> {r}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">How the confidence is derived</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="note">This applicant falls into a group of <b>{n_similar:,}</b> '
    f'training applicants that satisfy the same rules above. Within that group, the '
    f'historical approval rate is <b>{leaf_approve_share*100:.1f}%</b> — which is exactly '
    f'the probability the model reports. The confidence is grounded in observed outcomes, '
    f'not an arbitrary score.</div>',
    unsafe_allow_html=True,
)

# --- Supporting detail ------------------------------------------------
with st.expander("Feature importance across the model"):
    imp = pd.Series(model.feature_importances_, index=[PRETTY[f] for f in FEATURES])
    imp = imp.sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color=ACCENT, marker_line_width=0,
        text=[f"{v*100:.0f}%" for v in imp.values], textposition="auto",
    ))
    fig.update_layout(
        height=230, xaxis_title="Relative importance",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color=INK, family="Inter, system-ui, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=LINE),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Income and Loan Amount are the dominant drivers of the model's decisions; "
        "Age and Credit Score play smaller roles in this dataset."
    )

with st.expander("Applicant input summary"):
    show = input_df.copy()
    show["Income"] = show["Income"].map(lambda v: f"Rs {v:,.0f}")
    show["LoanAmount"] = show["LoanAmount"].map(lambda v: f"Rs {v:,.0f}")
    st.dataframe(show, hide_index=True, use_container_width=True)
