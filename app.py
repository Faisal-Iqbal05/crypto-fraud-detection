from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.helpers import (
    build_daily_volume_chart,
    build_feature_importance_table,
    build_prediction_report_pdf,
    compute_dataset_summary,
    compute_prediction,
    load_app_bundle,
    load_css,
)
from utils.preprocessing import build_prediction_input


BASE_DIR = Path(__file__).resolve().parent
STYLES_PATH = BASE_DIR / "assets" / "styles.css"
AUTHOR_NAME = "Product Engineering Team"
AUTHOR_EMAIL = "team@cryptoshield.ai"


st.set_page_config(
    page_title="CryptoShield AI",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme(dark_mode: bool) -> None:
    css = load_css(STYLES_PATH)
    theme_vars = """
    :root {
      --bg-main: #07111f;
      --bg-panel: rgba(11, 22, 44, 0.88);
      --bg-soft: rgba(255, 255, 255, 0.04);
      --text-main: #e8ecff;
      --text-muted: #9fb0df;
      --line: rgba(151, 166, 214, 0.18);
      --surface-gradient: linear-gradient(135deg, #04101d 0%, #07111f 40%, #08172a 100%);
      --sidebar-gradient: linear-gradient(180deg, rgba(8, 16, 34, 0.98), rgba(9, 18, 40, 0.92));
      --hero-gradient: linear-gradient(135deg, rgba(77, 163, 255, 0.24), rgba(139, 92, 246, 0.18));
    }
    """ if dark_mode else """
    :root {
      --bg-main: #f3f7ff;
      --bg-panel: rgba(255, 255, 255, 0.92);
      --bg-soft: rgba(26, 76, 255, 0.05);
      --text-main: #10213d;
      --text-muted: #4f6488;
      --line: rgba(52, 84, 160, 0.12);
      --surface-gradient: linear-gradient(135deg, #eef5ff 0%, #f8fbff 45%, #eef2ff 100%);
      --sidebar-gradient: linear-gradient(180deg, rgba(247, 250, 255, 0.98), rgba(235, 242, 255, 0.94));
      --hero-gradient: linear-gradient(135deg, rgba(77, 163, 255, 0.16), rgba(139, 92, 246, 0.12));
    }
    """
    st.markdown(f"<style>{theme_vars}{css}</style>", unsafe_allow_html=True)


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="hero-copy">
                <span class="hero-kicker">Crypto Fraud Command Center</span>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div class="hero-badge">
                <span>Live Monitoring</span>
                <strong>Production Ready</strong>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, delta: str, icon: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-copy">
                <span>{label}</span>
                <h3>{value}</h3>
                <p>{delta}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(bundle: dict[str, object]) -> None:
    render_header(
        "Fintech-grade fraud intelligence for crypto operations.",
        "Monitor transaction risk, model performance, and fraud patterns from a single polished workspace.",
    )

    summary = compute_dataset_summary(bundle["dataset"])
    metrics = bundle["metrics_table"]
    best_model = bundle["best_model_name"]
    accuracy = metrics.loc[best_model, "accuracy"]
    fraud_rate = summary["fraud_rate"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Transactions", f"{summary['total_transactions']:,}", "Synthetic production dataset", "TX", "tone-blue")
    with col2:
        render_metric_card("Fraud Detected", f"{summary['fraud_transactions']:,}", f"{fraud_rate:.2f}% flagged", "FR", "tone-red")
    with col3:
        render_metric_card("Accuracy", f"{accuracy * 100:.2f}%", "Best evaluated model", "ML", "tone-green")
    with col4:
        render_metric_card("Model Used", best_model, "Primary scoring engine", "AI", "tone-purple")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns((1.4, 1))
    with left_col:
        st.markdown("### Transaction Flow")
        daily_chart = build_daily_volume_chart(bundle["dataset"])
        st.plotly_chart(daily_chart, use_container_width=True)

    with right_col:
        st.markdown("### Fraud Mix")
        fraud_mix = pd.DataFrame(
            {
                "segment": ["Legitimate", "Fraudulent"],
                "count": [summary["legitimate_transactions"], summary["fraud_transactions"]],
            }
        )
        donut = px.pie(
            fraud_mix,
            names="segment",
            values="count",
            hole=0.68,
            color="segment",
            color_discrete_map={"Legitimate": "#1dd1a1", "Fraudulent": "#ff5d73"},
        )
        donut.update_traces(textposition="inside", textinfo="percent+label")
        donut.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8ecff"),
        )
        st.plotly_chart(donut, use_container_width=True)

    lower_left, lower_right = st.columns((1.1, 1))
    with lower_left:
        st.markdown("### Model Leaderboard")
        leaderboard = metrics.reset_index().rename(columns={"index": "model"})
        perf_chart = px.bar(
            leaderboard,
            x="model",
            y=["accuracy", "recall", "f1", "roc_auc"],
            barmode="group",
            color_discrete_sequence=["#4da3ff", "#9b7bff", "#19d3a2", "#ff7a59"],
        )
        perf_chart.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Metric",
            font=dict(color="#e8ecff"),
        )
        st.plotly_chart(perf_chart, use_container_width=True)

    with lower_right:
        st.markdown("### Operations Snapshot")
        insights = [
            ("Primary model", bundle["primary_model_name"]),
            ("Recommended threshold", "0.50 probability"),
            ("Most common region", summary["top_region"]),
            ("Most common transaction", summary["top_transaction_type"]),
        ]
        for label, value in insights:
            st.markdown(
                f"""
                <div class="insight-row">
                    <span>{label}</span>
                    <strong>{value}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_page(bundle: dict[str, object]) -> None:
    render_header(
        "Score high-value transactions before they become incidents.",
        "Use the guided transaction form for real-time risk scoring, confidence estimation, and downloadable analyst reports.",
    )

    models = bundle["model_names"]
    default_model_index = models.index(bundle["primary_model_name"])

    top_row_left, top_row_right = st.columns((1.2, 0.8))
    with top_row_left:
        model_name = st.selectbox("Scoring model", models, index=default_model_index)
    with top_row_right:
        live_mode = st.toggle("Real-time preview", value=True)

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1250.0, step=50.0)
            account_balance = st.number_input("Account Balance", min_value=0.0, value=4200.0, step=100.0)
            transaction_fee = st.number_input("Transaction Fee", min_value=0.0, value=2.4, step=0.1)
            confirmation_time_minutes = st.slider("Confirmation Time (minutes)", min_value=1, max_value=60, value=18)
        with col2:
            num_inputs = st.slider("Number of Inputs", min_value=1, max_value=12, value=3)
            num_outputs = st.slider("Number of Outputs", min_value=1, max_value=12, value=2)
            transaction_type = st.selectbox("Transaction Type", bundle["transaction_types"])
            region = st.selectbox("Region", bundle["regions"])

        advanced_open = st.checkbox("Show advanced context", value=False)
        account_id = None
        timestamp = None
        if advanced_open:
            adv1, adv2 = st.columns(2)
            with adv1:
                account_id = st.text_input("Account ID", value="acct_streamlit_demo")
            with adv2:
                timestamp = st.text_input(
                    "Timestamp",
                    value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    help="Expected format: YYYY-MM-DD HH:MM:SS",
                )

        submitted = st.form_submit_button("Predict Fraud", use_container_width=True)

    form_values = {
        "transaction_amount": transaction_amount,
        "account_balance": account_balance,
        "transaction_fee": transaction_fee,
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "confirmation_time_minutes": confirmation_time_minutes,
        "transaction_type": transaction_type,
        "region": region,
        "account_id": account_id,
        "timestamp": timestamp,
    }
    sample_df = build_prediction_input(form_values, bundle["sample_input_template"])

    if live_mode or submitted:
        prediction = compute_prediction(bundle, sample_df, model_name)
        label_class = "alert-danger" if prediction["prediction"] == 1 else "alert-success"
        emoji = "ALERT" if prediction["prediction"] == 1 else "CLEAR"

        result_col, gauge_col = st.columns((1, 1))
        with result_col:
            st.markdown(
                f"""
                <div class="result-panel {label_class}">
                    <span class="result-kicker">Risk decision</span>
                    <h2>{emoji} {prediction['label']}</h2>
                    <p>Fraud probability: {prediction['fraud_probability'] * 100:.2f}%</p>
                    <p>Confidence score: {prediction['confidence_score'] * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Analyst Notes")
            for reason in prediction["reasons"]:
                st.markdown(f"- {reason}")

        with gauge_col:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction["fraud_probability"] * 100,
                    number={"suffix": "%", "font": {"color": "#e8ecff"}},
                    title={"text": "Fraud Probability", "font": {"color": "#e8ecff"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8892c7"},
                        "bar": {"color": "#ff5d73"},
                        "bgcolor": "rgba(255,255,255,0.03)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40], "color": "rgba(29,209,161,0.30)"},
                            {"range": [40, 70], "color": "rgba(255,184,77,0.30)"},
                            {"range": [70, 100], "color": "rgba(255,93,115,0.35)"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=10),
                height=320,
            )
            st.plotly_chart(gauge, use_container_width=True)

        report_df = pd.DataFrame(
            [
                {
                    "model": prediction["model_name"],
                    "prediction": prediction["label"],
                    "fraud_probability": round(prediction["fraud_probability"], 4),
                    "confidence_score": round(prediction["confidence_score"], 4),
                    **sample_df.iloc[0].to_dict(),
                }
            ]
        )

        export_left, export_right = st.columns(2)
        with export_left:
            st.download_button(
                "Download CSV Report",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name="crypto_fraud_prediction_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_right:
            pdf_bytes = build_prediction_report_pdf(report_df.iloc[0].to_dict(), prediction)
            st.download_button(
                "Download PDF Brief",
                data=pdf_bytes,
                file_name="crypto_fraud_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


def render_analytics_page(bundle: dict[str, object]) -> None:
    render_header(
        "Interrogate model behavior with interactive fraud analytics.",
        "Explore confusion matrices, ROC performance, feature strength, and transaction patterns with analyst-friendly filters.",
    )

    filters_col, charts_col = st.columns((0.9, 1.1))
    with filters_col:
        selected_model = st.selectbox("Model focus", bundle["model_names"], index=bundle["model_names"].index(bundle["primary_model_name"]))
        selected_region = st.multiselect("Region filter", bundle["regions"], default=bundle["regions"])
        selected_transaction_types = st.multiselect(
            "Transaction type filter",
            bundle["transaction_types"],
            default=bundle["transaction_types"],
        )

        filtered_df = bundle["dataset"].copy()
        filtered_df = filtered_df[filtered_df["region"].isin(selected_region)]
        filtered_df = filtered_df[filtered_df["transaction_type"].isin(selected_transaction_types)]
        st.markdown("### Filtered Snapshot")
        st.dataframe(
            filtered_df[
                [
                    "timestamp",
                    "transaction_amount",
                    "account_balance",
                    "transaction_type",
                    "region",
                    "label",
                ]
            ].head(15),
            use_container_width=True,
            hide_index=True,
        )

    with charts_col:
        comparison_table = bundle["metrics_table"].reset_index().rename(columns={"index": "Model"})
        st.markdown("### Model Comparison")
        st.dataframe(
            comparison_table.style.format(
                {
                    "accuracy": "{:.3f}",
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1": "{:.3f}",
                    "roc_auc": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        conf_matrix = bundle["confusion_matrices"][selected_model]
        heatmap = px.imshow(
            conf_matrix,
            text_auto=True,
            color_continuous_scale=[[0, "#101c38"], [0.5, "#3f6cf6"], [1, "#ff5d73"]],
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            aspect="auto",
        )
        heatmap.update_layout(
            title=f"Confusion Matrix: {selected_model}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8ecff"),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(heatmap, use_container_width=True)

        roc_figure = go.Figure()
        for model_name, roc_points in bundle["roc_curves"].items():
            roc_figure.add_trace(
                go.Scatter(
                    x=roc_points["fpr"],
                    y=roc_points["tpr"],
                    mode="lines",
                    name=f"{model_name} (AUC {roc_points['auc']:.3f})",
                    line={"width": 3 if model_name == selected_model else 2},
                )
            )
        roc_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Baseline",
                line={"dash": "dash", "color": "#7f8ab6"},
            )
        )
        roc_figure.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8ecff"),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(roc_figure, use_container_width=True)

    bottom_left, bottom_right = st.columns((1, 1))
    with bottom_left:
        st.markdown("### Feature Importance")
        importance_df = build_feature_importance_table(bundle)
        importance_chart = px.bar(
            importance_df.head(15).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#0ea5e9", "#8b5cf6", "#22c55e"],
        )
        importance_chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8ecff"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(importance_chart, use_container_width=True)

    with bottom_right:
        st.markdown("### Transaction Pattern Explorer")
        filtered_df = bundle["dataset"].copy()
        filtered_df = filtered_df[filtered_df["region"].isin(selected_region)]
        filtered_df = filtered_df[filtered_df["transaction_type"].isin(selected_transaction_types)]
        scatter = px.scatter(
            filtered_df,
            x="transaction_amount",
            y="transaction_fee",
            color=filtered_df["label"].map({0: "Legitimate", 1: "Fraudulent"}),
            size="confirmation_time_minutes",
            hover_data=["account_balance", "region", "transaction_type"],
            color_discrete_map={"Legitimate": "#1dd1a1", "Fraudulent": "#ff5d73"},
        )
        scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8ecff"),
            legend_title_text="Class",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(scatter, use_container_width=True)


def render_about_page(bundle: dict[str, object]) -> None:
    render_header(
        "Built for real users, not classroom demos.",
        "This Streamlit application wraps a full fraud workflow: modern dashboard UX, reusable ML helpers, explainable analytics, and downloadable reports.",
    )

    overview_col, tech_col = st.columns((1.1, 0.9))
    with overview_col:
        st.markdown(
            """
            ### Project Overview
            CryptoShield AI helps analysts and operations teams assess suspicious cryptocurrency transactions
            with a streamlined, production-minded interface. The experience emphasizes clear decision support,
            readable analytics, and fast access to the signals that matter most.
            """
        )

        dataset_summary = compute_dataset_summary(bundle["dataset"])
        st.markdown(
            f"""
            ### Dataset
            - Source: `data/crypto_transactions.csv`
            - Volume: `{dataset_summary["total_transactions"]:,}` transactions
            - Fraud ratio: `{dataset_summary["fraud_rate"]:.2f}%`
            - Coverage: `{len(bundle["regions"])}` regions and `{len(bundle["transaction_types"])}` transaction types
            """
        )

    with tech_col:
        st.markdown(
            f"""
            ### Stack
            - Streamlit for the app shell and interaction model
            - scikit-learn + imbalanced-learn for the ML pipeline
            - Plotly for interactive analytics
            - ReportLab for PDF brief generation

            ### Author
            - Team: {AUTHOR_NAME}
            - Contact: {AUTHOR_EMAIL}
            - Build: Streamlit fintech dashboard edition
            """
        )

    st.markdown("### Architecture")
    st.code(
        """crypto_fraud_app/
app.py
assets/styles.css
utils/helpers.py
utils/preprocessing.py
models/
data/
src/train.py""",
        language="text",
    )


def main() -> None:
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    with st.sidebar:
        st.markdown("## CryptoShield AI")
        st.caption("Fraud detection workspace for crypto operations teams")
        page = st.radio("Navigate", ["Dashboard", "Predict Fraud", "Analytics", "About"], label_visibility="collapsed")
        st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
        st.markdown("---")
        st.markdown("**Live capabilities**")
        st.markdown("- Real-time scoring")
        st.markdown("- Model switcher")
        st.markdown("- CSV and PDF exports")
        st.markdown("- Interactive analyst charts")

    apply_theme(st.session_state.dark_mode)
    bundle = load_app_bundle()

    if page == "Dashboard":
        render_dashboard(bundle)
    elif page == "Predict Fraud":
        render_prediction_page(bundle)
    elif page == "Analytics":
        render_analytics_page(bundle)
    else:
        render_about_page(bundle)


if __name__ == "__main__":
    main()
