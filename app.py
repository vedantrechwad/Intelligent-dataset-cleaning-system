import os
import io
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.helpers import load_config, NpEncoder, logger
from src.ingestion.file_loader import load_dataset
from src.ingestion.schema_detector import detect_dataset_schema
from src.profiling.profiler import profile_dataset
from src.profiling.quality_metrics import compute_quality_score
from src.detection.anomaly_detector import detect_all_issues
from src.intelligence.ollama_client import OllamaClient, RECOMMENDED_MODELS
from src.intelligence.reasoning_engine import ReasoningEngine
from src.review.human_review import HumanReviewManager, validate_user_correction_input
from src.cleaning.pipeline import execute_cleaning_pipeline
from src.validation.validator import validate_cleaned_dataset
from src.validation.before_after import compute_before_after_comparison
from src.reporting.audit_logger import AuditLogger
from src.reporting.report_generator import generate_html_report
from src.evaluation.corruption_engine import DataCorruptionEngine
from src.evaluation.detection_metrics import evaluate_detection_performance
from src.evaluation.correction_metrics import evaluate_correction_performance
from src.evaluation.downstream_evaluation import evaluate_downstream_ml

# Configure Streamlit Page
st.set_page_config(
    page_title="Intelligent Data Quality & Auto-Cleaning System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern UI aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-auto {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-review {
        background-color: #78350f;
        color: #fbbf24;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-human {
        background-color: #881337;
        color: #f43f5e;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "working_df" not in st.session_state:
    st.session_state.working_df = None
if "dataset_meta" not in st.session_state:
    st.session_state.dataset_meta = None
if "schema_info" not in st.session_state:
    st.session_state.schema_info = None
if "raw_profile" not in st.session_state:
    st.session_state.raw_profile = None
if "raw_issues" not in st.session_state:
    st.session_state.raw_issues = None
if "raw_quality_score" not in st.session_state:
    st.session_state.raw_quality_score = None
if "human_decisions" not in st.session_state:
    st.session_state.human_decisions = {}
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "cleaning_audit" not in st.session_state:
    st.session_state.cleaning_audit = None
if "after_profile" not in st.session_state:
    st.session_state.after_profile = None
if "after_issues" not in st.session_state:
    st.session_state.after_issues = None
if "after_quality_score" not in st.session_state:
    st.session_state.after_quality_score = None
if "eval_benchmark_results" not in st.session_state:
    st.session_state.eval_benchmark_results = None

config = load_config()

# Sidebar: Ollama Intelligence & Model Selector
st.sidebar.markdown("### 🤖 Local Ollama LLM")
ollama_client = OllamaClient(
    host=config.get("ollama", {}).get("host", "http://127.0.0.1:11434"),
    model=config.get("ollama", {}).get("model", "qwen2.5:7b")
)

is_ollama_online = ollama_client.check_availability()
if not is_ollama_online:
    # Attempt auto start
    is_ollama_online = ollama_client.auto_start_if_needed()

if is_ollama_online:
    st.sidebar.success("🟢 Ollama Active (127.0.0.1:11434)")
    installed_models = ollama_client.get_installed_models()
    selected_model = st.sidebar.selectbox(
        "Active Reasoning Model",
        installed_models if installed_models else ["qwen2.5:7b"],
        index=0
    )
    ollama_client.model = selected_model
else:
    st.sidebar.warning("⚪ Ollama Offline (Fallback Heuristics Mode)")
    st.sidebar.caption("System works 100% deterministically with SciPy, RapidFuzz & Scikit-learn.")
    if st.sidebar.button("🔄 Check / Auto-Start Ollama"):
        ollama_client.auto_start_if_needed()
        st.rerun()

with st.sidebar.expander("💡 Recommended Local Models"):
    for m in RECOMMENDED_MODELS:
        st.markdown(f"**`{m['name']}`** ({m['parameters']})\n{m['description']}")

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Quick System Actions")
if st.sidebar.button("🧹 Reset All State", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main Application Title
st.markdown('<div class="main-header">Intelligent Dataset Quality & Automated Cleaning System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-Style Tabular Profiling, Anomaly Detection, Human-in-the-Loop Review, and Ground-Truth Evaluation</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tabs = st.tabs([
    "📁 1. Ingestion & Preview",
    "📊 2. Data Profiling",
    "🔍 3. Detected Issues",
    "🧑‍💻 4. Human Review",
    "⚡ 5. Cleaning Pipeline",
    "📈 6. Validation & Deltas",
    "🧪 7. Evaluation & ML Benchmark",
    "📥 8. Audit Trail & Export"
])

# -------------------------------------------------------------
# TAB 1: Ingestion & Upload
# -------------------------------------------------------------
with tabs[0]:
    st.markdown("### 📤 Upload Tabular Dataset")
    col_u1, col_u2 = st.columns([2, 1])

    with col_u1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Original raw data is preserved strictly without modification."
        )

    with col_u2:
        st.markdown("**Or test with built-in benchmark datasets:**")
        if st.button("🧪 Load Synthetic Dirty Customer Dataset (CSV)", use_container_width=True):
            sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "dirty_customer_dataset.csv")
            if os.path.exists(sample_path):
                orig_df, work_df, meta = load_dataset(sample_path)
                st.session_state.raw_df = orig_df
                st.session_state.working_df = work_df
                st.session_state.dataset_meta = meta
                # Run Initial Analysis
                with st.spinner("Analyzing dataset schema and issues..."):
                    st.session_state.schema_info = detect_dataset_schema(work_df)
                    st.session_state.raw_profile = profile_dataset(work_df, st.session_state.schema_info)
                    raw_issues = detect_all_issues(work_df)
                    reasoning_engine = ReasoningEngine(ollama_client)
                    st.session_state.raw_issues = reasoning_engine.enrich_issues(raw_issues, work_df, use_ollama=is_ollama_online)
                    st.session_state.raw_quality_score = compute_quality_score(work_df, detected_issues=st.session_state.raw_issues)
                st.success("Loaded 'dirty_customer_dataset.csv' successfully!")
                st.rerun()

    if uploaded_file is not None:
        try:
            orig_df, work_df, meta = load_dataset(uploaded_file, filename=uploaded_file.name)
            st.session_state.raw_df = orig_df
            st.session_state.working_df = work_df
            st.session_state.dataset_meta = meta
            if st.session_state.raw_profile is None:
                with st.spinner("Profiling dataset and detecting quality anomalies..."):
                    st.session_state.schema_info = detect_dataset_schema(work_df)
                    st.session_state.raw_profile = profile_dataset(work_df, st.session_state.schema_info)
                    raw_issues = detect_all_issues(work_df)
                    reasoning_engine = ReasoningEngine(ollama_client)
                    st.session_state.raw_issues = reasoning_engine.enrich_issues(raw_issues, work_df, use_ollama=is_ollama_online)
                    st.session_state.raw_quality_score = compute_quality_score(work_df, detected_issues=st.session_state.raw_issues)
                st.success(f"Loaded '{meta['filename']}' ({meta['row_count']} rows, {meta['column_count']} cols).")
        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}")

    if st.session_state.raw_df is not None:
        meta = st.session_state.dataset_meta
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{meta['row_count']:,}")
        m2.metric("Columns", meta['column_count'])
        m3.metric("Format", meta['format'])
        m4.metric("Memory", f"{meta['memory_usage_bytes'] / 1024:.1f} KB")
        m5.metric("Detected Issues", len(st.session_state.raw_issues) if st.session_state.raw_issues else 0)

        st.markdown("#### 🔍 Raw Dataset Preview (First 15 Rows)")
        st.dataframe(st.session_state.raw_df.head(15), use_container_width=True)

        st.markdown("#### 🧬 Inferred Logical Column Schema")
        schema_rows = []
        for col, s in st.session_state.schema_info.items():
            schema_rows.append({
                "Column Name": col,
                "Logical Type": s.get("logical_type"),
                "Semantic Hint": s.get("semantic_hint"),
                "Nullable": s.get("is_nullable"),
                "Sample Values": ", ".join([str(x) for x in s.get("sample_values", [])[:3]])
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)
    else:
        st.info("👆 Please upload a CSV/XLSX file or click 'Load Synthetic Dirty Customer Dataset' to begin.")

# -------------------------------------------------------------
# TAB 2: Profiling & Statistics
# -------------------------------------------------------------
with tabs[1]:
    if st.session_state.raw_profile is None:
        st.info("Please upload a dataset first in Tab 1.")
    else:
        profile = st.session_state.raw_profile
        qscore = st.session_state.raw_quality_score

        st.markdown("### 📊 Dataset Quality & Statistical Profile")
        
        # Quality Score Banner
        col_q1, col_q2 = st.columns([1, 2])
        with col_q1:
            score_val = qscore.get("overall_score", 0.0)
            grade = qscore.get("grade", "N/A")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_val,
                title={'text': f"Data Quality Score (Grade {grade})", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4f46e5"},
                    'steps': [
                        {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [60, 80], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [80, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ]
                }
            ))
            fig.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with col_q2:
            st.markdown("#### Quality Dimensions Breakdown")
            dims = qscore.get("dimensions", {})
            dim_df = pd.DataFrame([
                {"Dimension": k.replace("_", " ").title(), "Score (%)": v, "Status": "Optimal" if v >= 90 else ("Fair" if v >= 75 else "Needs Improvement")}
                for k, v in dims.items()
            ])
            st.dataframe(dim_df, use_container_width=True, hide_index=True)
            st.caption(f"**Explanation:** {qscore.get('explanation')}")

        st.divider()

        # Numerical Columns Profile
        if profile.get("numeric_columns"):
            st.markdown("#### 🔢 Numerical Columns Summary")
            num_rows = []
            for col in profile["numeric_columns"]:
                cp = profile["columns"].get(col, {})
                num_rows.append({
                    "Column": col,
                    "Min": cp.get("min"),
                    "Q1 (25%)": cp.get("q1"),
                    "Median (50%)": cp.get("median"),
                    "Mean": cp.get("mean"),
                    "Q3 (75%)": cp.get("q3"),
                    "Max": cp.get("max"),
                    "Std Dev": cp.get("std"),
                    "Missing (%)": f"{cp.get('missing_percentage')}%",
                    "Zeroes": cp.get("zero_count"),
                    "Negative Count": cp.get("negative_count")
                })
            st.dataframe(pd.DataFrame(num_rows), use_container_width=True)

        # Categorical Columns Profile
        if profile.get("categorical_columns"):
            st.markdown("#### 🔤 Categorical Columns Summary")
            cat_rows = []
            for col in profile["categorical_columns"]:
                cp = profile["columns"].get(col, {})
                cat_rows.append({
                    "Column": col,
                    "Unique Count": cp.get("unique_count"),
                    "Mode (Most Frequent)": cp.get("mode"),
                    "Missing Count": cp.get("missing_count"),
                    "Missing (%)": f"{cp.get('missing_percentage')}%"
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

            # Top value charts for categorical
            cat_to_plot = st.selectbox("Select Categorical Column to Visualize Distribution", profile["categorical_columns"])
            top_vals = profile["columns"].get(cat_to_plot, {}).get("top_frequent_values", {})
            if top_vals:
                plot_df = pd.DataFrame(list(top_vals.items()), columns=["Category", "Count"])
                fig_cat = px.bar(plot_df, x="Category", y="Count", title=f"Value Distribution for '{cat_to_plot}'", color="Count", color_continuous_scale="Viridis")
                fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cat, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Detected Quality Issues
# -------------------------------------------------------------
with tabs[2]:
    if st.session_state.raw_issues is None:
        st.info("Please upload a dataset first.")
    else:
        issues = st.session_state.raw_issues
        st.markdown(f"### 🔍 Detected Quality Issues ({len(issues)} Total)")

        # Summary Metrics
        c1, c2, c3, c4 = st.columns(4)
        auto_count = sum(1 for i in issues if i.get("routing_decision") == "AUTO_CORRECT")
        review_count = sum(1 for i in issues if i.get("routing_decision") == "SUGGEST_REVIEW")
        human_count = sum(1 for i in issues if i.get("routing_decision") == "HUMAN_REVIEW_REQUIRED")
        
        c1.metric("Total Issues", len(issues))
        c2.metric("🟢 Auto-Correctable", auto_count)
        c3.metric("🟡 Suggest Review", review_count)
        c4.metric("🔴 Human Review Required", human_count)

        st.divider()

        # Filtering Options
        f_col1, f_col2, f_col3 = st.columns(3)
        issue_types = sorted(list({i.get("issue_type") for i in issues}))
        selected_type = f_col1.selectbox("Filter by Issue Type", ["All"] + issue_types)
        
        routing_opts = ["All", "AUTO_CORRECT", "SUGGEST_REVIEW", "HUMAN_REVIEW_REQUIRED"]
        selected_routing = f_col2.selectbox("Filter by Routing Decision", routing_opts)
        
        all_cols = sorted(list({str(i.get("column")) for i in issues}))
        selected_col = f_col3.selectbox("Filter by Column", ["All"] + all_cols)

        filtered_issues = issues
        if selected_type != "All":
            filtered_issues = [i for i in filtered_issues if i.get("issue_type") == selected_type]
        if selected_routing != "All":
            filtered_issues = [i for i in filtered_issues if i.get("routing_decision") == selected_routing]
        if selected_col != "All":
            filtered_issues = [i for i in filtered_issues if str(i.get("column")) == selected_col]

        st.markdown(f"**Showing {len(filtered_issues)} matching issues:**")
        
        table_rows = []
        for i in filtered_issues:
            table_rows.append({
                "ID": i.get("issue_id"),
                "Row": i.get("row"),
                "Column": i.get("column"),
                "Issue Type": i.get("issue_type"),
                "Detected Value": str(i.get("original_value")),
                "Suggested Value": str(i.get("suggested_value")),
                "Confidence": i.get("correction_confidence"),
                "Decision": i.get("routing_decision"),
                "Reason": i.get("reason")
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

# -------------------------------------------------------------
# TAB 4: Human-in-the-Loop Review
# -------------------------------------------------------------
with tabs[3]:
    if st.session_state.raw_issues is None:
        st.info("Please upload a dataset first.")
    else:
        review_issues = [
            i for i in st.session_state.raw_issues 
            if i.get("routing_decision") in ["HUMAN_REVIEW_REQUIRED", "SUGGEST_REVIEW"]
        ]

        st.markdown("### 🧑‍💻 Human-in-the-Loop Review Queue")
        st.markdown("The system conservatively routes ambiguous values, invalid dates, and impossible domain values here. Please review and apply verified corrections.")

        if not review_issues:
            st.success("🎉 No issues require human review! All detected issues are high-confidence auto-correctable.")
        else:
            resolved_count = len(st.session_state.human_decisions)
            st.progress(min(1.0, resolved_count / max(1, len(review_issues))))
            st.caption(f"Reviewed {resolved_count} of {len(review_issues)} items")

            for idx, issue in enumerate(review_issues):
                iid = issue.get("issue_id")
                row = issue.get("row")
                col = issue.get("column")
                orig_val = issue.get("original_value")
                itype = issue.get("issue_type")
                suggested = issue.get("suggested_value")
                reason = issue.get("reason")

                with st.expander(f"📍 #{idx+1} [Row {row} | Column: {col}] - {itype.replace('_', ' ').title()}", expanded=(iid not in st.session_state.human_decisions)):
                    c_left, c_right = st.columns([2, 1])

                    with c_left:
                        st.markdown(f"**Current Raw Value:** `{orig_val}`")
                        st.markdown(f"**Detected Problem:** {reason}")
                        if suggested:
                            st.markdown(f"**Suggested Alternative:** `{suggested}` (Confidence: {issue.get('correction_confidence', 0.0)})")
                        if "llm_reasoning" in issue:
                            st.info(f"🤖 **Ollama Reasoning:** {issue['llm_reasoning']}")

                    with c_right:
                        status = st.session_state.human_decisions.get(iid, {}).get("status")
                        if status:
                            st.success(f"Decision: **{status.upper()}** (Value: `{st.session_state.human_decisions[iid].get('corrected_value')}`)")
                            if st.button("Edit Decision", key=f"re_{iid}"):
                                del st.session_state.human_decisions[iid]
                                st.rerun()
                        else:
                            st.markdown("**Enter Verified Correction:**")
                            user_input = st.text_input("Corrected Value", value=str(suggested or ""), key=f"inp_{iid}")
                            
                            btn_c1, btn_c2 = st.columns(2)
                            if btn_c1.button("✅ Accept", key=f"acc_{iid}", use_container_width=True):
                                is_valid, clean_val, err = validate_user_correction_input(col, itype, user_input)
                                if is_valid:
                                    st.session_state.human_decisions[iid] = {
                                        "status": "accepted",
                                        "row": row,
                                        "column": col,
                                        "original_value": orig_val,
                                        "corrected_value": clean_val,
                                        "issue_type": itype,
                                        "notes": "Verified by user in review queue"
                                    }
                                    st.success("Correction saved!")
                                    st.rerun()
                                else:
                                    st.error(err)

                            if btn_c2.button("❌ Reject / Skip", key=f"rej_{iid}", use_container_width=True):
                                st.session_state.human_decisions[iid] = {
                                    "status": "rejected",
                                    "row": row,
                                    "column": col,
                                    "original_value": orig_val,
                                    "corrected_value": orig_val,
                                    "issue_type": itype,
                                    "notes": "Rejected by user"
                                }
                                st.warning("Marked as rejected.")
                                st.rerun()

# -------------------------------------------------------------
# TAB 5: Automated Cleaning Pipeline Execution
# -------------------------------------------------------------
with tabs[4]:
    if st.session_state.working_df is None or st.session_state.raw_issues is None:
        st.info("Please upload a dataset first.")
    else:
        st.markdown("### ⚡ Configure & Run Cleaning Pipeline")

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            st.markdown("**Missing Numerical Imputation**")
            num_strat = st.selectbox("Strategy", ["median", "mean", "knn", "constant", "skip"], index=0)
        
        with c_opt2:
            st.markdown("**Missing Categorical Imputation**")
            cat_strat = st.selectbox("Strategy", ["mode", "constant", "skip"], index=0)

        with c_opt3:
            st.markdown("**Numerical Outlier Handling**")
            outlier_strat = st.selectbox("Policy", ["flag_only", "cap", "remove"], index=0, help="'flag_only' keeps values unchanged and documents audit acknowledgement.")

        st.markdown("**Deterministic Cleaning Flags**")
        fl_1, fl_2, fl_3 = st.columns(3)
        rm_dups = fl_1.checkbox("Remove Exact Duplicate Rows", value=True)
        norm_case = fl_2.checkbox("Normalize Whitespace & Casing", value=True)
        norm_typos = fl_3.checkbox("Auto-Correct High-Confidence Typos (>=0.95)", value=True)

        st.divider()

        human_count = len(st.session_state.human_decisions)
        st.write(f"💼 **Human Review Corrections Ready to Apply:** {human_count} items")

        if st.button("🚀 Execute Deterministic Cleaning Pipeline", type="primary", use_container_width=True):
            with st.spinner("Executing cleaning transformations and compiling audit logs..."):
                cleaning_opts = {
                    "missing_numeric_strategy": num_strat,
                    "missing_categorical_strategy": cat_strat,
                    "outlier_strategy": outlier_strat,
                    "remove_exact_duplicates": rm_dups,
                    "normalize_whitespace": norm_case,
                    "normalize_text_case": norm_case,
                    "impute_missing": (num_strat != "skip" or cat_strat != "skip")
                }
                
                cleaned_df, audit_trail = execute_cleaning_pipeline(
                    st.session_state.working_df,
                    st.session_state.raw_issues,
                    user_corrections=st.session_state.human_decisions,
                    cleaning_options=cleaning_opts
                )
                
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_audit = audit_trail

                # Validate Cleaned State
                af_prof, af_issues, af_score = validate_cleaned_dataset(cleaned_df)
                st.session_state.after_profile = af_prof
                st.session_state.after_issues = af_issues
                st.session_state.after_quality_score = af_score

            st.success(f"✨ Cleaning Completed! Applied {len(audit_trail)} transformations.")
            st.balloons()

        if st.session_state.cleaned_df is not None:
            st.markdown("#### 📋 Cleaned Dataset Preview (First 15 Rows)")
            st.dataframe(st.session_state.cleaned_df.head(15), use_container_width=True)

# -------------------------------------------------------------
# TAB 6: Validation & Before/After Deltas
# -------------------------------------------------------------
with tabs[5]:
    if st.session_state.cleaned_df is None:
        st.info("Run the cleaning pipeline in Tab 5 to view before vs after validation.")
    else:
        st.markdown("### 📈 Scientific Validation & Before-vs-After Progression")

        comparison = compute_before_after_comparison(
            st.session_state.raw_profile,
            st.session_state.after_profile,
            st.session_state.raw_issues,
            st.session_state.after_issues,
            st.session_state.raw_quality_score,
            st.session_state.after_quality_score
        )

        # High Level Metric Delta Cards
        m = comparison["metrics"]
        d1, d2, d3, d4 = st.columns(4)
        
        q_delta = m["quality_score"]["delta"]
        q_sign = "+" if q_delta >= 0 else ""
        d1.metric("Quality Score", f"{m['quality_score']['after']}/100", delta=f"{q_sign}{q_delta} pts")
        
        miss_delta = m["missing_cells"]["delta"]
        d2.metric("Missing Cells", f"{m['missing_cells']['after']}", delta=f"{miss_delta}", delta_color="inverse")
        
        dup_delta = m["duplicate_rows"]["delta"]
        d3.metric("Duplicate Rows", f"{m['duplicate_rows']['after']}", delta=f"{dup_delta}", delta_color="inverse")

        iss_delta = m["total_issues"]["delta"]
        d4.metric("Active Issues", f"{m['total_issues']['after']}", delta=f"{iss_delta}", delta_color="inverse")

        st.divider()

        # Quality Dimension Comparison Bar Chart
        dim_deltas = comparison["dimension_deltas"]
        dim_plot_data = []
        for d_name, vals in dim_deltas.items():
            dim_plot_data.append({"Dimension": d_name.replace("_", " ").title(), "Stage": "Before Cleaning", "Score (%)": vals["before"]})
            dim_plot_data.append({"Dimension": d_name.replace("_", " ").title(), "Stage": "After Cleaning", "Score (%)": vals["after"]})
        
        dim_fig = px.bar(
            pd.DataFrame(dim_plot_data),
            x="Dimension",
            y="Score (%)",
            color="Stage",
            barmode="group",
            title="Quality Dimensions: Before vs After",
            color_discrete_map={"Before Cleaning": "#f59e0b", "After Cleaning": "#10b981"}
        )
        dim_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(dim_fig, use_container_width=True)

        st.markdown("#### 🎯 Issue Resolution Summary")
        st.dataframe(pd.DataFrame(comparison["issue_type_comparison"]), use_container_width=True)

# -------------------------------------------------------------
# TAB 7: Evaluation Framework & Downstream ML
# -------------------------------------------------------------
with tabs[6]:
    st.markdown("### 🧪 Scientific Data Corruption & ML Evaluation Framework")
    st.markdown("Inject known, ground-truth synthetic corruptions into a clean reference dataset to scientifically benchmark detection precision, recall, F1, and restoration accuracy.")

    eval_col1, eval_col2 = st.columns([1, 2])
    with eval_col1:
        st.markdown("**Ground-Truth Corruption Parameters**")
        corrupt_missing = st.slider("Missing Value Rate", 0.0, 0.20, 0.05, step=0.01)
        corrupt_dups = st.slider("Duplicate Rows", 0, 10, 3)
        corrupt_outliers = st.slider("Numerical Outliers", 0, 10, 4)
        corrupt_invalids = st.slider("Invalid Ages/Ranges", 0, 10, 3)
        corrupt_dates = st.slider("Invalid Dates", 0, 10, 3)
        corrupt_typos = st.slider("Spelling Typos", 0, 10, 4)

        if st.button("🔥 Run Ground-Truth Benchmark", type="primary", use_container_width=True):
            clean_ref_path = os.path.join(os.path.dirname(__file__), "sample_data", "clean_benchmark_dataset.csv")
            if os.path.exists(clean_ref_path):
                clean_ref_df = pd.read_csv(clean_ref_path)
                corrupter = DataCorruptionEngine(seed=42)
                
                with st.spinner("Injecting controlled corruptions and running evaluation pipeline..."):
                    corrupted_df, ground_truth = corrupter.inject_corruptions(
                        clean_ref_df,
                        missing_rate=corrupt_missing,
                        duplicate_count=corrupt_dups,
                        outlier_count=corrupt_outliers,
                        invalid_numeric_count=corrupt_invalids,
                        invalid_date_count=corrupt_dates,
                        spelling_typo_count=corrupt_typos
                    )

                    # Detect Issues
                    detected = detect_all_issues(corrupted_df)
                    reasoner = ReasoningEngine(ollama_client)
                    enriched_det = reasoner.enrich_issues(detected, corrupted_df, use_ollama=is_ollama_online)

                    # Detection Metrics
                    det_metrics = evaluate_detection_performance(ground_truth, enriched_det)

                    # Clean and evaluate correction
                    eval_cleaned, eval_audit = execute_cleaning_pipeline(corrupted_df, enriched_det)
                    corr_metrics = evaluate_correction_performance(ground_truth, eval_cleaned, eval_audit)

                    # Downstream ML
                    ml_metrics = evaluate_downstream_ml(corrupted_df, eval_cleaned, target_column="Churn", random_seed=42)

                    st.session_state.eval_benchmark_results = {
                        "det_metrics": det_metrics,
                        "corr_metrics": corr_metrics,
                        "ml_metrics": ml_metrics,
                        "total_ground_truth": len(ground_truth),
                        "total_detected": len(enriched_det)
                    }
                st.success("Benchmark completed successfully!")

    with eval_col2:
        if st.session_state.eval_benchmark_results:
            bench = st.session_state.eval_benchmark_results
            det = bench["det_metrics"]
            corr = bench["corr_metrics"]
            ml = bench["ml_metrics"]

            st.markdown("#### 🎯 Detection Performance Metrics")
            o = det["overall"]
            ep1, ep2, ep3, ep4 = st.columns(4)
            ep1.metric("Precision", f"{o['precision'] * 100:.1f}%")
            ep2.metric("Recall", f"{o['recall'] * 100:.1f}%")
            ep3.metric("F1 Score", f"{o['f1_score']:.3f}")
            ep4.metric("True Positives", f"{o['true_positives']} / {o['total_injected_errors']}")

            # Breakdown by issue type
            by_type = det.get("by_issue_type", {})
            if by_type:
                bt_df = pd.DataFrame([
                    {
                        "Issue Type": k.replace("_", " ").title(),
                        "TP": v["true_positives"],
                        "FP": v["false_positives"],
                        "FN": v["false_negatives"],
                        "Precision": f"{v['precision']*100:.1f}%",
                        "Recall": f"{v['recall']*100:.1f}%",
                        "F1": v["f1_score"]
                    }
                    for k, v in by_type.items()
                ])
                st.dataframe(bt_df, use_container_width=True)

            st.markdown("#### 🛠️ Restoration & Correction Accuracy")
            cp1, cp2, cp3 = st.columns(3)
            cp1.metric("Auto-Correction Accuracy", f"{corr['automatic_correction_accuracy']}%")
            cp2.metric("Auto-Restored Values", f"{corr['automatic_corrections_restored']} / {corr['automatic_corrections_attempted']}")
            cp3.metric("Unresolved Issues", corr["unresolved_injected_issues"])

            st.markdown("#### 🤖 Downstream Machine Learning Benchmark")
            if "corrupted_metrics" in ml and "cleaned_metrics" in ml:
                ml_c = ml["corrupted_metrics"]
                ml_cl = ml["cleaned_metrics"]
                ml_table = pd.DataFrame([
                    {"Metric": "Accuracy", "Corrupted Dataset": f"{ml_c['accuracy']*100:.2f}%", "Cleaned Dataset": f"{ml_cl['accuracy']*100:.2f}%"},
                    {"Metric": "F1 (Weighted)", "Corrupted Dataset": f"{ml_c['f1_weighted']:.4f}", "Cleaned Dataset": f"{ml_cl['f1_weighted']:.4f}"},
                    {"Metric": "Precision", "Corrupted Dataset": f"{ml_c['precision_weighted']:.4f}", "Cleaned Dataset": f"{ml_cl['precision_weighted']:.4f}"},
                    {"Metric": "Recall", "Corrupted Dataset": f"{ml_c['recall_weighted']:.4f}", "Cleaned Dataset": f"{ml_cl['recall_weighted']:.4f}"}
                ])
                st.dataframe(ml_table, use_container_width=True)
                st.caption(f"ℹ️ *{ml.get('disclaimer')}*")
        else:
            st.info("Click 'Run Ground-Truth Benchmark' to execute the scientific evaluation.")

# -------------------------------------------------------------
# TAB 8: Audit Trail & Export
# -------------------------------------------------------------
with tabs[7]:
    if st.session_state.cleaned_df is None:
        st.info("Clean a dataset in Tab 5 first to export results.")
    else:
        st.markdown("### 📥 Audit Log & Cleaned Dataset Export")

        logger_inst = AuditLogger(dataset_name=st.session_state.dataset_meta.get("filename", "dataset"))
        logger_inst.log_detected_issues(st.session_state.raw_issues)
        logger_inst.log_applied_actions(st.session_state.cleaning_audit)
        audit_report = logger_inst.generate_audit_report()

        exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)

        # 1. Download CSV
        csv_buffer = io.StringIO()
        st.session_state.cleaned_df.to_csv(csv_buffer, index=False)
        exp_c1.download_button(
            "📄 Download Cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 2. Download Excel XLSX
        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            st.session_state.cleaned_df.to_excel(writer, index=False, sheet_name="CleanedData")
        exp_c2.download_button(
            "📊 Download Cleaned XLSX",
            data=xlsx_buffer.getvalue(),
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # 3. Download JSON Audit Trail
        json_str = logger_inst.to_json()
        exp_c3.download_button(
            "📜 Download Audit Log (JSON)",
            data=json_str,
            file_name="cleaning_audit_log.json",
            mime="application/json",
            use_container_width=True
        )

        # 4. Download HTML Audit Report
        html_report = generate_html_report(
            dataset_name=st.session_state.dataset_meta.get("filename", "dataset"),
            before_profile=st.session_state.raw_profile,
            after_profile=st.session_state.after_profile,
            before_score=st.session_state.raw_quality_score,
            after_score=st.session_state.after_quality_score,
            audit_log=audit_report
        )
        exp_c4.download_button(
            "🌐 Download HTML Audit Report",
            data=html_report,
            file_name="data_quality_report.html",
            mime="text/html",
            use_container_width=True
        )

        st.divider()
        st.markdown("#### 📜 Granular Audit Log Preview")
        st.json(audit_report, expanded=False)
