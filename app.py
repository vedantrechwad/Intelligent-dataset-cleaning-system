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
from src.intelligence.domain_inferencer import infer_dynamic_domain_rules
from src.review.human_review import HumanReviewManager, validate_user_correction_input
from src.cleaning.pipeline import execute_cleaning_pipeline
from src.validation.validator import validate_cleaned_dataset
from src.validation.before_after import compute_before_after_comparison
from src.reporting.audit_logger import AuditLogger
from src.reporting.report_generator import generate_html_report

# Configure Streamlit Page
st.set_page_config(
    page_title="Intelligent Data Quality & Auto-Cleaning System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, mature, corporate UI aesthetics
st.markdown("""
<style>
    :root {
        --primary-color: #0f172a;
        --accent-color: #3b82f6;
        --text-color: #e2e8f0;
        --bg-panel: #1e293b;
    }
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
        margin-bottom: 2rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 1rem;
    }
    .metric-card {
        background: var(--bg-panel);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    .badge-auto {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-review {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-human {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 4px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
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
    "1. Ingestion & Preview",
    "2. Data Profiling",
    "3. Detected Issues",
    "4. Human Review",
    "5. Cleaning Pipeline",
    "6. Validation & Deltas",
    "7. Audit Trail & Export"
])

# -------------------------------------------------------------
# TAB 1: Ingestion & Upload
# -------------------------------------------------------------
with tabs[0]:
    st.markdown("### 📤 Upload Tabular Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Original raw data is preserved strictly without modification."
    )

    if uploaded_file is not None:
        try:
            orig_df, work_df, meta = load_dataset(uploaded_file, filename=uploaded_file.name)
            st.session_state.raw_df = orig_df
            st.session_state.working_df = work_df
            st.session_state.dataset_meta = meta
            if st.session_state.raw_profile is None:
                with st.status("Processing Dataset...", expanded=True) as status:
                    st.write("Extracting logical schema and computing memory footprints...")
                    st.session_state.schema_info = detect_dataset_schema(work_df)
                    
                    st.write("Dynamically inferring dataset domain constraints with LLM...")
                    st.session_state.dynamic_rules = infer_dynamic_domain_rules(work_df, st.session_state.schema_info, ollama_client)
                    
                    st.write("Executing statistical profiling across numerical distributions...")
                    st.session_state.raw_profile = profile_dataset(work_df, st.session_state.schema_info)
                    
                    st.write("Scanning for semantic outliers and domain boundary violations...")
                    raw_issues = detect_all_issues(work_df, dynamic_rules=st.session_state.dynamic_rules)
                    
                    st.write("Initializing local LLM inference for ambiguous categorical normalization...")
                    reasoning_engine = ReasoningEngine(ollama_client)
                    st.session_state.raw_issues = reasoning_engine.enrich_issues(raw_issues, work_df, use_ollama=is_ollama_online)
                    
                    st.write("Computing multi-dimensional quality scores...")
                    st.session_state.raw_quality_score = compute_quality_score(work_df, detected_issues=st.session_state.raw_issues)
                    
                    status.update(label="Processing Complete!", state="complete", expanded=False)
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
        st.info("👆 Please upload a CSV/XLSX file to begin.")

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
                "Reason": i.get("reason"),
                "AI Explanation": i.get("simple_explanation", "")
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
                        
                        st.info(f"💡 **AI Explanation:** {issue.get('simple_explanation', 'Unavailable')}")
                        
                        if "llm_reasoning" in issue:
                            st.success(f"🤖 **Deep Reasoning:** {issue['llm_reasoning']}")

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
            with st.status("Executing Cleaning Pipeline...", expanded=True) as status:
                st.write("Configuring missing imputation and outlier policies...")
                cleaning_opts = {
                    "missing_numeric_strategy": num_strat,
                    "missing_categorical_strategy": cat_strat,
                    "outlier_strategy": outlier_strat,
                    "remove_exact_duplicates": rm_dups,
                    "normalize_whitespace": norm_case,
                    "normalize_text_case": norm_case,
                    "impute_missing": (num_strat != "skip" or cat_strat != "skip")
                }
                
                st.write("Applying deterministic transformations and generating audit logs...")
                cleaned_df, audit_trail = execute_cleaning_pipeline(
                    st.session_state.working_df,
                    st.session_state.raw_issues,
                    user_corrections=st.session_state.human_decisions,
                    cleaning_options=cleaning_opts
                )
                
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_audit = audit_trail

                st.write("Validating post-cleaning dataset state and computing delta scores...")
                # Validate Cleaned State
                af_prof, af_issues, af_score = validate_cleaned_dataset(cleaned_df)
                st.session_state.after_profile = af_prof
                st.session_state.after_issues = af_issues
                st.session_state.after_quality_score = af_score
                
                status.update(label="Cleaning Complete!", state="complete", expanded=False)

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
# TAB 7: Audit Trail & Export
# -------------------------------------------------------------
with tabs[6]:
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
