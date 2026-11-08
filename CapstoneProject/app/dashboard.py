import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

# ------------------------------------------------------------------
# Streamlit Page & Theme Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Resume-JD Semantic Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for HR Badges and Metric Cards
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-high {
        background-color: #28a745; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold;
    }
    .badge-moderate {
        background-color: #ffc107; color: black; padding: 4px 12px; border-radius: 12px; font-weight: bold;
    }
    .badge-low {
        background-color: #dc3545; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar & API Configuration
# ------------------------------------------------------------------
st.sidebar.title("🛠️ System Settings")
API_BASE_URL = st.sidebar.text_input("FastAPI Service Endpoint", value="http://localhost:8000")

# API Health Status Check
try:
    health_res = requests.get(f"{API_BASE_URL}/health", timeout=55)
    if health_res.status_code == 200:
        st.sidebar.success(f"✅ Connected to Inference API: {health_res.status_code}")
    else:
        st.sidebar.warning(f"⚠️ API connection warning: {health_res.status_code}")
except Exception:
    st.sidebar.error("❌ API Service Offline")

st.sidebar.markdown("---")
st.sidebar.subheader("Matching Parameters")
top_k = st.sidebar.slider("Top Candidates to Display", min_value=1, max_value=20, value=5)
similarity_threshold = st.sidebar.slider("Skill Match Threshold", min_value=0.4, max_value=0.9, value=0.65, step=0.05)


# ------------------------------------------------------------------
# Main Header & Layout
# ------------------------------------------------------------------
st.title("🎯 Candidate Fit Evaluation & Skill Gap Analytics")
st.caption("AI-Powered Resume to Job Description Matcher using Sentence-BERT & Fine-Tuned TensorFlow Classifiers.")

col_jd, col_res = st.columns([1, 1])

with col_jd:
    st.subheader("1. Target Job Description (JD)")
    sample_jd = """Senior Data Scientist needed with strong background in Python, TensorFlow, PyTorch, and MLOps. 
Required skills include SQL, Natural Language Processing (NLP), Docker containerization, REST API development with FastAPI, 
and Kubernetes orchestration. Experience with Sentence-Transformers and cloud deployment on AWS is highly desirable."""
    
    jd_text = st.text_area("Paste Job Requirements:", value=sample_jd, height=220)

with col_res:
    st.subheader("2. Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Upload candidate resumes (.txt format):", 
        type=["txt"], 
        accept_multiple_files=True
    )
    
    # Pre-load sample candidate data if no files uploaded
    use_sample = st.checkbox("Use Sample Candidates for Quick Demo", value=True if not uploaded_files else False)

# ------------------------------------------------------------------
# Sample Data Handler
# ------------------------------------------------------------------
candidates_payload: List[Dict[str, str]] = []

if uploaded_files:
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        candidates_payload.append({"candidate_id": file.name, "resume_text": content})
elif use_sample:
    candidates_payload = [
        {
            "candidate_id": "Alex_Developer_Resume.txt",
            "resume_text": "Experienced Python Engineer specializing in Machine Learning, TensorFlow, NLP, Docker, FastAPI, and SQL databases. Strong background in Docker containers and PyTorch models."
        },
        {
            "candidate_id": "Jordan_DataAnalyst_Resume.txt",
            "resume_text": "Data Analyst with 4 years experience in SQL queries, Tableau reporting, Excel dashboards, basic Python scripting, and business intelligence analytics."
        },
        {
            "candidate_id": "Taylor_DevOpsEngine_Resume.txt",
            "resume_text": "Senior Cloud and DevOps Architect skilled in Kubernetes, Docker, AWS infrastructure, Terraform, CI/CD pipelines, and Python automation."
        }
    ]

# ------------------------------------------------------------------
# Ingestion & Evaluation Trigger
# ------------------------------------------------------------------
if st.button("🚀 Process & Rank Candidates", type="primary", use_container_width=True):
    if not jd_text.strip():
        st.error("Please provide a valid Job Description text.")
    elif not candidates_payload:
        st.error("Please upload candidate resumes or check the sample demo option.")
    else:
        with st.spinner("Analyzing semantics, classifying fit, and extracting skill gaps..."):
            try:
                # Call FastAPI batch endpoint
                response = requests.post(
                    f"{API_BASE_URL}/rank",
                    json={"job_description": jd_text, "resumes": candidates_payload},
                    timeout=15
                )
                
                if response.status_code == 200:
                    st.session_state["ranking_results"] = response.json()
                    st.success("Batch Analysis Complete!")
                else:
                    st.error(f"Inference error ({response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to API backend: {str(e)}")


# ------------------------------------------------------------------
# Visualization & Results Dashboard
# ------------------------------------------------------------------
if "ranking_results" in st.session_state:
    data = st.session_state["ranking_results"]
    rankings = data.get("rankings", [])
    
    st.markdown("---")
    st.header("📊 Evaluation Overview")
    
    # Key Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    total_eval = len(rankings)
    high_fits = sum(1 for r in rankings if r["fit_category"] == "High Fit")
    top_score = max([r["cosine_similarity"] for r in rankings]) if rankings else 0.0
    top_candidate = rankings[0]["candidate_id"] if rankings else "N/A"
    
    m1.metric("Total Candidates Evaluated", total_eval)
    m2.metric("High Fit Shortlist", high_fits)
    m3.metric("Top Semantic Score", f"{top_score:.2%}")
    m4.metric("Rank #1 Candidate", top_candidate)
    
    # --------------------------------------------------------------
    # Candidate Comparison Charts
    # --------------------------------------------------------------
    df_rankings = pd.DataFrame([
        {
            "Rank": r["rank"],
            "Candidate": r["candidate_id"],
            "Cosine Similarity": r["cosine_similarity"],
            "Confidence Score": r["confidence_score"],
            "Fit Category": r["fit_category"],
            "Matched Skills Count": len(r["skill_gap_analysis"]["matched_skills"]),
            "Missing Skills Count": len(r["skill_gap_analysis"]["missing_skills"])
        }
        for r in rankings
    ]).head(top_k)

    col_chart1, col_chart2 = st.columns([1.2, 1])

    with col_chart1:
        st.subheader("Candidate Ranking & Semantic Similarity")
        fig_bar = px.bar(
            df_rankings,
            x="Cosine Similarity",
            y="Candidate",
            orientation="h",
            color="Fit Category",
            color_discrete_map={
                "High Fit": "#28a745",
                "Moderate Fit": "#ffc107",
                "Low Fit": "#dc3545"
            },
            text_auto=".2f",
            title="Semantic Cosine Match Score by Candidate"
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.subheader("Skill Match Ratio Comparison")
        fig_skills = px.bar(
            df_rankings,
            x="Candidate",
            y=["Matched Skills Count", "Missing Skills Count"],
            title="Matched vs Missing Skill Breakdown",
            barmode="group",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        fig_skills.update_layout(height=350)
        st.plotly_chart(fig_skills, use_container_width=True)

    # --------------------------------------------------------------
    # Candidate Deep-Dive & Detailed Skill Gap Analysis
    # --------------------------------------------------------------
    st.markdown("---")
    st.header("🔎 Individual Candidate Deep-Dive & Skill Gap Analysis")
    
    selected_candidate_id = st.selectbox(
        "Select candidate to inspect detailed skill matrix:",
        options=[r["candidate_id"] for r in rankings]
    )
    
    candidate_detail = next(r for r in rankings if r["candidate_id"] == selected_candidate_id)
    skills_data = candidate_detail["skill_gap_analysis"]

    # Header Badges & Key Stats
    c1, c2, c3 = st.columns(3)
    fit_cat = candidate_detail["fit_category"]
    badge_class = "badge-high" if fit_cat == "High Fit" else ("badge-moderate" if fit_cat == "Moderate Fit" else "badge-low")
    
    c1.markdown(f"**Predicted Category:** <span class='{badge_class}'>{fit_cat}</span>", unsafe_allow_html=True)
    c2.metric("Model Confidence Score", f"{candidate_detail['confidence_score']:.2%}")
    c3.metric("Dense Cosine Match", f"{candidate_detail['cosine_similarity']:.4f}")

    col_match, col_miss = st.columns(2)

    with col_match:
        st.subheader("✅ Matched Skills")
        matched_list = skills_data["matched_skills"]
        if matched_list:
            df_matched = pd.DataFrame(matched_list)
            fig_match = px.bar(
                df_matched,
                x="score",
                y="skill",
                orientation="h",
                color="score",
                color_continuous_scale="Greens",
                range_x=[0, 1.0],
                title="Matched Skills & Similarity Confidence"
            )
            fig_match.update_layout(yaxis=dict(autorange="reversed"), height=300)
            st.plotly_chart(fig_match, use_container_width=True)
        else:
            st.info("No explicit technical skill matches detected.")

    with col_miss:
        st.subheader("❌ Missing / Unmatched JD Skills")
        missing_list = skills_data["missing_skills"]
        if missing_list:
            df_missing = pd.DataFrame(missing_list)
            fig_miss = px.bar(
                df_missing,
                x="score",
                y="skill",
                orientation="h",
                color="score",
                color_continuous_scale="Oranges",
                range_x=[0, 1.0],
                title="Missing Required Skills (Lowest Semantic Overlap)"
            )
            fig_miss.update_layout(yaxis=dict(autorange="reversed"), height=300)
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("Candidate satisfies all key JD skill requirements!")

# ------------------------------------------------------------------
# Audit Log & Failure Case Documentation
# ------------------------------------------------------------------
with st.expander("📌 System Limitations & Known Edge Cases"):
    st.markdown("""
    - **Over-qualification & Length Bias:** Candidates with extensive general history may exhibit slightly diluted embedding cosine scores compared to hyper-concise tailored resumes.
    - **Abbreviation Mismatch:** Non-standard acronyms (e.g., K8s for Kubernetes, GCP for Google Cloud) require pre-defined term expansion or domain-specific vocabulary fine-tuning.
    - **Human-in-the-Loop Protocol:** This dashboard is an assistance tool for HR screening and does not perform autonomous hiring or rejection decisions.
    """)