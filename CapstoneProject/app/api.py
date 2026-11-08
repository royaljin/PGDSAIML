import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
import spacy
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mlops-resume-matcher")

# Global ML model container populated during FastAPI lifespan startup
ml_models: Dict[str, Any] = {}

LABEL_MAPPING: Dict[int, str] = {
    0: "Low Fit",
    1: "Moderate Fit",
    2: "High Fit",
}


# ------------------------------------------------------------------
# Text Preprocessing, Similarity & Skill Extraction Helpers
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Sanitizes raw resume and job description text."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s\.\,\+\#\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_cosine_similarity(r_emb: np.ndarray, j_emb: np.ndarray) -> np.ndarray:
    """Computes row-wise cosine similarity as a (N, 1) float32 array."""
    r_norm = np.linalg.norm(r_emb, axis=1, keepdims=True)
    j_norm = np.linalg.norm(j_emb, axis=1, keepdims=True)
    denom = np.maximum(r_norm * j_norm, 1e-12)
    sim = np.sum(r_emb * j_emb, axis=1, keepdims=True) / denom
    return sim.astype(np.float32)


def extract_skill_gaps(
    resume_text: str,
    jd_text: str,
    sbert_model: SentenceTransformer,
    nlp_model: Any,
    similarity_threshold: float = 0.65,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts candidate skill phrases from JD noun chunks and matches against resume phrases."""
    jd_doc = nlp_model(clean_text(jd_text))
    res_doc = nlp_model(clean_text(resume_text))

    jd_keywords = sorted(
        {
            chunk.text.strip().lower()
            for chunk in jd_doc.noun_chunks
            if 1 <= len(chunk.text.split()) <= 3 and any(c.isalnum() for c in chunk.text)
        }
    )
    res_keywords = sorted(
        {
            chunk.text.strip().lower()
            for chunk in res_doc.noun_chunks
            if 1 <= len(chunk.text.split()) <= 3 and any(c.isalnum() for c in chunk.text)
        }
    )

    if not jd_keywords:
        return {"matched_skills": [], "missing_skills": []}

    if not res_keywords:
        return {
            "matched_skills": [],
            "missing_skills": [{"skill": kw, "score": 0.0} for kw in jd_keywords],
        }

    jd_embs = sbert_model.encode(jd_keywords, convert_to_tensor=True)
    res_embs = sbert_model.encode(res_keywords, convert_to_tensor=True)
    sim_matrix = util.cos_sim(jd_embs, res_embs)

    matched_skills: List[Dict[str, Any]] = []
    missing_skills: List[Dict[str, Any]] = []

    for idx, jd_kw in enumerate(jd_keywords):
        max_sim = float(sim_matrix[idx].max())
        item = {"skill": jd_kw, "score": round(max_sim, 4)}
        if max_sim >= similarity_threshold:
            matched_skills.append(item)
        else:
            missing_skills.append(item)

    matched_skills.sort(key=lambda x: x["score"], reverse=True)
    missing_skills.sort(key=lambda x: x["score"], reverse=True)

    return {"matched_skills": matched_skills, "missing_skills": missing_skills}


def build_fallback_classifier(embedding_dim: int = 384) -> tf.keras.Model:
    #Builds a bytecode-free Keras 3 fallback model matching dictionary input keys.
    r_in = tf.keras.layers.Input(shape=(embedding_dim,), name="resume_embedding")
    j_in = tf.keras.layers.Input(shape=(embedding_dim,), name="jd_embedding")
    s_in = tf.keras.layers.Input(shape=(1,), name="cosine_similarity")

    diff = tf.keras.layers.Subtract()([r_in, j_in])
    neg_diff = tf.keras.layers.Rescaling(-1.0)(diff)
    abs_diff = tf.keras.layers.Maximum()([diff, neg_diff])
    mult = tf.keras.layers.Multiply()([r_in, j_in])

    x = tf.keras.layers.Concatenate()([r_in, j_in, abs_diff, mult, s_in])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(3, activation="softmax", name="fit_category")(x)

    model = tf.keras.models.Model(
        inputs={
            "resume_embedding": r_in,
            "jd_embedding": j_in,
            "cosine_similarity": s_in,
        },
        outputs=out,
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


# ------------------------------------------------------------------
# FastAPI Lifespan Engine (Startup & Shutdown)
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads Sentence-BERT, SpaCy, and Keras 3 artifacts into memory on startup."""
    logger.info(
        f"Starting Resume-JD Matcher API | Python {sys.version.split()[0]} | TensorFlow {tf.__version__}"
    )

    # 1. Load Sentence-BERT Encoder
    sbert_name = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")
    logger.info(f"Loading SentenceTransformer model: {sbert_name}")
    ml_models["sbert"] = SentenceTransformer(sbert_name)

    # 2. Load SpaCy NER / Noun-Chunk Pipeline
    logger.info("Loading SpaCy pipeline: en_core_web_sm")
    ml_models["spacy"] = spacy.load("en_core_web_sm")

    # 3. Load Portable Keras v3 Classification Artifact
    model_path = os.getenv("MODEL_ARTIFACT_PATH", "/app/artifacts/fit_classifier.keras")
    if os.path.exists(model_path):
        logger.info(f"Loading Keras v3 classification model from {model_path}")
        ml_models["classifier"] = tf.keras.models.load_model(model_path)
    else:
        logger.warning(
            f"Artifact not found at '{model_path}'. Initializing fallback network for API stability."
        )
        ml_models["classifier"] = build_fallback_classifier(embedding_dim=384)

    logger.info("All ML models loaded and ready for inference.")
    yield

    ml_models.clear()
    logger.info("ML artifacts cleared from memory.")


# ------------------------------------------------------------------
# Pydantic v2 Request & Response Schemas
# ------------------------------------------------------------------
class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20, description="Full text of candidate resume.")
    job_description: str = Field(..., min_length=20, description="Full text of Job Description.")


class CandidateResumeItem(BaseModel):
    candidate_id: str = Field(..., min_length=1, description="Unique identifier or filename for candidate.")
    resume_text: str = Field(..., min_length=20, description="Candidate resume body text.")


class BatchMatchRequest(BaseModel):
    job_description: str = Field(..., min_length=20, description="Target Job Description text.")
    resumes: List[CandidateResumeItem] = Field(
        ..., min_length=1, description="List of candidate resumes to evaluate and rank."
    )


class SkillItem(BaseModel):
    skill: str
    score: float


class SkillGapReport(BaseModel):
    matched_skills: List[SkillItem]
    missing_skills: List[SkillItem]


class MatchResponse(BaseModel):
    fit_category: str
    confidence_score: float
    cosine_similarity: float
    skill_gap_analysis: SkillGapReport


class CandidateRankResponse(BaseModel):
    candidate_id: str
    rank: int
    fit_category: str
    confidence_score: float
    cosine_similarity: float
    skill_gap_analysis: SkillGapReport


class BatchMatchResponse(BaseModel):
    job_description_length: int
    total_candidates_evaluated: int
    rankings: List[CandidateRankResponse]


# ------------------------------------------------------------------
# FastAPI Application & Endpoints
# ------------------------------------------------------------------
app = FastAPI(
    title="Resume-JD Matcher MLOps API",
    description="REST service for semantic resume-to-job-description classification, candidate ranking, and skill-gap extraction.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {
        "service": "Resume-JD Matcher MLOps API",
        "status": "online",
        "endpoints": {
            "interactive_docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "rank": "/rank"
        }
    }

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Liveness and readiness check for container orchestration."""
    required_keys = ["sbert", "spacy", "classifier"]
    if not all(k in ml_models for k in required_keys):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models are not fully loaded into memory.",
        )
    return {
        "status": "healthy",
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "loaded_models": list(ml_models.keys()),
    }


@app.post("/predict", response_model=MatchResponse, status_code=status.HTTP_200_OK)
def predict_single_match(payload: MatchRequest) -> MatchResponse:
    """Evaluates a single Resume and Job Description pair."""
    try:
        clean_res = clean_text(payload.resume_text)
        clean_jd = clean_text(payload.job_description)

        sbert: SentenceTransformer = ml_models["sbert"]
        classifier: tf.keras.Model = ml_models["classifier"]

        # Encode texts as float32 numpy arrays
        r_emb = np.asarray(sbert.encode([clean_res], convert_to_numpy=True), dtype=np.float32)
        j_emb = np.asarray(sbert.encode([clean_jd], convert_to_numpy=True), dtype=np.float32)
        sim = compute_cosine_similarity(r_emb, j_emb)

        # Predict using dictionary input keys matching fit_classifier.keras
        model_inputs = {
            "resume_embedding": r_emb,
            "jd_embedding": j_emb,
            "cosine_similarity": sim,
        }
        probs = classifier.predict(model_inputs, verbose=0)[0]
        class_idx = int(np.argmax(probs))

        skill_analysis = extract_skill_gaps(
            resume_text=clean_res,
            jd_text=clean_jd,
            sbert_model=sbert,
            nlp_model=ml_models["spacy"],
        )

        return MatchResponse(
            fit_category=LABEL_MAPPING.get(class_idx, "Unknown"),
            confidence_score=round(float(probs[class_idx]), 4),
            cosine_similarity=round(float(sim[0][0]), 4),
            skill_gap_analysis=SkillGapReport(**skill_analysis),
        )

    except Exception as exc:
        logger.error(f"Error during single match inference: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {exc}",
        )


@app.post("/rank", response_model=BatchMatchResponse, status_code=status.HTTP_200_OK)
def rank_candidates(payload: BatchMatchRequest) -> BatchMatchResponse:
    """Evaluates and ranks multiple candidate resumes against a single Job Description."""
    try:
        clean_jd = clean_text(payload.job_description)
        clean_resumes = [clean_text(c.resume_text) for c in payload.resumes]

        sbert: SentenceTransformer = ml_models["sbert"]
        classifier: tf.keras.Model = ml_models["classifier"]
        spacy_nlp = ml_models["spacy"]

        # Vectorized batch encoding and Keras inference
        j_emb_single = np.asarray(sbert.encode([clean_jd], convert_to_numpy=True), dtype=np.float32)
        r_embs = np.asarray(sbert.encode(clean_resumes, convert_to_numpy=True), dtype=np.float32)
        j_embs = np.repeat(j_emb_single, repeats=len(clean_resumes), axis=0)
        sims = compute_cosine_similarity(r_embs, j_embs)

        model_inputs = {
            "resume_embedding": r_embs,
            "jd_embedding": j_embs,
            "cosine_similarity": sims,
        }
        batch_probs = classifier.predict(model_inputs, verbose=0)

        evaluated_candidates: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(payload.resumes):
            probs = batch_probs[idx]
            class_idx = int(np.argmax(probs))

            skill_analysis = extract_skill_gaps(
                resume_text=clean_resumes[idx],
                jd_text=clean_jd,
                sbert_model=sbert,
                nlp_model=spacy_nlp,
            )

            evaluated_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "fit_category": LABEL_MAPPING.get(class_idx, "Unknown"),
                    "confidence_score": round(float(probs[class_idx]), 4),
                    "cosine_similarity": round(float(sims[idx][0]), 4),
                    "skill_gap_analysis": SkillGapReport(**skill_analysis),
                }
            )

        evaluated_candidates.sort(
            key=lambda x: (x["cosine_similarity"], x["confidence_score"]),
            reverse=True,
        )

        rankings = [
            CandidateRankResponse(
                candidate_id=item["candidate_id"],
                rank=rank_idx,
                fit_category=item["fit_category"],
                confidence_score=item["confidence_score"],
                cosine_similarity=item["cosine_similarity"],
                skill_gap_analysis=item["skill_gap_analysis"],
            )
            for rank_idx, item in enumerate(evaluated_candidates, start=1)
        ]

        return BatchMatchResponse(
            job_description_length=len(clean_jd),
            total_candidates_evaluated=len(rankings),
            rankings=rankings,
        )

    except Exception as exc:
        logger.error(f"Error during batch candidate ranking: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch ranking failure: {exc}",
        )