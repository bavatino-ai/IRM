"""Central configuration: paths, model pins and pipeline constants.

All paths are relative to the repository root so the code runs after cloning.
Model IDs are *candidates* — pin the exact versions you validate in Google AI
Studio / Anthropic Console in `.env` (see `.env.example`).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# ------------------------------------------------------------------ paths
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EVAL_DIR = DATA / "evaluation"
INDEX_DIR = DATA / "index"
DOCS = ROOT / "docs"                       # evaluation + measurement records (weekly reports live in IRM/reports)
EVAL_RESULTS = DOCS / "evaluation"         # model pins, tier comparisons, before/after scores
BASELINE_DIR = DOCS / "baseline"           # manual-baseline protocol, HR timesheet, workload stats
FIGURES = DOCS / "figures"                 # EDA and result charts

PROFILE_CSV = RAW / "employee_profile_dataset.csv"
MANIFEST_CSV = RAW / "resume_manifest.csv"
RESUME_DIR = RAW / "resumes"
SCAN_DIR = INTERIM / "resumes_scan"
CACHE_DIR = INTERIM / "llm_cache"
EXTRACTION_DIR = INTERIM / "extractions"
SQLITE_DB = PROCESSED / "irm.db"

# ------------------------------------------------------------------ models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Week 1 tier comparison candidates (comma-separated in .env). Verify names in
# Google AI Studio before the run and record the pinned versions in docs/evaluation/model_pins.json.
GEMINI_OCR_CANDIDATES = [m.strip() for m in os.getenv(
    "GEMINI_OCR_CANDIDATES", "gemini-3.5-flash-lite,gemini-2.5-flash,gemini-3.1-pro-preview").split(",") if m.strip()]
GEMINI_OCR_MODEL = os.getenv("GEMINI_OCR_MODEL", "gemini-2.5-flash")          # selected after Week 1
GEMINI_ANSWER_MODEL = os.getenv("GEMINI_ANSWER_MODEL", "gemini-3.1-pro-preview")      # search / analysis
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CLAUDE_JUDGE_MODEL = os.getenv("CLAUDE_JUDGE_MODEL", "claude-sonnet-5-5")
BGE_EMBED_MODEL = os.getenv("BGE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Swapped-role experiment: Claude as OCR / answer model, Gemini as judge
# (`run.py extract --provider claude`, `run.py answer-eval --answer-provider claude --judge-provider gemini`).
CLAUDE_OCR_MODEL = os.getenv("CLAUDE_OCR_MODEL", "claude-opus-5-5")
CLAUDE_ANSWER_MODEL = os.getenv("CLAUDE_ANSWER_MODEL", "claude-opus-5-5")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash")
# Needed only for an Anthropic key that is not scoped to a workspace (sent as `anthropic-workspace-id`).
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "")

# Price per 1M tokens (USD). Left empty on purpose: fill from the official
# pricing page on the day of the run so cost figures are defensible.
def _price(name: str) -> float | None:
    v = os.getenv(name, "")
    return float(v) if v else None


PRICING = {  # model -> (input_per_1M, output_per_1M)
    m: (_price(f"PRICE_IN_{m.upper().replace('-', '_').replace('.', '_')}"),
        _price(f"PRICE_OUT_{m.upper().replace('-', '_').replace('.', '_')}"))
    for m in GEMINI_OCR_CANDIDATES + [GEMINI_ANSWER_MODEL, GEMINI_JUDGE_MODEL,
                                      CLAUDE_OCR_MODEL, CLAUDE_ANSWER_MODEL, CLAUDE_JUDGE_MODEL]
}

# ------------------------------------------------------------------ pipeline constants
RENDER_DPI = int(os.getenv("RENDER_DPI", "150"))
# Pause between Gemini requests (free tier has requests-per-minute limits).
GEMINI_MIN_INTERVAL_S = float(os.getenv("GEMINI_MIN_INTERVAL_S", "6"))
SEED = 42
# Skill columns that carry "years of experience" on the resume (Technical Summary).
LANG_SKILLS = ["C#", "Python", "TypeScript", "Golang", "PHP", "HTML", "CSS"]
FW_SKILLS = [".NET", "React.js"]
YEAR_SKILLS = LANG_SKILLS + FW_SKILLS

# Guardrail: never sent to an LLM, never used for ranking (proposal §4.8).
SENSITIVE_FIELDS = ["Income", "Age", "Gender", "Marital", "Dependents", "Attrition",
                    "Phone", "Email", "DistanceFromHome"]
# Role-restricted (HR / authorised managers only); stored, filterable, not embedded.
RESTRICTED_FIELDS = ["PerformanceRating", "JobSatisfaction", "WorkLifeBalance",
                     "CompanyReputation", "Promotions", "Leadership", "Innovation",
                     "Recognition", "Overtime", "Remote", "TeamSize"]
