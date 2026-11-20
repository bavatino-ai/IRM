# IRM – Intelligent Resume Management Application

![status](https://img.shields.io/badge/status-week%202%20of%204-blue)
![version](https://img.shields.io/badge/version-0.2.0-informational)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![tests](https://img.shields.io/badge/tests-19%20passed-brightgreen)
![licence](https://img.shields.io/badge/licence-MIT-green)

> **Why:** every year HR reads each employee's updated resume and re-types the changes by hand.
> IRM reads the resume, finds the employee, shows HR only what changed, and makes the whole
> workforce searchable by skill, experience and project.

| | |
|---|---|
| **Program / track** | upGrad – Generative AI · Bring Your Own Capstone |
| **Author** | Phuong Dang |
| **Business problem** | Annual resume updates are read and re-typed manually, case by case: slow, error-prone, and no searchable view of skills across the workforce. |
| **Approach** | Gemini vision OCR + schema-constrained extraction → record matching and field-level change detection with HR approval → section-aware chunks in ChromaDB + BM25 → hybrid retrieval, cross-encoder reranking and cited answers (LlamaIndex) → Ragas + Claude as independent judge. |
| **Headline result (to date)** | Gemini extraction micro-F1 0.999 (gemini-3.5-flash-lite, 20 digital resumes) on a harness validated on all 281 resumes (6,925 field slots); SQLite system of record and hybrid index built; scan and answer-quality runs pending (see *Results*). |

Deployment model: **standalone pilot** – IRM works on its own copy of the employee dataset; no
integration with enterprise systems until the pilot is assessed (Week 4).

---

## Installation & setup

```bash
git clone <repo-url>
cd irm                                                  # this folder
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then add GEMINI_API_KEY and ANTHROPIC_API_KEY
```

**Environment variables** (`.env`, never committed): `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, model pins
(`GEMINI_OCR_CANDIDATES`, `GEMINI_OCR_MODEL`, `GEMINI_ANSWER_MODEL`, `GEMINI_EMBED_MODEL`,
`CLAUDE_JUDGE_MODEL`) and optional `PRICE_IN_*` / `PRICE_OUT_*` (USD per 1M tokens) for cost tracking.
Keys are only read from the environment; nothing is hard-coded.

**OS notes.** Windows: use `.venv\Scripts\activate` and `copy .env.example .env`. The first BGE run
downloads the model from Hugging Face (~130 MB).

### Run the pipeline

```bash
# offline steps (no API key needed)
python run.py week2            # EDA, scan variants, SQLite, chunks + Chroma/BM25 (LSA), eval set, retrieval check
python run.py week1            # 20-resume sample + human-check sheet, workload stats, rules reference baseline
pytest -q                      # 19 tests, including the Gemini provider with a mocked client

# API steps (need .env keys)
python run.py check-apis                                 # pins model versions -> docs/evaluation/model_pins.json
python run.py compare-tiers --prompt v1                  # Gemini tier comparison on the 20-resume sample
python run.py before-after --model <selected-model>      # prompt v1 vs v2 (Week 1 fix round)
python run.py extract --provider gemini --set sample-scan --prompt v2   # scan robustness
python run.py index --embedder bge && python run.py retrieval-check --embedder bge
python run.py index --embedder gemini && python run.py retrieval-check --embedder gemini
```

---

## Data provenance

| | Employee_profile | Employee_resume |
|---|---|---|
| **Source** | Synthetic tabular dataset generated with CTGAN (Google Colab) from the author's own dummy R&D employee data | 281 PDF resumes rendered from a random sample of Employee_profile (`scripts/generate_resumes.py`, ReportLab, seed 42) |
| **Location** | `data/raw/employee_profile_dataset.csv` (999 × 48) | `data/raw/resumes/` + `data/raw/resume_manifest.csv` |
| **Link** | [Google Drive folder](https://drive.google.com/drive/folders/1VVU0Rf7A18Zmc2xipWLeLFWnN3vNL_o9?usp=sharing) | same |
| **Licence** | Author's own work – permission granted for use in this upGrad capstone (not an open licence) | same |

**Preprocessing.** Phone kept as text with "+"; whitespace normalisation; `YearOfIssue` as nullable integer
(19 records without certificate kept as NULL, no imputation); multi-value fields exploded into child tables;
sensitive and restricted fields split into separate tables (`data/processed/cleaning_report.json`).

**Synthetic data justification.** Realistic enterprise structure with no real personal data, so the full
pipeline – including external API calls – can be built and tested safely. Each resume adds +1 to +2 years
to every non-zero skill (capped at total experience), creating 1,331 known changes for change-detection evaluation.

**Limitations / bias.** One resume template (layout variety is low; scan variants partly compensate);
generated project descriptions; class imbalance (595 Senior vs 4 Fresher in the profile; 1 Fresher among
the 281 resumes); English-only resumes; Vietnamese names only.

---

## Repository structure

```
irm/
├── src/irm/                    # production logic
│   ├── schema.py               # Pydantic extraction schema + mapping to Employee_profile
│   ├── ingestion.py            # PDF discovery, page rendering, text-layer probe (PyMuPDF)
│   ├── ocr/                    # provider-agnostic OCR: gemini.py (short-term), textlayer.py (non-LLM reference)
│   ├── prompts.py              # versioned extraction prompts (v1, v2)
│   ├── validation.py           # business rules + grounding check -> HR review fields
│   ├── matching.py             # record matching (Code -> name -> fuzzy name)
│   ├── ground_truth.py         # expected values from Employee_profile + manifest
│   ├── extraction_runner.py    # batch runs, sample selection, per-run summaries
│   ├── evaluation/             # extraction metrics + error taxonomy, eval set, retrieval metrics
│   ├── eda.py · scans.py ·     database.py · chunking.py · baseline.py
│   ├── retrieval/              # embedders (gemini / bge / lsa), Chroma + BM25 + RRF
│   └── llm/                    # API checks, Claude judge, response cache + cost
├── data/
│   ├── raw/                    # source CSV, manifest, 281 resumes, template
│   ├── processed/              # cleaned CSV, SQLite DB, EDA summary, cleaning log, inventories
│   ├── evaluation/             # 20-resume sample, human-check sheet, eval queries, human-label subset
│   └── interim/ · index/       # regenerable: extractions, LLM cache, scan variants, Chroma/BM25 (git-ignored)
├── notebooks/          # supporting EDA / harness walkthroughs (not production code)
├── tests/              # unit + integration tests (offline)
├── docs/               # evaluation + measurement records (weekly reports live in IRM/reports/)
│   ├── evaluation/             # model pins, tier comparison, before/after scores
│   ├── baseline/               # manual-baseline protocol, HR timesheet, workload statistics
│   ├── figures/                # EDA and result charts
│   └── issue_log.md            # issues, decisions and open actions (IRM-W1-xx, IRM-W2-xx)
├── scripts/            # dataset generator (provenance), notebook builder, result charts
├── run.py              # single entry point
├── requirements.txt · .env.example · LICENSE
```

---

## Usage & demo

Entry point: `python run.py <command>` (list: `python run.py -h`).

Sample input → output (rules reference on `20003-Ta_Long_Binh.pdf`, same schema Gemini returns):

```json
{
"employee_code": 20003, 
"full_name": "Ta Long Binh", 
"position": "Senior Backend Developer",
"current_project": "Logistics Tracking Platform",
"total_experience_years": 8,
 "skills": [
    {"name": "Golang", "years": 2.5}, 
    {"name": "PHP", "years": 5.0}, 
    {"name": "HTML", "years": 6.0}, 
    {"name": "CSS", "years": 6.0}
    ],
 "education": {
    "degree_level": "Bachelor's Degree", 
    "major": "Cybersecurity",
    "school": "University of Economics Ho Chi Minh City", 
    "start_year": 1993, 
    "graduation_year": 1997
    }
}
```

Change detection for this employee (manifest): PHP 3.5 → 5.0, HTML/CSS 4.5 → 6.0, Golang 1.0 → 2.5.
Screenshots / GIF of the Streamlit review UI: added in Week 3–4.

---

## Results, impact & licence

| Metric | Target | Status (end of Week 2, updated 20 Nov 2026) |
|---|---|---|
| Field-level extraction F1 (Gemini, 20-resume sample, digital) | ≥ 0.90 | **0.999** gemini-3.5-flash-lite · 0.989 gemini-2.5-flash · scans pending |
| Non-LLM reference F1, digital PDFs | – | 0.984 (v1) → 1.000 (v2) on 6,925 slots; scanned PDFs: 0.000 |
| Record matching accuracy | ≥ 99% | 100% on 281 resumes (rules reference); 20/20 on Gemini output |
| Change detection P / R | – | Ground truth ready: 1,331 known changes |
| Hit@5, offline LSA / BM25 / hybrid | uplift vs dense | 0.742 / 0.935 / 0.855 (preliminary; BGE / Gemini pending) |
| Ragas faithfulness / relevance | ≥ 0.85 / 0.80 | Answer + LLM-judge pipeline built (`run.py answer-eval`); run blocked by Anthropic credit |
| HR time per resume | vs baseline | Baseline timing pending (protocol ready) |

**Business impact (to be quantified after the HR timing session):** 24.6 field values per resume are read
and compared by hand today, of which 4.7 actually change; IRM turns that into review-and-approve.

**Limitations & future work.** Single template; synthetic data; Gemini results pending; long-term roadmap:
local OCR model behind the same interface, DOCX / image inputs, retrieval fine-tuning, enterprise integration.

**Licence.** Code: MIT (see `LICENSE`). Data: author's own synthetic work, capstone use only.
