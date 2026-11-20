"""Builds the two supporting notebooks (run from the repo root)."""
import nbformat as nbf


def nb(cells, path):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, path)


SETUP = ("import sys, json\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent / 'src'))\n"
         "import pandas as pd\nfrom irm import config")

nb([("md", "# Week 1 – Extraction harness walkthrough\nSupporting notebook: schema mapping, rules reference run and "
           "error taxonomy. Production logic lives in `src/irm/`. Run `python run.py week1` first."),
    ("code", SETUP),
    ("code", "from irm.schema import FIELD_MAP\npd.DataFrame(FIELD_MAP, columns=['eval_field','source','column','comparison','notes'])"),
    ("code", "s1 = json.loads((config.EXTRACTION_DIR/'w1_textlayer_v1_all_digital'/'_summary.json').read_text())\n"
             "s2 = json.loads((config.EXTRACTION_DIR/'w1_textlayer_v2_all_digital'/'_summary.json').read_text())\n"
             "pd.DataFrame({'v1': {g: m['f1'] for g, m in s1['by_group'].items()}, 'v2': {g: m['f1'] for g, m in s2['by_group'].items()}})"),
    ("code", "slots = pd.read_csv(config.EXTRACTION_DIR/'w1_textlayer_v1_all_digital'/'_slots.csv')\n"
             "slots[slots.error.notna()].head(9)"),
    ("md", "## Gemini runs\nAfter `python run.py compare-tiers --prompt v1`, load "
           "`docs/evaluation/tier_comparison_v1.csv` here."),
    ], "notebooks/01_week1_extraction_harness.ipynb")

nb([("md", "# Week 2 – EDA and retrieval check\nSupporting notebook over the outputs of `python run.py week2`."),
    ("code", SETUP),
    ("code", "eda = json.loads((config.PROCESSED/'eda_summary.json').read_text())\npd.Series(eda['profile']['consistency'])"),
    ("code", "pd.read_csv(config.PROCESSED/'resume_inventory.csv').describe()"),
    ("code", "import sqlite3\ncon = sqlite3.connect(config.SQLITE_DB)\n"
             "pd.read_sql('SELECT skill, COUNT(*) n, ROUND(AVG(years),2) avg_years FROM employee_skill_years "
             "GROUP BY skill ORDER BY n DESC', con)"),
    ("code", "r = pd.read_csv(config.PROCESSED/'retrieval_check_lsa.csv')\n"
             "r.pivot_table(index='category', columns='mode', values='hit@5').round(2)"),
    ], "notebooks/02_week2_eda_retrieval.ipynb")
print("notebooks written")
