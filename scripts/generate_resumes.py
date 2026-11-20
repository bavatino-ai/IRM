"""
IRM - Synthetic resume generator
Builds PDF resumes from employee_synthetic_dataset.csv following the layout of
"21001-Vo_Tri_Tam [Template].pdf".

Usage:  python3 generate_resumes.py <dataset.csv> <output_dir> [seed]
"""
import sys, os, random, re, csv
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether

SRC, OUT = sys.argv[1], sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
NOW = 2026 * 12 + 10            # November 2026 (0-based month index)

# ---------------------------------------------------------------- vocab maps
TITLE_LABEL = {"Fresher": "Fresher", "Junior": "Junior", "Mid-level": "Middle",
               "Senior": "Senior", "Architect": "Architect"}
LADDER = ["Fresher", "Junior", "Mid-level", "Senior", "Architect"]
ARCHITECT_NAME = {"Software Engineer": "Software Architect", "Backend Developer": "Backend Architect",
                  "Frontend Developer": "Frontend Architect", "Fullstack Developer": "Solution Architect",
                  "DevOps Engineer": "Cloud & DevOps Architect", "Data Scientist": "Data & AI Architect",
                  "Data Analyst": "Data Architect", "QA Engineer": "QA & Test Architect",
                  "Project Manager": "Technical Program Manager (Architect level)"}
FRONTEND = {"Svelte", "Vue.js", "Angular", "Next.js", "Nuxt", "React", "Blazor", "Flutter"}
BACKEND = {"Go", "Laravel", "Spring Boot", "Django", "FastAPI", "GraphQL", ".NET Core", "Node.js", "NestJS"}
DATABASE = {"TimescaleDB", "MySQL", "Oracle", "PostgreSQL", "MariaDB", "SQL Server", "MongoDB"}
DATA_TOOLS = {"Redis": "Redis", "Firebase": "Firebase", "Prisma": "Prisma ORM", "Docker": "Docker"}
LANG_COLS = ["C#", "Python", "TypeScript", "Golang", "PHP", "HTML", "CSS"]
FW_COLS = [".NET", "React.js"]
SKILL_PAIRS = {"C#": ".NET", "HTML": "CSS"}   # identical in source; keep them in sync
BOOST_STEPS = [1.0, 1.5, 2.0]                  # +1 to +2 years over the dataset baseline


def boost_skills(r, rng):
    """Add 1-2 years to every skill > 0, capped at total experience (CompanyTenure)."""
    cap = int(r.CompanyTenure) // 6 / 2          # total years, floored to 0.5
    out, capped = {}, 0
    for c in LANG_COLS + FW_COLS:
        if c in out: continue
        base = float(r[c])
        if base <= 0:
            out[c] = 0.0
        else:
            v = base + rng.choice(BOOST_STEPS)
            if v > cap:
                v = max(base, cap); capped += 1
            out[c] = v
        for a, b in SKILL_PAIRS.items():
            if c == a: out[b] = out[c] if float(r[b]) > 0 else 0.0
            if c == b: out[a] = out[c] if float(r[a]) > 0 else 0.0
    return out, capped


def title_for_months(m):
    if m < 12: return "Fresher"
    if m < 36: return "Junior"
    if m < 83: return "Mid-level"
    if m < 192: return "Senior"
    return "Architect"


def position(title, stack):
    if title == "Architect":
        return ARCHITECT_NAME.get(stack, f"{stack} Architect")
    return f"{TITLE_LABEL[title]} {stack}"


def fmt(mi):
    return f"{mi % 12 + 1:02d}/{mi // 12}"


def split(s):
    return [x.strip() for x in str(s).split("|") if x.strip()]


# ---------------------------------------------------------------- content pools
DESC = {
    "Real-time Dashboard": [
        "This project is a real-time analytics dashboard that aggregates operational data from multiple sources and visualizes KPIs, alerts, and trends for {domain} stakeholders.",
        "A real-time monitoring dashboard that streams live business metrics, supports drill-down reporting, and notifies users when thresholds are breached.",
    ],
    "ERP Solution": [
        "This project is an ERP solution that integrates finance, procurement, inventory, and HR modules into a single platform for a {domain} organization.",
        "An enterprise resource planning system that standardizes back-office processes, approval workflows, and reporting across business units.",
    ],
    "E-commerce Platform": [
        "This project is an e-commerce platform that supports product catalog management, shopping cart, online payment, order fulfillment, and promotions.",
        "A multi-vendor online marketplace that enables merchants to manage storefronts and customers to search, purchase, and track orders.",
    ],
    "CRM System": [
        "This project is a CRM system that manages leads, customer accounts, sales pipelines, and service tickets for a {domain} client.",
        "A customer relationship management platform that centralizes customer interactions, campaign tracking, and sales performance reporting.",
    ],
    "Fintech App": [
        "This project is a fintech application that provides digital wallets, peer-to-peer transfers, bill payment, and transaction history for end users.",
        "A financial technology app that delivers personal finance management, e-KYC onboarding, and secure money transfer features.",
    ],
    "Healthcare System": [
        "This project is a healthcare management system that handles patient registration, appointment scheduling, electronic medical records, and billing.",
        "A hospital information system that digitizes clinical workflows, prescriptions, and lab results while supporting data privacy requirements.",
    ],
    "AI Chatbot": [
        "This project is an AI-powered chatbot that answers user questions, automates customer support, and integrates with internal knowledge bases.",
        "A conversational AI platform that lets users interact with virtual agents for FAQs, request handling, and escalation to human operators.",
    ],
    "Learning Management System": [
        "This project is a learning management system that delivers online courses, assessments, progress tracking, and certification for learners.",
        "An LMS platform that supports course authoring, virtual classrooms, quizzes, and learner analytics for a {domain} organization.",
    ],
    "Logistics Tracking Platform": [
        "This project is a logistics tracking platform that monitors shipments, fleet locations, and delivery status in real time.",
        "A supply chain visibility system that manages orders, warehouse operations, route planning, and proof of delivery.",
    ],
    "Payment Gateway": [
        "This project is a payment gateway that processes card, e-wallet, and bank transfer payments with reconciliation and fraud checks.",
        "A payment processing platform that exposes secure APIs for merchants to accept online payments and manage refunds and settlements.",
    ],
    "Mobile Banking App": [
        "This project is a mobile banking application that offers account overview, fund transfers, card management, and savings products.",
        "A mobile banking solution that enables customers to perform banking transactions, receive notifications, and manage accounts securely.",
    ],
    "IoT Monitoring System": [
        "This project is an IoT monitoring system that collects sensor data from connected devices, detects anomalies, and triggers alerts.",
        "An IoT platform that manages device provisioning, telemetry ingestion, and time-series visualization for operations teams.",
    ],
}

ROLE_RESP = {
    "Developer": ["Implement assigned features and fix defects according to specifications",
                  "Write unit tests and participate in code reviews",
                  "Collaborate with QA to verify and stabilize releases"],
    "Senior Developer": ["Design and implement core modules and complex business logic",
                         "Review code and mentor junior team members",
                         "Optimize application performance, scalability, and maintainability"],
    "Full-stack Developer": ["Develop end-to-end features across frontend and backend layers",
                             "Integrate UI components with RESTful APIs and data services",
                             "Participate in deployment and production support"],
    "Tech Lead": ["Lead technical design and define coding standards for the team",
                  "Break down requirements into technical tasks and estimate effort",
                  "Review code, resolve technical blockers, and ensure delivery quality"],
    "Team Lead": ["Lead a development team, plan sprints, and track delivery progress",
                  "Coordinate with business analysts, QA, and stakeholders on scope and priorities",
                  "Coach team members and manage task allocation"],
    "Architect": ["Define solution architecture, integration patterns, and non-functional requirements",
                  "Evaluate technologies and guide teams on architectural best practices",
                  "Review system design for security, scalability, and reliability"],
}

STACK_RESP = {
    "Software Engineer": ["Develop application features using {fe} and {be}",
                          "Design data models and queries on {db}"],
    "Backend Developer": ["Build and maintain backend services and APIs using {be}",
                          "Design database schemas and optimize queries on {db}"],
    "Frontend Developer": ["Build responsive user interfaces using {fe}",
                           "Integrate frontend with backend APIs and manage application state"],
    "Fullstack Developer": ["Develop frontend screens with {fe} and backend APIs with {be}",
                            "Manage data persistence and migrations on {db}"],
    "DevOps Engineer": ["Set up CI/CD pipelines and containerized deployments for {be} services",
                        "Monitor environments, manage infrastructure, and improve system reliability"],
    "Data Scientist": ["Analyze project data stored in {db} and build predictive models",
                       "Expose analytical results through {be} services and dashboards"],
    "Data Analyst": ["Extract, transform, and analyze data from {db} for reporting",
                     "Build dashboards and reports to support business decisions"],
    "QA Engineer": ["Design test plans and test cases for {fe} and {be} components",
                    "Execute functional, regression, and API testing and track defects"],
    "Project Manager": ["Plan project scope, schedule, and resources; track progress and risks",
                        "Report project status to stakeholders and manage change requests"],
}


def classify(tech_list):
    fe = [t for t in tech_list if t in FRONTEND]
    be = [t for t in tech_list if t in BACKEND]
    db = [t for t in tech_list if t in DATABASE]
    if "TypeScript" in tech_list and not fe: fe = ["TypeScript"]
    return fe, be, db


def uniq(seq):
    out = []
    for x in seq:
        if x not in out: out.append(x)
    return out


def alloc(total, n, rng):
    """split `total` months into n integer durations (>=1)."""
    n = max(1, min(n, total))
    w = [rng.uniform(1, 3) for _ in range(n)]
    s = sum(w)
    d = [max(1, int(total * x / s)) for x in w]
    diff = total - sum(d)
    i = 0
    while diff != 0:
        j = i % n
        if diff > 0: d[j] += 1; diff -= 1
        elif d[j] > 1: d[j] -= 1; diff += 1
        i += 1
    return d


# ---------------------------------------------------------------- styles
B = "Times-Bold"
st = {
    "title": ParagraphStyle("t", fontName=B, fontSize=20, leading=26, alignment=TA_CENTER, spaceAfter=18),
    "hdr": ParagraphStyle("h", fontName=B, fontSize=12, leading=15),
    "sec": ParagraphStyle("s", fontName=B, fontSize=12, leading=15, spaceBefore=14, spaceAfter=2),
    "body": ParagraphStyle("b", fontName="Times-Roman", fontSize=12, leading=15),
    "b1": ParagraphStyle("b1", fontName="Times-Roman", fontSize=12, leading=15, leftIndent=36, bulletIndent=18),
    "b2": ParagraphStyle("b2", fontName="Times-Roman", fontSize=12, leading=15, leftIndent=72, bulletIndent=54),
    "end": ParagraphStyle("e", fontName="Times-Italic", fontSize=12, leading=15, alignment=TA_CENTER, spaceBefore=6),
}


def header_footer(c, doc):
    c.saveState()
    c.setLineWidth(0.8)
    c.line(2.5 * cm, A4[1] - 1.5 * cm, A4[0] - 2.5 * cm, A4[1] - 1.5 * cm)
    c.restoreState()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bullet(text, lvl=1):
    return Paragraph(esc(text), st["b1" if lvl == 1 else "b2"], bulletText="•" if lvl == 1 else "o")


# ---------------------------------------------------------------- build one resume
def build(r, rng):
    code, name = int(r.Code), r.Fullname
    projects, roles, techs = split(r.Project), split(r.Role), split(r.Technology)
    techs = [[t.strip() for t in x.split(",")] for x in techs]
    stacks = uniq(split(r.Techstack))   # career roles, most recent first (dedup)
    frameworks = split(r.Framework)
    T, AT = int(r.CompanyTenure), int(r.AtCompany)
    title = r.Title
    stack0 = stacks[0]

    # ---- timeline: current stint (AtCompany) + earlier stints (CompanyTenure - AtCompany)
    n = len(projects)
    prior = T - AT
    n_cur = max(1, round(n * AT / T)) if T else n
    if prior > 0 and n >= 2: n_cur = min(n_cur, n - 1)
    n_cur = min(n_cur, AT, n)
    n_pri = min(n - n_cur, prior)
    cur_d = alloc(AT, n_cur, rng)
    pri_d = alloc(prior, n_pri, rng) if n_pri else []

    stints = []
    end = NOW + 1
    plist, k = [], 0
    for d in cur_d:
        plist.append((k, end - d, end)); end -= d; k += 1
    stints.append({"title": title, "stack": stack0, "start": NOW + 1 - AT, "end": None, "projects": plist})
    if prior > 0:
        rest = stacks[1:] or [stack0]
        groups = max(1, min(len(rest), len(pri_d))) if pri_d else 1
        if pri_d:
            per = [len(pri_d) // groups + (1 if i < len(pri_d) % groups else 0) for i in range(groups)]
            idx = 0
            for g in range(groups):
                pl = []
                g_end = end
                for d in pri_d[idx: idx + per[g]]:
                    pl.append((k, end - d, end)); end -= d; k += 1
                idx += per[g]
                exp_at_end = T - (NOW + 1 - g_end)
                t = title_for_months(exp_at_end)
                if LADDER.index(t) > LADDER.index(title): t = title
                stints.append({"title": t, "stack": rest[g], "start": end, "end": g_end - 1, "projects": pl})
        else:
            stints.append({"title": title_for_months(prior), "stack": rest[0], "start": NOW + 1 - T,
                           "end": NOW - AT, "projects": []})
    used = k  # number of projects actually placed

    # ---- skills
    all_tech = uniq([t for p in techs[:used] for t in p])
    fe, be, db = classify(all_tech)
    sk, capped = boost_skills(r, random.Random(f"{SEED}-{code}-skills"))
    langs = [f"{c} ({sk[c]:g} yrs)" for c in LANG_COLS if sk[c] > 0]
    fws = [f"{c} ({sk[c]:g} yrs)" for c in FW_COLS if sk[c] > 0]
    fws += [f for f in uniq(frameworks + fe + be) if f not in (".NET", "React.js") and f not in ("TypeScript", "GraphQL")]
    fws = uniq(fws)
    tools = ["Git", "VS Code"] + [v for kk, v in DATA_TOOLS.items() if kk in all_tech]
    dbs = db + [x for x in ("Redis", "Firebase") if x in all_tech]
    net = ["RESTful APIs", "HTTP/HTTPS"] + (["GraphQL"] if "GraphQL" in all_tech else [])

    yrs = T // 12
    exp_txt = f"{yrs}+ years" if yrs >= 1 else f"{T} months"
    summary_tech = uniq(all_tech + frameworks)

    # ---- education
    edu_len = {"Bachelor's Degree": 4, "Master's Degree": 2, "PhD": 4, "Associate Degree": 2}[r.Education]
    deg = {"Bachelor's Degree": f"Bachelor of {r.Major}", "Master's Degree": f"Master of {r.Major}",
           "PhD": f"PhD in {r.Major}", "Associate Degree": f"Associate Degree in {r.Major}"}[r.Education]

    E = []
    E.append(Paragraph("RESUME", st["title"]))
    for lbl, val in (("ID", code), ("NAME", name), ("POSITION/ROLE", position(title, stack0)),
                     ("PROJECT", projects[0])):
        E.append(Paragraph(f"{lbl}: {esc(val)}", st["hdr"]))

    E.append(Paragraph("EXPERIENCE SUMMARY", st["sec"]))
    E.append(bullet(f"{exp_txt} of experience as a {stack0}: using {', '.join(summary_tech)}."))
    E.append(bullet(f"Business domain experience: {r.Domain}."))
    E.append(Spacer(1, 6))

    E.append(Paragraph("TECHNICAL SUMMARY", st["sec"]))
    E.append(bullet("Languages: " + (", ".join(langs) if langs else "N/A")))
    E.append(bullet("Frameworks & Libraries: " + ", ".join(fws)))
    if dbs: E.append(bullet("Databases: " + ", ".join(dbs)))
    E.append(bullet("Operating Systems: Windows, Linux"))
    E.append(bullet("Networking: " + ", ".join(net)))
    E.append(bullet("Tools: " + ", ".join(tools)))
    E.append(Spacer(1, 6))

    E.append(Paragraph("EDUCATION &amp; QUALIFICATIONS", st["sec"]))
    for lang, lvl in zip(split(r.Language), split(r.Level)):
        E.append(bullet(f"{lang} - {lvl}"))
    if pd.notna(r.YearOfIssue):
        E.append(bullet(f"{int(r.YearOfIssue)}: {r.Certificate}"))
    E.append(bullet(f"{int(r.Graduation) - edu_len} – {int(r.Graduation)}: {deg} in {r.School}."))
    E.append(Spacer(1, 10))

    E.append(Paragraph("PROFESSIONAL EXPERIENCE", st["sec"]))
    for s in stints:
        span = f"[{fmt(s['start'])} – {'Now' if s['end'] is None else fmt(s['end'])}]"
        E.append(Spacer(1, 8))
        E.append(Paragraph(f"{span}: {esc(position(s['title'], s['stack']))}", st["hdr"]))
        if not s["projects"]:
            E.append(bullet(f"Worked as {position(s['title'], s['stack'])}; project details not recorded."))
        for (pi, ps, pe) in s["projects"]:
            p, role, tl = projects[pi], roles[pi], techs[pi]
            pfe, pbe, pdb = classify(tl)
            if not pdb and "Prisma" in tl: pdb = ["Prisma ORM"]
            if not pdb and "Redis" in tl: pdb = ["Redis"]
            if not pbe and "Firebase" in tl: pbe = ["Firebase"]
            ctx = {"fe": ", ".join(pfe) or "the frontend framework", "be": ", ".join(pbe) or "the backend framework",
                   "db": ", ".join(pdb) or "the project database", "domain": r.Domain}
            desc = rng.choice(DESC.get(p, ["This project is a {domain} software solution."])).format(**ctx)
            resp = rng.sample(ROLE_RESP.get(role, ROLE_RESP["Developer"]), 2) + \
                   [x.format(**ctx) for x in STACK_RESP.get(s["stack"], STACK_RESP["Software Engineer"])]
            if "Redis" in tl: resp.append("Apply Redis for caching and session management")
            if "Docker" in tl: resp.append("Containerize services with Docker for consistent deployments")
            if "Firebase" in tl: resp.append("Integrate Firebase for authentication and push notifications")
            if "GraphQL" in tl: resp.append("Design GraphQL schemas and resolvers for client data access")
            pspan = f"[{fmt(ps)} – {'Now' if pe == NOW + 1 else fmt(pe - 1)}]"
            blk = [Spacer(1, 6), Paragraph(pspan, st["body"]),
                   bullet(f"Project: {p}"),
                   bullet(f"Project description: {desc}"),
                   bullet(f"Role: {role}"),
                   bullet("Responsibilities:")]
            blk += [bullet(x, 2) for x in resp]
            blk.append(bullet("Technologies: " + ", ".join(tl) + "."))
            E.append(KeepTogether(blk))
    E.append(Spacer(1, 6))
    E.append(Paragraph("---------- End of Document ---------", st["end"]))

    fname = f"{code}-{name.replace(' ', '_')}.pdf"
    doc = SimpleDocTemplate(os.path.join(OUT, fname), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.2 * cm, bottomMargin=2 * cm, title=f"Resume - {name}", author="IRM synthetic data")
    doc.build(E, onFirstPage=header_footer, onLaterPages=header_footer)
    return fname, position(title, stack0), projects[0], used, n, sk, capped


# ---------------------------------------------------------------- main
def main():
    df = pd.read_csv(SRC)
    rng = random.Random(SEED)
    N = rng.randint(200, 300)
    rows = df.sample(n=N, random_state=SEED).sort_values("Code")
    os.makedirs(OUT, exist_ok=True)
    man = []
    for _, r in rows.iterrows():
        f, pos, proj, used, n, sk, capped = build(r, random.Random(f"{SEED}-{r.Code}"))
        man.append({**{f"{c}_base": float(r[c]) for c in LANG_COLS + FW_COLS},
                    **{f"{c}_resume": sk[c] for c in LANG_COLS + FW_COLS}, "SkillsCapped": capped,"Code": r.Code, "Fullname": r.Fullname, "File": f, "Position": pos,
                    "CurrentProject": proj, "ProjectsInSource": n, "ProjectsRendered": used,
                    "Title": r.Title, "CompanyTenure_months": r.CompanyTenure, "AtCompany_months": r.AtCompany})
    pd.DataFrame(man).to_csv(os.path.join(OUT, "_resume_manifest.csv"), index=False)
    print("generated", N)


if __name__ == "__main__":
    main()
