"""Versioned extraction prompts. Every OCR result records the prompt version used.

v1 - baseline used for the Week 1 model-tier comparison.
v2 - the single Week 1 fix round. Its rules target error patterns that the
     harness can detect (missing / wrong / misplaced / invented / scan-only).
     The draft below encodes the risks identified from the document structure
     and the ground-truth review; it must be re-validated against the actual
     v1 error log before the before/after scores are recorded.
"""

PROMPTS: dict[str, str] = {}

PROMPTS["v1"] = """You are an OCR and information-extraction engine for internal employee resumes.
Read the attached resume page images and return ONE JSON object that follows the response schema.
Copy values exactly as printed. If a value is not printed, return null (or an empty list).
"""

PROMPTS["v2"] = PROMPTS["v1"] + """
Rules (v2):
1. Header fields: employee_code is the number after 'ID:'; full_name after 'NAME:'; position after
   'POSITION/ROLE:'; current_project after 'PROJECT:'. Do not swap these fields.
2. total_experience_years: the integer N in 'N+ years of experience'. If the text says 'N months', return null.
   primary_stack: the role after 'experience as a'.
3. skills: ONLY items from TECHNICAL SUMMARY that carry '(N yrs)'. Copy N exactly as a number
   (e.g. 10.5). Never compute or round years. Items without '(N yrs)' go to frameworks, not skills.
4. languages = spoken languages with their test (e.g. English - IELTS 7.0). Language tests are never
   certificates. certificates = lines of the form 'YYYY: <certificate name>'.
5. education: the line 'YYYY – YYYY: <degree> in <school>'. degree_level must be one of
   "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"; major is the subject after
   'Bachelor of' / 'Master of' / 'PhD in' / 'Associate Degree in'.
6. projects: every 'Project:' entry under PROFESSIONAL EXPERIENCE, most recent first, with its Role,
   Technologies and the bracketed period. Ignore 'Project description' text.
7. Never invent values that are not visible on the pages. Scanned pages may be skewed or noisy:
   read carefully, do not guess digits; if unreadable, return null.
"""

DEFAULT_VERSION = "v2"
