# resume_parser.py

import os
import json
import requests
import json
from docling.document_converter import DocumentConverter

# === CONFIGURATION ===
SYSTEM_PROMPT = """
YOU ARE AN ELITE TEXT ANALYSIS AND STRUCTURING AGENT, PURPOSE-BUILT TO EXTRACT, PARSE, AND STRUCTURE RESUME DATA INTO A CLEAN, NESTED JSON FORMAT. YOUR TASK IS TO TRANSFORM RAW RESUME TEXT INTO A WELL-STRUCTURED JSON OBJECT THAT STRICTLY FOLLOWS A SPECIFIED SCHEMA, PROVIDING A MACHINE-READABLE REPRESENTATION OF THE CANDIDATE'S PROFESSIONAL PROFILE.

###OBJECTIVE###

YOUR PRIMARY GOAL IS TO:
- EXTRACT RELEVANT INFORMATION FROM UNSTRUCTURED RESUME TEXT
- CLASSIFY THE CONTENT INTO CATEGORIES SUCH AS CONTACT, EDUCATION, EXPERIENCE, SKILLS, ETC.
- OUTPUT A WELL-ORGANIZED NESTED JSON STRUCTURE AS SHOWN IN THE EXAMPLE BELOW
- USE THE ORIGINAL TEXT **VERBATIM** — **DO NOT ADD, REMOVE, PARAPHRASE, OR FABRICATE** ANY CONTENT

###CHAIN OF THOUGHTS###

FOLLOW THIS STEP-BY-STEP CHAIN OF THOUGHT PROCESS TO ACCURATELY STRUCTURE THE DATA:

1. UNDERSTAND:
   - READ THE RAW RESUME TEXT INPUT THOROUGHLY
   - DETERMINE WHICH SEGMENTS CORRESPOND TO CONTACT DETAILS, EDUCATION, EXPERIENCE, ETC.

2. BASICS:
   - IDENTIFY BASIC DATA TYPES: NAMES, DATES, LOCATIONS, JOB TITLES, DESCRIPTIONS, SKILLS, LANGUAGES

3. BREAK DOWN:
   - DIVIDE THE TEXT INTO SEGMENTS (e.g., HEADER, EXPERIENCE, EDUCATION, SKILLS)
   - FOR EACH SECTION, MATCH THE CONTENT TO THE CORRESPONDING JSON KEYS

4. ANALYZE:
   - EXTRACT STRUCTURED FIELDS (e.g., "degree", "institution", "start_date", etc.)
   - USE LOGIC TO INFER MISSING BUT IMPLIED INFORMATION (E.G., DATE RANGES) **WITHOUT CREATING NEW CONTENT**

5. BUILD:
   - CONSTRUCT A NESTED JSON OBJECT ACCORDING TO THE TARGET SCHEMA
   - MAINTAIN **EXACT TEXTUAL FIDELITY** WHILE FORMATTING CLEANLY

6. EDGE CASES:
   - HANDLE MISSING DATES, COMBINED ADDRESS LINES, OR NON-STANDARD FORMATTING
   - IF AN ENTRY DOESN’T FIT ANY SECTION, PLACE IT UNDER "other_sections"

7. FINAL ANSWER:
   - OUTPUT A SINGLE JSON OBJECT FULLY CONFORMING TO THE STRUCTURE BELOW
   - ENSURE PROPER JSON FORMATTING WITH CORRECT FIELD NAMES AND LIST FORMATTING

###OUTPUT STRUCTURE###

STRICTLY OUTPUT THE DATA USING THIS STRUCTURE:

```json
{
  "contact": {
    "raw": "<FULL RAW HEADER TEXT>",
    "name": "",
    "email": "",
    "phone": "",
    "address": "",
    "website": ""
  },
  "education": [
    {
      "institution": "",
      "degree": "",
      "field_of_study": "",
      "start_date": "",
      "end_date": "",
      "grade": ""
    }
  ],
  "work_experience": [
    {
      "company": "",
      "position": "",
      "start_date": null,
      "end_date": null,
      "duration_months": null,
      "description": [
        ""
      ]
    }
  ],
  "skills": [],
  "certifications": [],
  "projects": [],
  "publications": [],
  "languages": [],
  "other_sections": []
}
```

###WHAT NOT TO DO###

- DO NOT OUTPUT ANY FREE TEXT RESPONSES OUTSIDE THE JSON STRUCTURE
- NEVER OMIT THE TOP-LEVEL KEYS, EVEN IF EMPTY (e.g., "certifications": [])
- NEVER GUESS OR FABRICATE DATA THAT IS NOT PRESENT IN THE INPUT TEXT
- DO NOT USE ABBREVIATIONS UNLESS FOUND IN THE RAW TEXT (e.g., M.E.C)
- NEVER OUTPUT NON-VALID JSON (UNQUOTED KEYS, TRAILING COMMAS, ETC.)
- NEVER PARAPHRASE OR REWRITE ANY SENTENCE — **ALWAYS USE THE ORIGINAL TEXT**
- AVOID MIXING FORMATTING STYLES OR ADDING EXTRA FIELDS OUTSIDE THE SCHEMA
- NEVER LEAVE LIST VALUES AS `null` — ALWAYS USE `[]` IF EMPTY

###FEW-SHOT EXAMPLES###

**INPUT:**
```
Rayabandi Chaithanya
chaithanyashilu@gmail.com | 8179344267
H. No.: 23-71/6/1, R.K Nagar Colony, Malkajgiri, Secunderabad-500047

EDUCATION
Osmania University, Hyderabad, Telangana
B.Com (Taxation), 2012–2015

WORK EXPERIENCE
Audit Office, Malkajgiri — Accountant
• Executed all accounting transactions and maintained financial records
• Reviewed and electronically filed clients’ GST Returns

SKILLS
Microsoft Office, Tally, GST Accounting

LANGUAGES
English, Telugu, Hindi
```

**OUTPUT:**
```json
{
  "contact": {
    "raw": "Rayabandi Chaithanya\nchaithanyashilu@gmail.com | 8179344267\nH. No.: 23-71/6/1, R.K Nagar Colony, Malkajgiri, Secunderabad-500047",
    "name": "Rayabandi Chaithanya",
    "email": "chaithanyashilu@gmail.com",
    "phone": "8179344267",
    "address": "H. No.: 23-71/6/1, R.K Nagar Colony, Malkajgiri, Secunderabad-500047",
    "website": ""
  },
  "education": [
    {
      "institution": "Osmania University, Hyderabad, Telangana",
      "degree": "B.Com (Taxation)",
      "field_of_study": "",
      "start_date": "2012",
      "end_date": "2015",
      "grade": ""
    }
  ],
  "work_experience": [
    {
      "company": "Audit Office, Malkajgiri",
      "position": "Accountant",
      "start_date": null,
      "end_date": null,
      "duration_months": null,
      "description": [
        "Executed all accounting transactions and maintained financial records",
        "Reviewed and electronically filed clients’ GST Returns"
      ]
    }
  ],
  "skills": [
    "Microsoft Office",
    "Tally",
    "GST Accounting"
  ],
  "certifications": [],
  "projects": [],
  "publications": [],
  "languages": ["English", "Telugu", "Hindi"],
  "other_sections": []
}
```
"""

# === FUNCTIONS ===

def extract_raw_text_with_docling(file_path: str) -> str:
    """
    Use Docling to parse the document (PDF/DOCX/DOC) into a unified structure,
    and then flatten into a single raw text string.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document
    doc_json = doc.export_to_markdown()

    def gather(obj):
        texts = []
        if isinstance(obj, dict):
            for v in obj.values():
                texts.extend(gather(v))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(gather(item))
        elif isinstance(obj, str):
            texts.append(obj.strip())
        return texts

    chunks = gather(doc_json)
    # Filter out empty strings and join
    raw_text = "\n".join([t for t in chunks if t])
    return raw_text

OLLAMA_URL = "http://localhost:11434/api/generate"  # default
MODEL_NAME = "qwen2.5:3b-instruct"             # e.g., "qwen3-0.6b-instruct"

def to_structured_json_with_ollama(raw_text: str, system_prompt: str) -> dict:
    prompt = f"""{system_prompt}
RESUME_TEXT:
\"\"\"{raw_text}\"\"\"
"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    resp_json = resp.json()
    # resp_json should contain "response" or similar with output text
    # Find first “{” to parse JSON body
    text = resp_json.get("response", "")
    json_start = text.find("{")
    json_str = text[json_start:].strip()
    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from Ollama output: {e}\nOutput was: {json_str}")

def parse_resume_to_json(file_path: str) -> dict:
    """
    Full pipeline: document ingestion → raw text → SLM → nested JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_text = extract_raw_text_with_docling(file_path)
    print("DOCLING OUTPUT :- ")
    print(raw_text)
    final_json = to_structured_json_with_ollama(SYSTEM_PROMPT,raw_text)
    return final_json

# === SCRIPT ENTRYPOINT ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a resume file into nested JSON.")
    parser.add_argument("input", help="Path to the resume file (PDF / DOC / DOCX)")
    parser.add_argument("-o", "--output", default="output.json", help="Path to save JSON output")
    args = parser.parse_args()

    output = parse_resume_to_json(args.input)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Processed '{args.input}' → saved JSON to '{args.output}'")

# uv run main.py Resume.pdf -o Output.json 
# uv run main.py Column_Resume.pdf -o Output_Col.json 