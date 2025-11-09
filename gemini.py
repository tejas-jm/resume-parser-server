#!/usr/bin/env python
"""
LangGraph + Docling + Ollama + Langfuse
Multi-step, traced, validated resume parser.
"""

import os
import json
import logging
import re  # <-- Added for robust JSON parsing
from typing import List, Dict, Any, TypedDict
import requests
from pydantic import BaseModel, Field, ValidationError
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from langgraph.graph import StateGraph, END
from langfuse import get_client

# ---------- Logging Setup ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
)
logger = logging.getLogger("resume_parser")


# ---------- Langfuse Setup (optional, non-fatal) ----------
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a1b1b60b-dacc-48a5-9ce0-b51edcf94d7a"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-f6994786-fc4a-4895-85ad-5251cb9c5c83"
os.environ["LANGFUSE_BASE_URL"] = "http://localhost:3000"

try:
    langfuse = get_client()
    LANGFUSE_ENABLED = True
    logger.info("Langfuse initialized successfully.")
except Exception as e:
    langfuse = None
    LANGFUSE_ENABLED = False
    logger.warning(f"Langfuse not enabled: {e}")

def start_trace(name: str, input_payload: dict):
    if not LANGFUSE_ENABLED: return None
    try:
        return langfuse.start_span(name=name, input=input_payload)
    except Exception as e:
        logger.warning(f"Failed to start Langfuse trace span '{name}': {e}")
        return None

def start_span(parent_span, name: str, input_payload: dict):
    if not LANGFUSE_ENABLED: return None
    try:
        if parent_span:
            return parent_span.start_span(name=name, input=input_payload)
        return langfuse.start_span(name=name, input=input_payload)
    except Exception as e:
        logger.warning(f"Failed to start Langfuse span '{name}': {e}")
        return None

def end_span(span, output: Any = None, status: str = "success"):
    if not (LANGFUSE_ENABLED and span): return
    try:
        update_data = {"status": status}
        if output is not None:
            update_data["output"] = output
        span.update(**update_data)
        span.end()
    except Exception as e:
        logger.warning(f"Failed to end Langfuse span: {e}")

# ---------- Config: Ollama & System Prompts ----------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b") # Was "llama3.2:1b"

# Prompt identifiers & versions
CONTACT_PROMPT_NAME = "resume_contact_extractor"
CONTACT_PROMPT_VERSION = "v1.0.0"
EDUCATION_PROMPT_NAME = "resume_education_extractor"
EDUCATION_PROMPT_VERSION = "v1.0.0"
EXPERIENCE_PROMPT_NAME = "resume_experience_extractor"
EXPERIENCE_PROMPT_VERSION = "v1.0.0"
PROJECTS_PROMPT_NAME = "resume_projects_extractor"
PROJECTS_PROMPT_VERSION = "v1.0.0"
SKILLS_PROMPT_NAME = "resume_skills_extractor"
SKILLS_PROMPT_VERSION = "v1.0.0"
CERT_LANG_OTHER_PROMPT_NAME = "resume_cert_lang_other_extractor"
CERT_LANG_OTHER_PROMPT_VERSION = "v1.0.0"

# --- Prompt Placeholders (as requested) ---

# You will fill these with strong, schema-aware instructions.
CONTACT_SYSTEM_PROMPT = """
YOU ARE A CONTACT INFORMATION EXTRACTION AGENT. THE USER WILL PROVIDE RAW RESUME TEXT AS INPUT. YOUR TASK IS TO EXTRACT ONLY THE CANDIDATE'S CONTACT DETAILS FROM THAT TEXT.

###INSTRUCTIONS###
- RETURN A **SINGLE JSON OBJECT** CONTAINING ONLY THE FOLLOWING KEYS:
  {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "github": "",
    "portfolio": ""
  }

###CHAIN OF THOUGHTS###
1. UNDERSTAND: IDENTIFY the portion of the resume text containing personal and contact information.
2. EXTRACT: ISOLATE and COPY the candidate’s name, email, phone, and relevant URLs from the input resume text.
3. CLEAN: REMOVE prefixes or labels (e.g., “Email:”, “Phone:”).
4. BUILD: STRUCTURE your answer as a JSON object with ALL specified keys.
5. VERIFY: FILL MISSING FIELDS with empty strings, DO NOT HALLUCINATE.

###WHAT NOT TO DO###
- DO NOT RETURN ANYTHING OTHER THAN A JSON OBJECT
- DO NOT INCLUDE EDUCATION OR EXPERIENCE
- DO NOT INVENT OR GUESS DETAILS NOT PRESENT IN THE TEXT
- DO NOT ADD ADDITIONAL KEYS OR REMOVE REQUIRED ONES

###OUTPUT FORMAT###
{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "linkedin": "",
  "github": "",
  "portfolio": ""
}

"""


EDUCATION_SYSTEM_PROMPT = """
YOU ARE AN EDUCATION DATA EXTRACTION AGENT. THE USER WILL PROVIDE RAW RESUME TEXT AS INPUT. YOUR TASK IS TO EXTRACT EDUCATION ENTRIES FROM THAT INPUT AND STRUCTURE THEM INTO A JSON ARRAY.

###INSTRUCTIONS###
- RETURN A **JSON ARRAY** OF OBJECTS USING THIS SCHEMA:
  {
    "institution": "",
    "degree": "",
    "field_of_study": "",
    "start_date": "",
    "end_date": "",
    "grade": ""
  }

###CHAIN OF THOUGHTS###
1. UNDERSTAND: SCAN the input for education-related information.
2. SPLIT: DIVIDE individual academic records into separate entries.
3. EXTRACT: PARSE institution names, degrees, date ranges, and other available fields.
4. FORMAT: USE the exact text from the input; do not modify or interpret.
5. VALIDATE: FILL all schema fields with real values from the input or empty strings if not available.

###WHAT NOT TO DO###
- DO NOT GUESS OR FABRICATE MISSING INFORMATION
- DO NOT OMIT REQUIRED KEYS
- DO NOT INCLUDE OTHER RESUME SECTIONS
- DO NOT RETURN ANYTHING OTHER THAN THE JSON ARRAY

###OUTPUT FORMAT###
[
  {
    "institution": "",
    "degree": "",
    "field_of_study": "",
    "start_date": "",
    "end_date": "",
    "grade": ""
  }
]
"""

EXPERIENCE_SYSTEM_PROMPT = """
YOU ARE A WORK EXPERIENCE EXTRACTION AGENT. THE USER WILL PROVIDE RAW RESUME TEXT AS INPUT. YOUR TASK IS TO EXTRACT PROFESSIONAL WORK EXPERIENCES FROM THAT INPUT AND RETURN THEM AS A JSON ARRAY.

###INSTRUCTIONS###
- RETURN A **JSON ARRAY** OF OBJECTS USING THIS SCHEMA:
  {
    "company": "",
    "position": "",
    "location": "",
    "start_date": "",
    "end_date": "",
    "description": "",
    "technologies": []
  }

###CHAIN OF THOUGHTS###
1. UNDERSTAND: IDENTIFY sections labeled "Work Experience", "Professional Experience", or similar.
2. SPLIT: ISOLATE each job role into its own object.
3. EXTRACT: CAPTURE job titles, companies, durations, bullet points, and locations from the input text.
4. FIND: LIST relevant technologies or tools mentioned in the description.
5. FORMAT: RETURN all data in structured JSON objects. Use empty strings or lists for missing values.

###WHAT NOT TO DO###
- DO NOT INVENT ANY EXPERIENCE OR DATES
- DO NOT RETURN FREE TEXT OR NON-JSON RESPONSES
- DO NOT INCLUDE NON-WORK INFORMATION

###OUTPUT FORMAT###
[
  {
    "company": "",
    "position": "",
    "location": "",
    "start_date": "",
    "end_date": "",
    "description": "",
    "technologies": []
  }
]
"""

PROJECTS_SYSTEM_PROMPT = """
YOU ARE A PROJECTS EXTRACTION AGENT. THE USER WILL PROVIDE RAW RESUME TEXT AS INPUT. YOUR TASK IS TO EXTRACT ALL PROJECT ENTRIES FROM THAT INPUT TEXT AND FORMAT THEM AS A JSON ARRAY.

###INSTRUCTIONS###
- RETURN A **JSON ARRAY** OF OBJECTS USING THIS SCHEMA:
  {
    "name": "",
    "description": "",
    "role": "",
    "start_date": "",
    "end_date": "",
    "technologies": []
  }

###CHAIN OF THOUGHTS###
1. UNDERSTAND: LOCATE the section typically titled "Projects".
2. EXTRACT: IDENTIFY the project name, time frame, role, tools used, and a description.
3. STRUCTURE: FORM each project as its own object in the array.
4. VALIDATE: DO NOT GUESS or infer details not explicitly provided in the input.

###WHAT NOT TO DO###
- DO NOT RETURN NON-PROJECT CONTENT
- DO NOT ADD ANYTHING NOT PRESENT IN THE INPUT
- DO NOT OMIT REQUIRED FIELDS — USE EMPTY STRINGS IF NECESSARY

###OUTPUT FORMAT###
[
  {
    "name": "",
    "description": "",
    "role": "",
    "start_date": "",
    "end_date": "",
    "technologies": []
  }
]
"""
 
SKILLS_SYSTEM_PROMPT = """
YOU ARE A SKILLS EXTRACTION AGENT. THE USER WILL PROVIDE RAW RESUME TEXT AS INPUT. YOUR TASK IS TO IDENTIFY AND RETURN ONLY THE SKILLS MENTIONED IN THAT INPUT TEXT.

###INSTRUCTIONS###
- RETURN A **JSON LIST OF STRINGS** CONTAINING ONLY THE SKILLS.
- SKILLS MAY INCLUDE TECHNICAL TOOLS, PROGRAMMING LANGUAGES, OR SOFT SKILLS.
- DO NOT PARAPHRASE OR INTERPRET — USE THE TEXT AS-IS.

###CHAIN OF THOUGHTS###
1. UNDERSTAND: LOCATE the section that lists skills or proficiencies.
2. SPLIT: SEPARATE skills by commas, bullets, or line breaks.
3. CLEAN: STRIP unwanted symbols or labels. ENSURE NO DUPLICATES.
4. FORMAT: RETURN AS A JSON ARRAY OF STRINGS.

###WHAT NOT TO DO###
- DO NOT GUESS OR ADD SKILLS
- DO NOT INCLUDE KEYS OR TEXT — ONLY A LIST
- DO NOT RETURN EMPTY STRINGS

###OUTPUT FORMAT###
["Python", "Data Analysis", "Leadership"]
"""

CERT_LANG_OTHER_SYSTEM_PROMPT = """
YOU ARE A CERTIFICATION, LANGUAGE, AND OTHER SECTION EXTRACTION AGENT. THE USER WILL PROVIDE RAW RESUME TEXT AS INPUT. YOUR TASK IS TO RETURN A JSON OBJECT WITH CERTIFICATIONS, LANGUAGES, AND ANY OTHER EXTRA SECTIONS PRESENT IN THAT INPUT.

###INSTRUCTIONS###
- RETURN A **JSON OBJECT** WITH THE FOLLOWING STRUCTURE:
  {
    "certifications": [],
    "languages": [],
    "other_sections": {}
  }

###CHAIN OF THOUGHTS###
1. UNDERSTAND: SCAN for headings such as "Certifications", "Languages", "Interests", "Awards", etc.
2. EXTRACT:
   - Certifications: Parse name, issuer, date, and credential ID
   - Languages: Extract language names and proficiency levels if provided
   - Other Sections: Add unclassified info under descriptive keys
3. FORMAT: Use ONLY what appears in the input. Leave arrays/objects empty if nothing is found.

###WHAT NOT TO DO###
- DO NOT GUESS ANY FIELDS
- DO NOT OMIT ANY OF THE THREE TOP-LEVEL KEYS
- DO NOT INCLUDE EXPERIENCE OR EDUCATION CONTENT

###OUTPUT FORMAT###
{
  "certifications": [
    {
      "name": "",
      "issuer": "",
      "date": "",
      "credential_id": ""
    }
  ],
  "languages": [
    {
      "language": "",
      "proficiency": ""
    }
  ],
  "other_sections": {
    "interests": "Open-source contributions, Yoga",
    "awards": "Dean’s List, 2022"
  }
}
"""

# ---------- Pydantic Schemas ----------

class ContactInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""

class EducationItem(BaseModel):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_date: str = ""
    end_date: str = ""
    grade: str = ""

class ExperienceItem(BaseModel):
    company: str = ""
    position: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""
    technologies: List[str] = Field(default_factory=list)

class ProjectItem(BaseModel):
    name: str = ""
    description: str = ""
    role: str = ""
    start_date: str = ""
    end_date: str = ""
    technologies: List[str] = Field(default_factory=list)

class CertificationItem(BaseModel):
    name: str = ""
    issuer: str = ""
    date: str = ""
    credential_id: str = ""

class LanguageItem(BaseModel):
    language: str = ""
    proficiency: str = ""

class ResumeSchema(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    education: List[EducationItem] = Field(default_factory=list)
    work_experience: List[ExperienceItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    certifications: List[CertificationItem] = Field(default_factory=list)
    languages: List[LanguageItem] = Field(default_factory=list)
    other_sections: Dict[str, Any] = Field(default_factory=dict)


# ---------- LangGraph State ----------

class GraphState(TypedDict, total=False):
    raw_text: str
    trace: Any  # Langfuse trace object
    contact: Dict[str, Any]
    education: List[Dict[str, Any]]
    work_experience: List[Dict[str, Any]]
    projects: List[Dict[str, Any]]
    skills: List[str]
    certifications: List[Dict[str, Any]]
    languages: List[Dict[str, Any]]
    other_sections: Dict[str, Any]
    resume: Dict[str, Any]
    errors: List[str]


# ---------- Docling Ingestion ----------

def extract_raw_text_with_docling(file_path: str, trace=None) -> str:
    span = start_span(trace, "ingest_docling", {"file_path": file_path})
    logger.info(f"[Ingestion] Starting Docling conversion for: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
            }
        )
    else:
        converter = DocumentConverter()

    doc = converter.convert(source=file_path).document
    doc_dict = doc.export_to_markdown()

    def gather(node):
        texts = []
        if isinstance(node, dict):
            for v in node.values(): texts.extend(gather(v))
        elif isinstance(node, list):
            for item in node: texts.extend(gather(item))
        elif isinstance(node, str):
            if t := node.strip(): texts.append(t)
        return texts

    raw_text = "\n".join(gather(doc_dict))
    logger.info(f"[Ingestion] Extracted {len(raw_text)} characters from '{file_path}'")
    end_span(span, output={"raw_text_length": len(raw_text)})
    return raw_text

# ---------- Ollama JSON Caller ----------

# Define a regex to find JSON objects or arrays
JSON_BLOCK_REGEX = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)

def call_ollama_json(
    system_prompt: str,
    user_prompt: str,
    trace=None,
    node_name: str = "ollama_call",
    prompt_name: str | None = None,
    prompt_version: str | None = None,
) -> Any:
    """
    Calls Ollama /api/generate, expecting JSON output.
    Uses regex to robustly extract JSON from conversational text.
    """
    span_input = {
        "system_prompt_preview": system_prompt[:200],
        "user_prompt_preview": user_prompt[:200],
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
    }
    span = start_span(trace, f"{node_name}_llm", span_input)
    logger.info(
        f"[Ollama] Calling model '{OLLAMA_MODEL}' for {node_name} "
        f"(prompt={prompt_name}, version={prompt_version})"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180) # Added timeout
        resp.raise_for_status()
        data = resp.json()

        # 'response' is the key for /api/generate with format=json
        text = data.get("response", "")

        if not text:
            raise ValueError("Empty 'response' field from Ollama.")

        # --- Robust JSON Extraction ---
        match = JSON_BLOCK_REGEX.search(text)
        if not match:
            logger.warning(f"[Ollama] No JSON object or array found in response: {text[:200]}...")
            raise ValueError(f"Ollama did not return valid JSON: {text}")

        json_str = match.group(0)
        # --- End Robust JSON Extraction ---

        out = json.loads(json_str)
        end_span(span, output={"status": "ok"})
        return out

    except requests.RequestException as e:
        logger.error(f"[Ollama] HTTP Error during {node_name}: {e}")
        end_span(span, output={"error": str(e)}, status="error")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"[Ollama] JSON Decode Error during {node_name}. Response: {json_str[:200]}... Error: {e}")
        end_span(span, output={"error": str(e), "raw_response": json_str}, status="error")
        raise
    except Exception as e:
        logger.error(f"[Ollama] General Error during {node_name}: {e}")
        end_span(span, output={"error": str(e)}, status="error")
        raise

# ---------- LangGraph Nodes ----------

def node_extract_contact(state: GraphState) -> GraphState:
    try:
        out = call_ollama_json(
            system_prompt=CONTACT_SYSTEM_PROMPT,
            user_prompt=f"### RESUME TEXT - ###\n{state['raw_text']}",
            trace=state.get("trace"),
            node_name="extract_contact",
            prompt_name=CONTACT_PROMPT_NAME,
            prompt_version=CONTACT_PROMPT_VERSION,
        )
        if not isinstance(out, dict):
            raise ValueError("Contact extractor must return a JSON object")

        # Inlined filter_dict_keys and validation
        allowed = list(ContactInfo.model_fields.keys())
        cleaned = {k: v for k, v in out.items() if k in allowed}
        contact = ContactInfo(**cleaned)
        return {"contact": contact.model_dump()}
    except Exception as e:
        logger.warning(f"[Graph] contact_error: {e}")
        errors = state.get("errors", []) + [f"contact_error: {e}"]
        return {"errors": errors}

def node_extract_education(state: GraphState) -> GraphState:
    try:
        out = call_ollama_json(
            system_prompt=EDUCATION_SYSTEM_PROMPT,
            user_prompt=f"### RESUME TEXT - ###\n{state['raw_text']}",
            trace=state.get("trace"),
            node_name="extract_education",
            prompt_name=EDUCATION_PROMPT_NAME,
            prompt_version=EDUCATION_PROMPT_VERSION,
        )
        # Inlined filter_list_of_dicts and validation
        allowed = list(EducationItem.model_fields.keys())
        items = out.get("education", out) if isinstance(out, dict) else out
        
        validated = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    cleaned = {k: v for k, v in it.items() if k in allowed}
                    try:
                        validated.append(EducationItem(**cleaned).model_dump())
                    except ValidationError:
                        continue # Skip invalid items
        return {"education": validated}
    except Exception as e:
        logger.warning(f"[Graph] education_error: {e}")
        errors = state.get("errors", []) + [f"education_error: {e}"]
        return {"errors": errors}

def node_extract_experience(state: GraphState) -> GraphState:
    try:
        out = call_ollama_json(
            system_prompt=EXPERIENCE_SYSTEM_PROMPT,
            user_prompt=f"### RESUME TEXT - ###\n{state['raw_text']}",
            trace=state.get("trace"),
            node_name="extract_experience",
            prompt_name=EXPERIENCE_PROMPT_NAME,
            prompt_version=EXPERIENCE_PROMPT_VERSION,
        )
        allowed = list(ExperienceItem.model_fields.keys())
        items = out.get("work_experience", out) if isinstance(out, dict) else out

        validated = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    cleaned = {k: v for k, v in it.items() if k in allowed}
                    try:
                        validated.append(ExperienceItem(**cleaned).model_dump())
                    except ValidationError:
                        continue
        return {"work_experience": validated}
    except Exception as e:
        logger.warning(f"[Graph] experience_error: {e}")
        errors = state.get("errors", []) + [f"experience_error: {e}"]
        return {"errors": errors}

def node_extract_projects(state: GraphState) -> GraphState:
    try:
        out = call_ollama_json(
            system_prompt=PROJECTS_SYSTEM_PROMPT,
            user_prompt=f"### RESUME TEXT - ###\n{state['raw_text']}",
            trace=state.get("trace"),
            node_name="extract_projects",
            prompt_name=PROJECTS_PROMPT_NAME,
            prompt_version=PROJECTS_PROMPT_VERSION,
        )
        allowed = list(ProjectItem.model_fields.keys())
        items = out.get("projects", out) if isinstance(out, dict) else out

        validated = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    cleaned = {k: v for k, v in it.items() if k in allowed}
                    try:
                        validated.append(ProjectItem(**cleaned).model_dump())
                    except ValidationError:
                        continue
        return {"projects": validated}
    except Exception as e:
        logger.warning(f"[Graph] projects_error: {e}")
        errors = state.get("errors", []) + [f"projects_error: {e}"]
        return {"errors": errors}

def node_extract_skills(state: GraphState) -> GraphState:
    try:
        out = call_ollama_json(
            system_prompt=SKILLS_SYSTEM_PROMPT,
            user_prompt=f"### RESUME TEXT - ###\n{state['raw_text']}",
            trace=state.get("trace"),
            node_name="extract_skills",
            prompt_name=SKILLS_PROMPT_NAME,
            prompt_version=SKILLS_PROMPT_VERSION,
        )
        candidate = out.get("skills", out) if isinstance(out, dict) else out

        skills: List[str] = []
        if isinstance(candidate, list):
            skills = [str(s).strip() for s in candidate if str(s).strip()]
        elif isinstance(candidate, str):
            skills = [s.strip() for s in candidate.split(",") if s.strip()]
        
        return {"skills": skills}
    except Exception as e:
        logger.warning(f"[Graph] skills_error: {e}")
        errors = state.get("errors", []) + [f"skills_error: {e}"]
        return {"errors": errors}

def node_extract_cert_lang_other(state: GraphState) -> GraphState:
    try:
        out = call_ollama_json(
            system_prompt=CERT_LANG_OTHER_SYSTEM_PROMPT,
            user_prompt=f"### RESUME TEXT - ###\n{state['raw_text']}",
            trace=state.get("trace"),
            node_name="extract_cert_lang_other",
            prompt_name=CERT_LANG_OTHER_PROMPT_NAME,
            prompt_version=CERT_LANG_OTHER_PROMPT_VERSION,
        )
        if not isinstance(out, dict):
            raise ValueError("Expected dict with certifications/languages/other_sections")

        # Certifications
        allowed_cert = list(CertificationItem.model_fields.keys())
        cert_items = out.get("certifications", [])
        cert_valid = []
        if isinstance(cert_items, list):
            for c in cert_items:
                if isinstance(c, dict):
                    cleaned = {k: v for k, v in c.items() if k in allowed_cert}
                    try:
                        cert_valid.append(CertificationItem(**cleaned).model_dump())
                    except ValidationError:
                        continue
        
        # Languages
        allowed_lang = list(LanguageItem.model_fields.keys())
        lang_items = out.get("languages", [])
        lang_valid = []
        if isinstance(lang_items, list):
            for l in lang_items:
                if isinstance(l, dict):
                    cleaned = {k: v for k, v in l.items() if k in allowed_lang}
                    try:
                        lang_valid.append(LanguageItem(**cleaned).model_dump())
                    except ValidationError:
                        continue
        
        other = out.get("other_sections", {})
        if not isinstance(other, dict):
            other = {}

        return {
            "certifications": cert_valid,
            "languages": lang_valid,
            "other_sections": other,
        }
    except Exception as e:
        logger.warning(f"[Graph] cert_lang_other_error: {e}")
        errors = state.get("errors", []) + [f"cert_lang_other_error: {e}"]
        return {"errors": errors}

def node_merge(state: GraphState) -> GraphState:
    span = start_span(state.get("trace"), "merge_results", {})
    logger.info("[Merge] Combining all partial results")
    try:
        resume = ResumeSchema(
            contact=ContactInfo(**(state.get("contact") or {})),
            education=[EducationItem(**e) for e in state.get("education", [])],
            work_experience=[ExperienceItem(**w) for w in state.get("work_experience", [])],
            projects=[ProjectItem(**p) for p in state.get("projects", [])],
            skills=state.get("skills", []) or [],
            certifications=[CertificationItem(**c) for c in state.get("certifications", [])],
            languages=[LanguageItem(**l) for l in state.get("languages", [])],
            other_sections=state.get("other_sections", {}) or {},
        )
        resume_dict = resume.model_dump()
        end_span(span, {"keys": list(resume_dict.keys())})
        return {"resume": resume_dict}
    except Exception as e:
        end_span(span, {"error": str(e)}, status="error")
        logger.warning(f"[Graph] merge_error: {e}")
        errors = state.get("errors", []) + [f"merge_error: {e}"]
        return {"errors": errors}


# ---------- Build LangGraph ----------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("extract_contact", node_extract_contact)
    graph.add_node("extract_education", node_extract_education)
    graph.add_node("extract_experience", node_extract_experience)
    graph.add_node("extract_projects", node_extract_projects)
    graph.add_node("extract_skills", node_extract_skills)
    graph.add_node("extract_cert_lang_other", node_extract_cert_lang_other)
    graph.add_node("merge", node_merge)

    # All nodes run in parallel after ingestion
    graph.set_entry_point("extract_contact")
    graph.add_edge("extract_contact", "extract_education")
    graph.add_edge("extract_education", "extract_experience")
    graph.add_edge("extract_experience", "extract_projects")
    graph.add_edge("extract_projects", "extract_skills")
    graph.add_edge("extract_skills", "extract_cert_lang_other")
    graph.add_edge("extract_cert_lang_other", "merge")
    graph.add_edge("merge", END)

    return graph.compile()


# ---------- Public API ----------

def parse_resume_to_json(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    trace_span = start_trace("resume_parser_pipeline", {"file_path": file_path})
    try:
        raw_text = extract_raw_text_with_docling(file_path, trace=trace_span)
        app = build_graph()
        
        initial_state: GraphState = {
            "raw_text": raw_text,
            "trace": trace_span,
            "errors": [],
        }

        final_state = app.invoke(initial_state)

        # Use the final merged resume if available
        resume_dict = final_state.get("resume")
        if not resume_dict:
            # Fallback: create from partials if merge failed
            logger.warning("[Pipeline] Merge node failed, creating fallback resume.")
            resume = ResumeSchema(
                contact=ContactInfo(**(final_state.get("contact") or {})),
                education=[EducationItem(**e) for e in final_state.get("education", [])],
                work_experience=[ExperienceItem(**w) for w in final_state.get("work_experience", [])],
                projects=[ProjectItem(**p) for p in final_state.get("projects", [])],
                skills=final_state.get("skills", []) or [],
                certifications=[CertificationItem(**c) for c in final_state.get("certifications", [])],
                languages=[LanguageItem(**l) for l in final_state.get("languages", [])],
                other_sections=final_state.get("other_sections", {}) or {},
            )
            resume_dict = resume.model_dump()

        errors = final_state.get("errors", [])
        if errors:
            logger.warning(f"[Pipeline] Completed with warnings: {errors}")
            resume_dict["_parser_warnings"] = errors

        end_span(
            trace_span,
            output={"status": "completed", "has_warnings": bool(errors)},
            status="success",
        )
        return resume_dict

    except Exception as e:
        logger.error(f"[Pipeline] Fatal error: {e}", exc_info=True)
        end_span(trace_span, output={"error": str(e)}, status="error")
        raise

# ---------- CLI Entrypoint ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph + Docling + Ollama + Langfuse Resume Parser")
    parser.add_argument("input", help="Path to resume file (.pdf / .docx / .doc)")
    parser.add_argument("-o", "--output", help="Path to output JSON", default="resume_output.json")
    args = parser.parse_args()

    try:
        result = parse_resume_to_json(args.input)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved structured resume JSON to: {args.output}")
    except Exception as e:
        logger.error(f"Failed to parse resume: {e}")
        # Optionally save a minimal error JSON
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"error": str(e)}, f, indent=2)
            
            
# uv run gemini.py Resume.pdf -o gemini_output.json