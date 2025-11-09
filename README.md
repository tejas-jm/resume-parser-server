# 🧠 Resume Parser using Docling, SLM, LangGraph & Langfuse

## 📄 Overview
This project implements an **intelligent, layout-aware Resume Parser** that extracts structured data such as contact details, education, experience, projects, skills, and certifications from resumes in **PDF**, **DOCX**, and **DOC** formats.

It combines:
- **Docling** for layout-aware document parsing  
- **Qwen3 1.7B** (served via **Ollama**) for language understanding  
- **LangGraph** for modular LLM orchestration  
- **Pydantic** for schema enforcement and validation  
- **Langfuse** for end-to-end tracing and observability  

---

## 🚀 Key Features

| Component | Purpose |
|------------|----------|
| **Docling** | Extracts text and layout from resumes, handling 2-column formats and OCR. |
| **LangGraph** | Orchestrates multiple LLM calls as nodes (Contact, Education, Experience, etc.). |
| **Ollama + Qwen3:1.7B** | Performs semantic extraction with structured outputs following Pydantic schemas. |
| **Pydantic** | Defines strict JSON schemas and validates extracted data. |
| **Langfuse** | Provides live tracing, logging, and observability for each LLM call. |

---

## 🏗️ Architecture

```
Input (PDF/DOC/DOCX)
   │
   ▼
[Docling Ingestion + OCR + Layout Parsing]
   │
   ▼
[LangGraph Orchestrator]
   │
   ├──▶ Node 1: Contact Info
   ├──▶ Node 2: Education
   ├──▶ Node 3: Work Experience
   ├──▶ Node 4: Projects
   ├──▶ Node 5: Skills
   └──▶ Node 6: Certifications / Languages / Other
   │
   ▼
[Pydantic Validation + JSON Merge]
   │
   ▼
Output: Nested Resume JSON
```

---

## ⚙️ Tech Stack

| Technology | Description |
|-------------|-------------|
| **Python 3.11+** | Primary development language |
| **Docling** | PDF/DOCX document parser with OCR |
| **Ollama** | Local model server for Qwen3 |
| **Qwen3:1.7B** | Small language model used for extraction |
| **LangGraph** | Manages multi-node LLM pipelines |
| **Pydantic v2** | Schema validation for structured output |
| **Langfuse (Self-Hosted)** | Observability and tracing layer |

---

## 🧩 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/tejasjm/resume-parser-ai.git
   cd resume-parser-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Langfuse**
   - Launch your local Langfuse instance (Docker)
   - Create a project named `Graph_Trace`
   - Add the following environment variables:
     ```bash
     export LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
     export LANGFUSE_SECRET_KEY=sk-lf-yyyyyyyy
     export LANGFUSE_BASE_URL=http://localhost:3000
     ```

4. **Pull Qwen3 Model with Ollama**
   ```bash
   ollama pull qwen3:1.7b
   ```

---

## 🧠 How It Works

### 1️⃣ Ingestion
Docling parses resumes into structured text, preserving layout, tables, and OCR details.

### 2️⃣ Node-based Orchestration
LangGraph runs multiple nodes, each responsible for a different section of the resume.  
Each node calls Ollama with:
- A **system prompt** defining schema + constraints
- A **user prompt** containing the raw resume text

### 3️⃣ Structured Output
Ollama’s **Structured Output API** enforces schema conformance via:
```python
format = MySchema.model_json_schema()
```
and validation through:
```python
MySchema.model_validate_json(response.message.content)
```

### 4️⃣ Validation & Merge
Pydantic ensures each field matches the expected type, and all sections merge into a single `ResumeSchema` object.

### 5️⃣ Tracing
Each node call and the overall pipeline are logged to **Langfuse**, enabling full trace visibility and debugging.

---

## 📊 Example Output

```json
{
  "contact": {
    "name": "John Doe",
    "email": "Sample@gmail.com",
    "phone": "+91 099999999",
    "location": "Delhi",
    "linkedin": "linkedin.com/in/username",
    "github": "github.com/username"
  },
  "education": [
    {
      "institution": "VNR VJIET",
      "degree": "B.Tech",
      "field_of_study": "Computer Science",
      "start_date": "2021",
      "end_date": "2025"
    }
  ],
  "skills": ["Python", "LangGraph", "FastAPI", "LLMs"],
  "_parser_warnings": []
}
```

---

## 🧪 Running the Parser

```bash
python resume_parser_langgraph.py path/to/Resume.pdf -o output.json
```

### Logs
```
INFO - [Ingestion] Extracted 12707 characters from 'Resume.pdf'
INFO - [Ollama] Calling model 'qwen3:1.7b' for extract_contact...
INFO - [Pipeline] Completed successfully
```

---

## 🔍 Tracing in Langfuse

Each run creates a **root trace** and **child spans** per node:

- `resume_parser_pipeline`
  - `extract_contact_llm`
  - `extract_education_llm`
  - `extract_experience_llm`
  - ...
  - `merge_results`

Traces include:
- Input/output payloads
- Model metadata
- Duration
- Errors and warnings

Accessible via your Langfuse dashboard at `http://localhost:3000`.

---

## 📈 Future Work

- Fine-tune Qwen3 on domain-specific resume data.
- Add named-entity normalization for consistent institution and company names.
- Visual dashboard for LangGraph state visualization.
- Parallelize section extraction for performance.

---

## 🧑‍💻 Author

**Tejas JM**  
AI Developer • Hyderabad, India  
[GitHub](https://github.com/tejas-jm) | [Portfolio](https://tejasjm.com)

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to modify and use it.

---

## ⭐ Acknowledgments

- [Docling](https://github.com/docling-project/docling)  
- [Ollama](https://ollama.com/)  
- [LangGraph](https://github.com/langchain-ai/langgraph)  
- [Langfuse](https://langfuse.com/)  
- [Qwen Team](https://huggingface.co/Qwen)
