import os
import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from collections import defaultdict

# ========== Schema ==========
class ResumeInfo(BaseModel):
    name: str
    skills: list = []
    education: list = []
    work_experience: list = []
    projects: list = []
    candidate_summary: str
    work_experience_years: int
    filepath: str

# ========== PDF → Text ==========
def pdf_to_text(pdf_path: str) -> str:
    elements = partition_pdf(pdf_path, strategy="fast")
    return "\n".join(el.text.strip() for el in elements if el.text and el.text.strip())

# ========== JSON Schema Helper ==========
def get_json_schema():
    return ResumeInfo.model_json_schema()

# ========== LLM JSON Extraction Call ==========
def LLM_call(prompt: str, schema: dict) -> Dict[str, Any]:
    from datetime import datetime

    client = OpenAI(base_url="http://172.16.2.214:8000/v1", api_key="-")
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"guided_json": schema}
    )

    output_text = response.choices[0].message.content

    # Optional: Create a timestamped log file or just write to a static file
    with open("llm_responses_log.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Response at {datetime.now()} ---\n")
        f.write(output_text + "\n\n")

    print(output_text)
    return json.loads(output_text)

# ========== Resume Parsing ==========
def parsing_helper(markdown_text: str, filepath: str) -> dict:
    schema = get_json_schema()
    prompt = f"""
You are a precise and strict **Information Extraction Assistant** trained to extract structured data from **unstructured CV/resume text**.

Your objective is to return only a valid JSON object that matches the schema provided below. The source file path for this CV is: "{filepath}"

---

### 🎯 Task Objective:

Extract key structured information from the raw CV text. You must:

1. **Extract only keywords or phrases** (avoid full sentences).
2. **Normalize inconsistent formats**, including:
   - Capitalization (e.g., convert "python", "PYTHON" → "Python")
   - Abbreviations and synonyms to canonical forms, such as:
     - "bscs", "B.Sc. CS", "Bachelors in Computer Science" → `"BSc Computer Science"`
     - "nlp", "Natural Language Processing" → `"NLP"`
     - "ai", "Artificial Intelligence" → `"AI"`
     - "numpy", "NumPy" → `"NumPy"`
     - "ml", "machine learning" → `"Machine Learning"`
     - "ba", "Business Analyst" → `"Business Analyst"`
3. Extract only what is **explicitly stated** in the CV text — no guessing or inferring.

---

###  Special Rule for `work_experience_years`:

- You **must calculate `work_experience_years`** only from **clearly defined start and end dates** present under the **work experience** section.
- Accept date formats like:  
  - "1 Nov 2021 -Current"  
  - "1 Aug 2022 – 30 Oct 2022"  
  - Month abbreviations (Jan, Feb, etc.) and numeric formats like "02/2021 – 10/2022"
- If the end date is `"Current"` or `"Present"`, use today’s date for calculation.
- Do **not** include time from:
  - Projects
  - Internships not listed under work experience
  - Education or training
- Round total experience to 1 decimal place.

---

###Special Rule for `candidate_summary`:

- If a summary section is explicitly present in the CV (e.g., under "Profile", "Summary", or "About Me"), extract a concise 2–3 sentence version.
- If **no summary is stated**, **generate** a professional 2–3 sentence summary using **only clearly stated information** from the CV, focusing on:
  - Key skills
  - Work experience highlights
  - Technologies used
- Do **not** invent or infer any unstated experience or traits.

---

### 📦 Output JSON Schema:

{json.dumps(schema, indent=2)}

If a value is missing or not mentioned, use:

- null for missing string fields  
- [] for missing lists  

`filepath` must exactly match "{filepath}".  
No extra formatting — output valid JSON only.

---

For `work_experience`:
- Extract company names, job titles, and durations
- Focus on recent/current positions first
- Normalize company name variations
- Ignore roles listed under other sections (e.g., "Projects" or "Academics")
"""
    return LLM_call(prompt, schema)
def cv_parser_pipeline(path: str) -> List[dict]:
    candidates = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            continue
        text = pdf_to_text(file_path)
        structured = parsing_helper(text, file_path)
        candidates.append(structured)
    return candidates

# ========== Qdrant Setup ==========
client = QdrantClient("localhost", port=6333)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
required_fields = ["skills", "education", "work_experience", "projects"]

def initialize_collection(collection_name: str = "cv_data"):
    """Delete the collection if it exists, then create it from scratch"""
    try:
        # Check if the collection exists
        client.get_collection(collection_name)
        # If no exception is raised, collection exists -> delete it
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        # Collection does not exist
        print(f"No existing collection named: {collection_name}")

    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            f: VectorParams(size=384, distance=Distance.COSINE)
            for f in required_fields
        }
    )
    print(f"Created collection: {collection_name}")

def zero_vector(dim=384) -> List[float]:
    return [0.0] * dim

def join_and_embed(field_list: list, model=embedding_model) -> List[float]:
    if not field_list:
        return zero_vector()

    flat_strings = []

    for item in field_list:
        if isinstance(item, str):
            flat_strings.append(item)
        elif isinstance(item, dict):
            # Join all string values in the dict
            for value in item.values():
                if isinstance(value, str):
                    flat_strings.append(value)
                elif isinstance(value, list):
                    # If nested list (e.g. technologies), flatten that too
                    flat_strings.extend([v for v in value if isinstance(v, str)])
        elif isinstance(item, list):
            # Flatten nested lists of strings (e.g., [["Python", "C++"]])
            flat_strings.extend([v for v in item if isinstance(v, str)])

    text = " ".join(flat_strings)

    if not text.strip():
        return zero_vector()

    return model.encode([text])[0].tolist()

def insert_candidate(candidate: dict, collection_name="cv_data"):
    vector_data = {f: join_and_embed(candidate.get(f, [])) for f in required_fields}
    payload = {
        "name": candidate.get("name"),
        "filepath": candidate.get("filepath"),
        "skills": candidate.get("skills", []),
        "education": candidate.get("education", []),
        "work_experience": candidate.get("work_experience", []),
        "projects": candidate.get("projects", []),
        "candidate_summary": candidate.get("candidate_summary", ""),
        "work_experience_years": candidate.get("work_experience_years", None),
    }
    pid = candidate.get("id", hash(candidate.get("name", "")) & 0xFFFFFFFFFFFFFFFF)
    client.upsert(collection_name=collection_name, points=[PointStruct(
        id=pid,
        vector=vector_data,
        payload=payload
    )])

def create_vec_db(candidates: List[dict]):
    for cand in candidates:
        insert_candidate(cand)

# ========== Job Description Parsing ==========
def job_description_parser(job_description: str) -> dict:
    schema = get_json_schema()
    prompt = f"""
You are a highly accurate and strict **Information Extraction Assistant**.

Your task is to extract **only explicitly stated information** from a **job description** and convert it into a structured JSON format. The data must be extracted as **normalized, lowercase keywords** and must strictly follow the rules and schema provided.

---

### Extraction Instructions:
1. Extract only **keywords or short phrases** — avoid full sentences.
2. All extracted data must be:
   - **Lowercased**
   - **Normalized** for abbreviations and variants. Examples:
     - "Bachelors in Computer Science", "BSCS", "BS in CS" → `"bscs"`
     - "Natural Language Processing" → `"nlp"`
     - "NumPy", "numpy" → `"numpy"`
     - "Machine Learning" → `"machine learning"`

3. Do **not infer or assume** any information — extract only what is **explicitly stated**.
4. Count `work_experience_years` **only if a specific number of years is clearly mentioned**.

---

### Required JSON Output Schema:

{json.dumps(schema, indent=2)}

If a field is not present in the job description, return empty lists or null/0 for years.

Output must be valid JSON only—no text.

Job Description:
{job_description}
"""
    return LLM_call(prompt, schema)

# ========== Qdrant Search & Ranking ==========
def searching_Qdrant(parsed: dict, top_k: int = 5, weights: dict = None) -> Dict[int, float]:
    job_vectors = {f: join_and_embed(parsed.get(f, [])) for f in required_fields}
    if weights is None:
        weights = {f: 0.25 for f in required_fields}
    else:
        total = sum(weights.values())
        weights = {f: (weights[f]/total) for f in weights}

    all_hits = {f: client.search(
        collection_name="cv_data",
        query_vector=(f, job_vectors[f]),
        limit=top_k,
        with_payload=True,
        with_vectors=False
    ) for f in required_fields}

    scores = defaultdict(float)
    for f, hits in all_hits.items():
        for hit in hits:
            scores[hit.id] += hit.score * weights.get(f, 0)
    return scores

def sorting_candidates(score_board: Dict[int, float], top_k: int = 5) -> List[dict]:
    ranked = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[:top_k]
    candidate_ids = {cid for cid, _ in ranked}
    results = []
    offset = 0
    while len(results) < top_k:
        pts, next_offset = client.scroll("cv_data", with_payload=True, with_vectors=False, limit=100, offset=offset)
        for pt in pts:
            if pt.id in candidate_ids:
                info = pt.payload.copy()
                info["id"] = pt.id
                info["score"] = round(score_board[pt.id], 4)
                results.append(info)
        if not next_offset:
            break
        offset = next_offset
    return results

# ========== Detailed Analysis Prompt ==========
def analysis(job_description, top_candidates, top_k=5):
    # Format candidate information for the prompt
    candidate_details = []
    for idx, cand in enumerate(top_candidates):
        cand_str = f"\n\nCandidate #{idx+1}: {cand.get('name', 'N/A')}"
        cand_str += f"\n- Summary: {cand.get('candidate_summary', 'No summary available')}"
        cand_str += f"\n- Work Experience: {', '.join(cand.get('work_experience', ['No experience listed']))}"
        cand_str += f"\n- Experience Years: {cand.get('work_experience_years', 'N/A')}"
        cand_str += f"\n- Skills: {', '.join(cand.get('skills', []))}"
        cand_str += f"\n- Education: {', '.join(cand.get('education', []))}"
        candidate_details.append(cand_str)
    
    candidates_block = "\n".join(candidate_details)
    
    prompt = f"""
You are an expert technical recruiter and AI career advisor. Use the information provided below to perform an in-depth candidate evaluation.

---

### Job Description
{job_description}

---

### Top {top_k} Candidates
{candidates_block}

---

### Task
Analyze the candidates with respect to the job description and:

1. **Compare** each candidate's qualifications in terms of:
   - Skills match
   - Relevant work experience
   - Education background
   - Career summary alignment
   - Total experience years

2. **Rank** candidates from most to least suitable based on:
   - Overall match to job requirements
   - Depth of relevant experience
   - Conciseness and relevance of candidate summary

3. **Highlight Key Differentiators**:
   - Unique strengths from candidate summaries
   - Notable companies/positions in work experience
   - Experience years compared to job requirements

4. **Provide Insights**:
   - 1-2 sentence assessment for each candidate
   - Suggestions for interview focus areas
   - Potential concerns based on work history gaps

5. **Format**:
   - Use professional, concise language
   - Present results in clear markdown
   - Start with your top recommended candidate
   - Do not disclose vector scores

---
"""
    client = OpenAI(
        base_url="http://172.16.2.214:8000/v1", 
        api_key="-" 
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


###--------------------------------------------------


##writing a function for chatbot
def Resume_data():
    all_data = []
    offset = None

    while True:
        scroll_result, next_offset = client.scroll(
            collection_name="cv_data",
            with_payload=True,
            with_vectors=True,
            limit=100,
            offset=offset
        )

        for point in scroll_result:
            all_data.append({
                "id": point.id,
                "name": point.payload.get("name", "unknown"),
                "filepath": point.payload.get("filepath", ""),
                "skills": point.payload.get("skills", []),
                "education": point.payload.get("education", []),
                "work_experience": point.payload.get("work_experience", []),
                "projects": point.payload.get("projects", []),
                "candidate_summary": point.payload.get("candidate_summary", ""),
                "work_experience_years": point.payload.get("work_experience_years", 0)
            })

        if next_offset is None:
            break
        offset = next_offset

    return all_data
def update_cached_resumes():
    global cached_resume_data
    cached_resume_data = Resume_data() 

def summarize_resumes(resumes, client, model="Qwen/Qwen2.5-32B-Instruct-AWQ", limit=10):
    summary_list = []

    for r in resumes[:limit]:
        try:
            # Use candidate summary if available, otherwise generate one
            if r.get("candidate_summary"):
                summary = r["candidate_summary"]
            else:
                raw_profile = {
                    "name": r.get("name", "Unknown"),
                    "education": r.get("education", []),
                    "skills": r.get("skills", []),
                    "work_experience": r.get("work_experience", []),
                    "experience_years": r.get("work_experience_years", 0)
                }
                prompt = f"Create a 1-sentence professional summary for: {json.dumps(raw_profile)}"
                result = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
                summary = result.choices[0].message.content.strip()
            
            exp_years = r.get("work_experience_years", 0)
            summary_list.append(f"- **{r.get('name', 'Unknown')}** ({exp_years} yrs exp): {summary}")

        except Exception as e:
            summary_list.append(f"- **{r.get('name', 'Unknown')}**: ❌ Failed to summarize ({e})")

    return "\n".join(summary_list)


cached_resume_data = []

def estimate_tokens(text):
    """Rough estimation: 1 word ≈ 1.5 tokens."""
    return int(len(text.split()) * 1.5)


def truncate_history(chat_history, system_prompt, max_tokens=3500):
    """Remove oldest messages until history fits within token limit."""
    while estimate_tokens(system_prompt + "\n".join(f"{r}: {m}" for r, m in chat_history)) > max_tokens:
        if len(chat_history) > 2:
            chat_history.pop(1)
        else:
            break
    return chat_history


def summarize_history(chat_history, client):
    """Summarize older messages and replace them with a single summary block."""
    if len(chat_history) <= 5:
        return chat_history 
    to_summarize = chat_history[:-4]
    preserved = chat_history[-4:]

    try:
        history_text = "\n".join(f"{role}: {msg}" for role, msg in to_summarize)
        summary_result = client.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "Summarize this conversation briefly:"},
                {"role": "user", "content": history_text}
            ]
        )
        summary = summary_result.choices[0].message.content.strip()
        summarized_history = [("system", f"Summary of earlier conversation:\n{summary}")]
        return summarized_history + preserved

    except Exception as e:
        return [("system", f"Summarization failed: {e}")] + preserved

def chatbot(chat_history, user_prompt):
    global cached_resume_data

    client = OpenAI(
        base_url="http://172.16.2.214:8000/v1",
        api_key="-"
    )

    resume_summary = summarize_resumes(cached_resume_data,client=client)
    system_prompt = f"""
You are a structured, intelligent, and professional Resume Screening Assistant.

Your knowledge is restricted to the following parsed resume dataset:

### Resume Dataset Snapshot
{resume_summary}

Responsibilities:
- Parse job descriptions into normalized keyword fields: skills, education, work_experience, projects.
- Match job descriptions with parsed resumes using semantic similarity (Qdrant vector search).
- Recommend top candidates with professional justifications.
- Respond accurately to follow-up queries using only explicitly available data.
- Never infer or fabricate information.
- Only respond to queries related to resumes and CVs. For unrelated queries, politely deny.

Style:
- Maintain a concise, formal tone.
- Use bullet points or markdown tables where useful.
- Do not disclose internal vector details unless explicitly asked.
""".strip()

    chat_history.append(("user", user_prompt))

    # Estimate tokens and apply summarization/truncation if needed
    full_text = system_prompt + "\n" + "\n".join(f"{r}: {m}" for r, m in chat_history)
    if estimate_tokens(full_text) > 3500:
        chat_history = summarize_history(chat_history, client)
        chat_history = truncate_history(chat_history, system_prompt)

    # Decide interaction type
    is_job_desc = "job description" in user_prompt.lower() or len(user_prompt.split()) > 80

    try:
        if is_job_desc:
            parsed = job_description_parser(user_prompt)
            scores = searching_Qdrant(parsed, top_k=5)
            top_candidates = sorting_candidates(scores, top_k=5)
            response = analysis(user_prompt, top_candidates, candidates=None, top_k=5)
        else:
            # Construct message payload
            messages = [{"role": "system", "content": system_prompt}]
            for role, msg in chat_history:
                messages.append({"role": role, "content": msg})
            messages.append({"role": "user", "content": user_prompt})

            result = client.chat.completions.create(
                model="Qwen/Qwen2.5-32B-Instruct-AWQ",
                messages=messages
            )
            response = result.choices[0].message.content

    except Exception as e:
        response = f"Error: {e}"

    chat_history.append(("assistant", response))
    return chat_history, response
