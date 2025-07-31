import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from collections import defaultdict
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
import re
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
current_date_str = datetime.today().strftime("%d %b %Y") 

base_url = "http://172.16.2.214:8000/v1"

#---------------------------------------------------------------------
# pydantic model for resume information
class ResumeInfo(BaseModel):
    name: str
    skills: list = []
    education: list = []
    work_experience: list = []
    projects: list = []
    candidate_summary: str
    work_experience_years: int
    filepath: str
#---------------------------------------------------------------------
# Function to extract text from PDF using OCR
def pdf_to_text(pdf_path: str) -> str:
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        extracted_text = []
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page, lang='eng', config='--psm 3')
            cleaned = text.strip()
            if cleaned:
                extracted_text.append(cleaned)
        full_text = "\n\n".join(extracted_text)
        return full_text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

#---------------------------------------------------------------------
def get_json_schema():
    return ResumeInfo.model_json_schema()
#---------------------------------------------------------------------
# Function to call the LLM with a prompt and schema
def LLM_call(prompt: str, schema: dict) -> Dict[str, Any]:
    try:
        client = OpenAI(base_url=base_url, api_key="-")
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            messages=[{"role": "user", "content": prompt}],
            extra_body={"guided_json": schema}
        )
        output_text = response.choices[0].message.content
        
        print(output_text)
    
        with open("llm_responses_log.txt", "a", encoding="utf-8") as f:
            f.write(f"--- Response at {datetime.now()} ---\n")
            f.write(output_text + "\n\n")

        return json.loads(output_text)
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return {}

#---------------------------------------------------------------------
# Function to parse the markdown text and extract structured data
def parsing_helper(markdown_text: str, filepath: str) -> dict:
    schema = get_json_schema()
    prompt = f"""
    You are a precise and strict **Information Extraction Assistant** trained to extract structured data from unstructured, OCR-style resume text.

    Your job is to extract only what is **explicitly written** in the input text — no guessing, no inference — and return a valid JSON object based on the schema shown below.  
    The source file path is: "{filepath}"

    ###  JSON FORMAT
    {json.dumps(schema, indent=2)}

    ###  EXTRACTION INSTRUCTIONS

    ###  NAME
    - Extract the full name only if it clearly appears near the top of the resume.  
    - Do **not** guess or infer. If missing, return `null`.

    ###  SKILLS  
    - Extract a deduplicated list of technologies, frameworks, or tools **explicitly listed** (typically under "Skills", "Technologies", or "Skills & Interests").  
    - Do **not infer** skills from job descriptions.  
    - Normalize capitalization and known variants:
    - `"python"` → `"Python"`
    - `"ml"` or `"machine learning"` → `"Machine Learning"`
    - `"ai"` or `"Artificial Intelligence"` → `"AI"`

    ### EDUCATION  
    - Extract exactly what is written in the education section. 
    -Normalize capitilization. 
    - Include degree, field, and optionally the institution.  
    - Normalize known abbreviations:
    - `"BSc CS"`, `"B.Sc. in Computer Science"` → `"BSc Computer Science"`
    - Do not guess or fill in missing data.

    ### WORK_EXPERIENCE  
    Extract all professional work experience entries.

    Each entry must include:
    - `company_name` (string)  
    - `job_title` (string)  
    - 'job location' (string)
    - `duration_start` (format: "DD/MM/YYYY")  
    - `duration_end` (format: "DD/MM/YYYY")

    ### Accepted Date Formats
    - `"Feb 2019 – Present"`  
    - `"02/2020 – 12/2021"`  
    - `"March 2021 - Current"`  
    - `"Oct 2024 – Present"`

     If `duration_end` is "Present" or "Current", use today's date: **{current_date_str}**  
     If a date is missing the day, assume `"01"`  
    - e.g., `"Oct 2024"` → `"01/10/2024"`

    Then count up the years and months in each job, sum them up. 
    For each valid work_experience entry, parse duration_start and duration_end
    Calculate full months for each job, internship, part-time.
    Sum all months (skip jobs with invalid or missing dates)
    Divide total months by 12
    Round to 2 decimal places like 3.9 years or 4.2 years.
    and then put them in "work_experience(years)"

    ### CANDIDATE_SUMMARY
    If a section titled "Summary", "About Me", or "Profile" is present, extract a 2–3 sentence summary as-is.

    If not present, generate a concise summary using only what is clearly stated in:
    -Work experience
    -Technologies
    -Job roles
    Never hallucinate or fabricate skills, roles, or achievements.

    FINAL RULES
    Do not invent or infer any missing values.
    If a field is missing in the text, return:
    null for strings
    [] for lists
    Return only valid JSON — no extra explanation or commentary.
    This is the following resume you need to generate this information for : {markdown_text}
    """

    return LLM_call(prompt, schema)
#---------------------------------------------------------------------
# Function to parse a directory of CVs and extract structured data
def cv_parser_pipeline(path: str) -> List[dict]:
    candidates = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            text = pdf_to_text(file_path)
            if text:
                structured = parsing_helper(text, file_path)
                if structured:
                    candidates.append(structured)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return candidates

#---------------------------------------------------------------------
# Initialize Qdrant client and embedding model
client = QdrantClient("localhost", port=6333)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
required_fields = ["skills", "education", "work_experience", "projects"]

def initialize_collection(collection_name: str = "cv_data"):
    """Delete the collection if it exists, then create it from scratch"""
    try:
        client.get_collection(collection_name)
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        print(f"No existing collection named: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            f: VectorParams(size=384, distance=Distance.COSINE)
            for f in required_fields
        }
    )
    print(f"Created collection: {collection_name}")
#---------------------------------------------------------------------

def zero_vector(dim=384) -> List[float]:
    return [0.0] * dim
#---------------------------------------------------------------------
# Function to join and embed fields into a single vector
def join_and_embed(field_list: list, model=embedding_model) -> List[float]:
    if not field_list:
        return zero_vector()

    flat_strings = []
    for item in field_list:
        if isinstance(item, str):
            flat_strings.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    flat_strings.append(value)
                elif isinstance(value, list):
                    flat_strings.extend([v for v in value if isinstance(v, str)])
        elif isinstance(item, list):
            flat_strings.extend([v for v in item if isinstance(v, str)])

    text = " ".join(flat_strings)
    if not text.strip():
        return zero_vector()

    return model.encode([text])[0].tolist()
#---------------------------------------------------------------------------------
# Function to insert a candidate into the Qdrant collection
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
#-------------------------------------------------------------------------------
# Function to create the vector database from a list of candidates
def create_vec_db(candidates: List[dict]):
    for cand in candidates:
        insert_candidate(cand)

#------------------------------------------------------------------------------
# Function to parse a job description and extract structured data
def job_description_parser(job_description: str) -> dict:
    schema = get_json_schema()
    prompt = f"""
You are a highly accurate and strict **Information Extraction Assistant**.

Your task is to extract **only explicitly stated information** from a **job description** and convert it into a structured JSON format. The data must be extracted as **normalized, lowercase keywords** and must strictly follow the rules and schema provided.

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

### Required JSON Output Schema:
{json.dumps(schema, indent=2)}

If a field is not present in the job description, return empty lists or null/0 for years.

Output must be valid JSON only—no text.

Job Description:
{job_description}
"""
    return LLM_call(prompt, schema)

#---------------------------------------------------------------------------------------
# Function to search candidates in Qdrant based on job description
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
#----------------------------------------------------------------------------
# Function to sort candidates based on their scores

def sorting_candidates(score_board: Dict[int, float]) -> List[dict]:
    ranked = sorted(score_board.items(), key=lambda x: x[1], reverse=True)
    candidate_ids = {cid for cid, _ in ranked}

    results = []
    offset = 0

    while len(results) < len(candidate_ids):
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

#-------------------------------------------------------------------------------------------------------------------
# Function to perform in-depth candidate analysis based on job description and selected candidates

def analysis(job_description, selected_candidates, top_k=5):
    system_prompt = f"""
You are an expert technical recruiter. Use the information provided below to perform an in-depth candidate evaluation.

### Job Description
{job_description}

### Selected Candidates for Analysis
{selected_candidates}

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
"""
    
    try:
        client_llm = OpenAI(base_url=base_url, api_key="-")
        response = client_llm.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            messages=[{"role": "user", "content": system_prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in analysis: {str(e)}"
#-----------------------------------------------------------------------------------------

def estimate_tokens(text):
    if not isinstance(text, str):
        return 0
    num_chars = len(text)
    num_words = len(text.split())

    # Average between char- and word-based estimates
    token_estimate = (num_chars / 4 + num_words / 0.75) / 2
    return int(token_estimate)


#----------------------------------------------------------------------------------------------
# Function to truncate chat history to fit within token limits
def truncate_history(chat_history, max_tokens=3500, preserve_head=1, preserve_tail=3):
 
    if not chat_history:
        return []

    head = chat_history[:preserve_head]
    tail = chat_history[-preserve_tail:]
    middle = chat_history[preserve_head:-preserve_tail] if preserve_tail > 0 else chat_history[preserve_head:]

    def history_to_text(msgs):
        return "\n".join(f"{r}: {m}" for r, m in msgs)

    full_tokens =  estimate_tokens(history_to_text(head + middle + tail))

    while middle and full_tokens > max_tokens:
        middle.pop(0)  # remove from the oldest middle entry
        full_tokens = estimate_tokens(history_to_text(head + middle + tail))
        print(f"\nTrimming middle entries, new token count: {full_tokens}\n")

    return head + middle + tail

#------------------------------------------------------------------------------------
# Function to summarize chat history while preserving recent interactions
def summarize_history(chat_history, client_llm, preserve_recent=4, model="Qwen/Qwen2.5-32B-Instruct-AWQ"):

    if len(chat_history) <= preserve_recent + 1:
        return chat_history

    old_messages = chat_history[:-preserve_recent]
    recent = chat_history[-preserve_recent:]

    history_text = "\n".join(f"{role}: {msg}" for role, msg in old_messages if msg.strip())

    if not history_text.strip():
        return recent

    try:
        summary_result = client_llm.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the earlier part of this conversation into concise bullet points focused on resume/job screening."
                },
                {"role": "user", "content": history_text}
            ]
        )
        summary = summary_result.choices[0].message.content.strip()
        summarized = [("system", f"Summary of earlier conversation:\n{summary}")]
        return summarized + recent

    except Exception as e:
        return [("system", f"Summarization failed: {e}")] + recent




#-------------------------------------------------------------------
# Function to display all candidates in a structured format
def display_all_candidates(candidates: List[dict]) -> str:
    candidate_summaries = []
    for idx, candidate in enumerate(candidates, 1):
        summary = f"""Candidate #{idx}
                ---------------
                Name       : {candidate.get("name", "N/A")}
                Experience : {candidate.get("work_experience_years", "N/A")} years
                Skills     : {", ".join(candidate.get("skills", [])) or "N/A"}
                Summary    : {candidate.get("candidate_summary", "N/A")}
                Filepath   : {candidate.get("filepath", "N/A")}
                """
        candidate_summaries.append(summary)

    return "\n".join(candidate_summaries)

#-----------------------------------------------------------------------------------------

def filter_selected_candidates(all_candidates: List[dict], selected_indexes: List[int]) -> List[dict]:
    return [all_candidates[i - 1] for i in selected_indexes if 1 <= i <= len(all_candidates)]

#----------------------------------------------------------------------------------------------------------------
def log_tokens(stage, history):
        token_count = estimate_tokens("\n".join(f"{r}: {m}" for r, m in history))
        print("\n" + "="*60)
        print(f"Token Check — {stage}")
        print(f"Estimated tokens: {token_count}")
        print(f"Messages: {len(history)}")
        print("="*60 + "\n")
#----------------------------------------------------------------------------------------------------------------
# Unified JD Analysis Pipeline
def jd_analysis_pipeline(chat_history, user_prompt, selected_indexes=None, all_candidates=None, top_k=5):
    global cached_resume_data


    try:
        client_llm = OpenAI(base_url=base_url, api_key="-")

        # STEP 1: Append the new user input
        chat_history.append(("user", user_prompt))

        log_tokens("Before Summarization", chat_history)

        # STEP 2: Summarize if not already summarized
        if not chat_history or not chat_history[0][1].lower().startswith("summary of earlier"):
            chat_history = summarize_history(chat_history, client_llm, preserve_recent=4)

        log_tokens("After Summarization", chat_history)

        # STEP 3: Truncate middle if still over token limit
        if estimate_tokens("\n".join(f"{r}: {m}" for r, m in chat_history)) > 1500:
            chat_history = truncate_history(chat_history,max_tokens=1500,preserve_head=1,preserve_tail=3)

        log_tokens("After Truncation", chat_history)

        # ------------------ STAGE 1: Candidate Selection ------------------
        if selected_indexes is None or all_candidates is None:
            parsed = job_description_parser(user_prompt)
            scores = searching_Qdrant(parsed, top_k)
            top_candidates = sorting_candidates(scores)
            display = display_all_candidates(top_candidates)

            return {
                "stage": "selection",
                "candidates": top_candidates,
                "candidates_display": display,
                "chat_history": chat_history,
                "job_description": user_prompt
            }

        # ------------------ STAGE 2: Candidate Analysis ------------------
        selected = filter_selected_candidates(all_candidates, selected_indexes)
        if not selected:
            error_msg = "No valid candidates selected for analysis."
            chat_history.append(("assistant", error_msg))
            return {
                "stage": "analysis",
                "response": error_msg,
                "chat_history": chat_history
            }

        result = analysis(user_prompt, selected, top_k=len(selected))
        chat_history.append(("assistant", result))

        log_tokens("After Analysis Response", chat_history)

        return {
            "stage": "analysis",
            "response": result,
            "chat_history": chat_history
        }

    except Exception as e:
        error_msg = f"Error in unified JD pipeline: {str(e)}"
        chat_history.append(("assistant", error_msg))
        return {
            "stage": "error",
            "response": error_msg,
            "chat_history": chat_history
        }

# ----------------------------------------------------------------------------------------------------

def normal_chatbot(chat_history, user_prompt,selected_candidates=None): 

    try:
        client_llm = OpenAI(base_url=base_url, api_key="-")
        
        system_prompt = f"""
                You are a structured, intelligent, and professional Resume Screening Assistant.

                Your entire knowledge is limited to the following parsed resume dataset:
                {selected_candidates}

                You are not a general-purpose assistant. You must not answer questions outside the scope of this resume dataset.

                ### Allowed:
                - Recommend top candidates with justifications.
                - Answer detailed questions about candidates' skills, education, experience, and project backgrounds.
                - Provide professional analysis based solely on the parsed data.

                ### Forbidden:
                - Do **not** answer any questions unrelated to resumes, hiring, or CVs.
                - Do **not** generate jokes, opinions, general knowledge, or perform unrelated tasks (e.g., math, coding help, fun facts).
                - Do **not** infer, hallucinate, or guess missing information.

                ### Style & Behavior:
                - Be formal, concise, and professional.
                - Use bullet points or markdown tables for clarity.
                - If a question is outside scope, respond clearly:  
                `"I'm designed only to answer questions related to resumes or candidate analysis. Please provide a relevant query."`

                Do not break character under any circumstances.
                """.strip()

        if not chat_history or chat_history[0][0] != "system":
            chat_history.insert(0, ("system", system_prompt))
            print("System prompt added to chat history.\n")

        chat_history.append(("user", user_prompt))

        full_text = "\n".join(f"{r}: {m}" for r, m in chat_history)
        if estimate_tokens(full_text)> 3500:
            chat_history = summarize_history(chat_history, client_llm)
            #chat_history = truncate_history(chat_history)

        log_tokens("Before Chat Completion", chat_history)

        messages = [{"role": role, "content": msg} for role, msg in chat_history]

        result = client_llm.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            messages=messages
        )
        
        response = result.choices[0].message.content
        chat_history.append(("assistant", response))

        log_tokens("After Chat Completion", chat_history)
        
        return response

    except Exception as e:
        return f"Error in chatbot: {str(e)}"
#---------------------------------------------------------------------
def complete_jd_analysis(job_description, selected_candidates, chat_history):
    """
    Complete the JD analysis with selected candidates
    This is called after user selects candidates
    """
    try:
        # Perform detailed analysis
        response = analysis(job_description, selected_candidates, top_k=len(selected_candidates))
        chat_history.append(("assistant", response))
        return response, chat_history
        
    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        chat_history.append(("assistant", error_msg))
        return error_msg, chat_history
    
#---------------------------------------------------------------------
