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
from transformers import AutoTokenizer
import tiktoken 
from pdf2image import convert_from_path
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
    work_experience_years: float
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

        print(f" \nouput_text :  {output_text} \n")

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
    - Include degree, field, and the institution. (e.g., `"BSc Computer Science, University of XYZ"`).
    - Extract the location of the instituiton as well, as `"education_location"` (e.g., `"Islamabad, Pakistan"`).
    - Normalize known abbreviations:
    - `"BSc CS"`, `"B.Sc. in Computer Science"` → `"BSc Computer Science"`
    - Do not guess or fill in missing data.

    ### WORK_EXPERIENCE  
    Extract all professional work experience entries., dont include extra details like job descriptions or responsibilities.
    - Each entry must be clearly labeled as a job, internship, or part-time role.
    -Extract the following fields from each entry:
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
    print("Loading models and initializing Qdrant client...")
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
        print(f"Inserting candidate: {cand.get('name', 'Unknown')} from {cand.get('filepath', 'Unknown')}")
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

markdown text example : 
```markdown
### Job Description Example
We are looking for a **Senior Software Engineer** with at least **5 years of experience** in **Python** and **Django**.
The candidate should have a strong background in **web development** and be familiar with **RESTful APIs**.
Preferred qualifications include experience with **Docker**, **Kubernetes**, and **AWS**.
### example output 
###json schema:
    "skills": ["python", "django", "web development", "restful apis", "docker", "kubernetes", "aws"],
    "education": [],
    "work_experience": [],
    "work_experience_years": 5

If a field is not present in the job description, return empty lists or null/0 for years.
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


    print(f"\nTotal candidates sorted: {len(results)}")
    print(f"Top candidates: {results[:5]}\n")

    return results


def summarizer(candidates: List[dict]) -> List[dict]:
    """"summarizes the information that is provided in the candidates list into a more concise format"""""
    summarize = []
    for candidate in candidates: 
        summary = {
            "name": candidate.get("name", "N/A"),
            "skills": list(set(candidate.get("skills", []))),
            "education": list(set(candidate.get("education", []))),
            "work_experience_years": candidate.get("work_experience_years", 0),
            "filepath": candidate.get("filepath", "N/A")
        }
        summarize.append(summary)
        print(f"Summarized candidate: {summary['name']} with {summary['work_experience_years']} years of experience.")

    return summarize


#-------------------------------------------------------------------------------------------------------------------
# Function to perform in-depth candidate analysis based on job description and selected candidates

def analysis(job_description, selected_candidates, top_k=5):

    if selected_candidates is None or not selected_candidates:
        return "No candidates selected for analysis."

    if len(selected_candidates) > 5:


        print("Summarizing candidates for analysis to prevent token overflow...")
        print(f"candidates before summarizing :  {len(selected_candidates)} ,(for analysis).")


        selected_candidates = summarizer(selected_candidates)

        print(f"candidates after summarizing :  {len(selected_candidates)} ,(for analysis).")

    user_prompt = f"""
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

**Format**:
   - Use professional, concise language
   - Present results in clear markdown
   - Start with your top recommended candidate
   - Do not disclose vector scores
"""
    
    try:
        client_llm = OpenAI(base_url=base_url, api_key="-")
        response = client_llm.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            messages=[{"role": "user", "content": user_prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in analysis: {str(e)}"
#-----------------------------------------------------------------------------------------


def estimate_tokens(text, model_name="Qwen/Qwen2.5-32B-Instruct-AWQ"):
        if not isinstance(text, str):
            text = str(text)  # Ensure text is a string
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokens = tokenizer.encode(text)
        return len(tokens)

#----------------------------------------------------------------------------------------------
def truncate_history(chat_history, max_tokens=30000, preserve_head=1, preserve_tail=2):
    if not chat_history or estimate_tokens(chat_history) <= max_tokens:
        return chat_history

    head = chat_history[:preserve_head]
    tail = chat_history[-preserve_tail:]
    middle = chat_history[preserve_head:-preserve_tail] if preserve_tail > 0 else chat_history[preserve_head:]

    # Calculate token counts for efficient removal
    head_tokens = estimate_tokens(head)
    tail_tokens = estimate_tokens(tail)
    base_tokens = head_tokens + tail_tokens

    print(f"Initial token count: {base_tokens} (head: {head_tokens}, tail: {tail_tokens})")
    
    # Process middle section from oldest to newest
    new_middle = []
    current_tokens = base_tokens
    
    for msg in reversed(middle):
        msg_tokens = estimate_tokens([msg])
        if current_tokens + msg_tokens <= max_tokens:
            new_middle.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break

    return head + new_middle + tail
#------------------------------------------------------------------------------------
def summarize_history(chat_history, client_llm, preserve_recent=2, model="Qwen/Qwen2.5-32B-Instruct-AWQ"):
    """Summarizes older conversation while preserving recent messages"""\

    print("Summarizing chat history to reduce token count...\n")
    print(f"Current chat history length: {len(chat_history)} messages\n")

    print("content of chat history : \n")
    display_chat_history(chat_history)

    if len(chat_history) <= preserve_recent + 1:
        return chat_history

    old_messages = chat_history[:-preserve_recent]
    recent = chat_history[-preserve_recent:]

    # Convert to text for summarization
    history_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in old_messages if msg["content"].strip())

    
    if not history_text.strip():
        return recent

    try:
        # More focused summarization prompt
        summary_result = client_llm.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """Summarize this resume screening conversation into 3–5 concise bullet points per candidate.
                            - Ensure every candidate mentioned is covered individually.
                            - Retain all relevant qualifications, skills, certifications, and experiences.
                            - Do not omit any technical terms, role titles, project names, or numerical data (e.g., years of experience, scores).
                            - Use clear, professional language to reflect resume screening context.
                            - Do not generalize or merge data between candidates — maintain distinct bullet points per candidate."""

                },
                {"role": "user", "content": history_text}
            ],
            max_tokens=800,
            temperature=0.1
        )
        summary = summary_result.choices[0].message.content.strip()
        return [("system", f"Summary of earlier conversation:\n{summary}")] + recent

    except Exception as e:
        return [("system", "Could not summarize history")] + recent[-2:]


#-----------------------------------------------------------------------------------------
# initializing the chat_history 
chat_history = []
#--------------------------------------------------------------------
def maintaining_chat_history(chat_history, prompt, role="user"):
     chat_entry = {
        "role": role,
        "content": prompt
     }
     chat_history.append(chat_entry)
     return chat_history

#-----------------------------------------------------------------------------------------
#function to display the chat history in a readable format
def display_chat_history(chat_history: List[Dict[str, str]]):
  
    if not chat_history:
        return "No chat history available."

    display_lines = []
    for entry in chat_history:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        display_lines.append(f"[{role}]: {content}")

    print("\n\n".join(display_lines))


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

#-----------------------------------------------------------------------------------------

import re
import string

def clean_user_input(text: str, lowercase: bool = True, remove_special_chars: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-printable characters
    text = ''.join(filter(lambda x: x in string.printable, text))
    if remove_special_chars:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Optional: Convert to lowercase
    if lowercase:
        text = text.lower()

    return text
#----------------------------------------------------------------------------------------------------------------
# Unified JD Analysis Pipeline
def jd_analysis_pipeline(chat_history, user_prompt, selected_indexes=None, all_candidates=None, top_k=5):
    global cached_resume_data


    try:
        # STEP 1: Append the new user input
        if isinstance(user_prompt, str) and user_prompt.strip():
            user_prompt = clean_user_input(user_prompt)
            print(f"User prompt cleaned: {user_prompt}\n")

            maintaining_chat_history(chat_history, user_prompt, role="user")

            print(f"User prompt added to chat history: {user_prompt}\n")
            print(f"Current chat history length: {len(chat_history)} messages\n")


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
                "job_description": user_prompt,
                "parsed_job_description": parsed
            }

        # ------------------ STAGE 2: Candidate Analysis ------------------
        selected = filter_selected_candidates(all_candidates, selected_indexes) # returns only the selected candidates based on the indexes provided by the user
        if not selected:
            error_msg = "No valid candidates selected for analysis."
            maintaining_chat_history(chat_history, error_msg, role="assistant")
            return {
                "stage": "analysis",
                "response": error_msg,
                "chat_history": chat_history
            }

        result = analysis(user_prompt, selected, top_k=len(selected))

        maintaining_chat_history(chat_history, result, role="assistant")

        print(f"Analysis completed for {len(selected)} candidates.\n")
        print(f"Chat history length after analysis: {len(chat_history)} messages\n")

        print("Current chat history content:")
        display_chat_history(chat_history)

        return {
            "stage": "analysis",
            "response": result,
            "chat_history": chat_history
        }

    except Exception as e:
        error_msg = f"Error in unified JD pipeline: {str(e)}"
        maintaining_chat_history(chat_history, error_msg, role="assistant")
        return {
            "stage": "error",
            "response": error_msg,
            "chat_history": chat_history
        }

#---------------------------------------------------------------------
import ast
def format_candidate_data(data_str: str) -> str:
    """
    Converts a string representation of Python dicts/lists (with single quotes, escaped backslashes)
    into a clean JSON-like formatted string for use in LLM prompts/logs.
    """
    try:
        # Convert string with single quotes to proper Python object
        data = ast.literal_eval(data_str)

        # Convert the Python object to a JSON-like pretty string
        pretty_string = json.dumps(data, indent=2, ensure_ascii=False)
        return pretty_string
    except Exception as e:
        return f"Error parsing candidate data: {e}"
# ----------------------------------------------------------------------------------------------------

def normal_chatbot(chat_history, user_prompt,jd,selected_candidates=None,): 
    ## entering normal chatbot mode 

    print(f"Entering normal chatbot mode")
    print("\n\nCurrent chat history content:")
    display_chat_history(chat_history)

    client_llm = OpenAI(base_url=base_url, api_key="-")

    try:

        if not chat_history or chat_history[0]["role"] != "system":
            print("No system prompt found, creating a new one...")
            if  selected_candidates is not None: 
                selected_candidates = format_candidate_data(selected_candidates)
            
            system_prompt = f"""
                                You are a structured Resume Screening Assistant. Your knowledge is limited to:
                                {selected_candidates or 'NO RESUME DATA PROVIDED'}
                                Your task is to assist with resume analysis and candidate selection based on the provided job description.
                                ## Job Description:
                                {jd}
                                ### Allowed:
                                - Candidate recommendations with justifications
                                - Questions about skills/education/experience
                                - Professional analysis of resume data

                                ### Forbidden:
                                - Non-resume related topics
                                - Hallucination beyond provided data
                                - Jokes/opinions/general knowledge

                                Respond to irrelevant queries: "I specialize in resume analysis only."
                                """.strip()

            
            chat_history = [{"role": "system", "content": system_prompt}] + chat_history


        maintaining_chat_history(chat_history, user_prompt, role="user")

        print(f"User prompt added to chat history: {user_prompt}\n")
        print(f"Current chat history length: {len(chat_history)} messages\n")

        print("Current chat history content:")
        display_chat_history(chat_history)


        full_text = "\n".join(f"{entry['role']}: {entry['content']}" for entry in chat_history)

        # if estimate_tokens(full_text)> 2500:
        # #    print("Chat history exceeds 2500 tokens, summarizing and truncating...\n")
        #    # chat_history = summarize_history(chat_history, client_llm) 
            #chat_history = truncate_history(chat_history) # only triggers if the tokens is over 3000 (default in the truncate_history function)

        messages = chat_history
        result = client_llm.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            messages=messages
        )
        
        response = result.choices[0].message.content
        print(f"Chatbot response: {response}\n")
        maintaining_chat_history(chat_history, response, role="assistant")

        
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
        maintaining_chat_history(chat_history, response, role="assistant")
        return response, chat_history
        
    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        maintaining_chat_history(chat_history, error_msg, role="assistant")
        return error_msg, chat_history
    
#---------------------------------------------------------------------
