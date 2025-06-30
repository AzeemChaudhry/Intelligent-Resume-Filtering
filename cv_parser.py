from unstructured.partition.pdf import partition_pdf

from LLM import LLM_call
import os 

def pdf_to_text(pdf_path: str) -> str:
    elements = partition_pdf(pdf_path, strategy="fast")
    return "\n".join([el.text.strip() for el in elements if el.text.strip()])






def parsing_helper(markdown_text,filepath):
  prompt = f"""
  You are a precise and strict **Information Extraction Assistant**.

  Your task is to extract structured data from **unstructured CV text**, strictly following the provided JSON schema.
  add this filepath as well {filepath}

  json schema as follow : 
        name: str
        skills: list = []
        education: list = []
        work_experience: list = []
        projects: list = []
        filepath : str 
    


  ---

  ### Rules:
  - Only extract information that is **explicitly stated** in the CV text.
  - If a field is **missing**, use:'
    - `null` for missing strings
    - `[]` for missing lists
  - Do **not** hallucinate, infer, summarize, or rewrite content.
  - Preserve original text exactly as it appears.
  - Return a **valid JSON object only** — no markdown, no extra explanation.
  ### CV Text: 

  {markdown_text}

  ### Output(matches the schema):
"""
  
  return LLM_call(prompt)


def cv_parser_pipeline(path):
    candidates = []
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory!")
    print("Converting CV to text")
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isdir(file_path):
            continue
        text =  pdf_to_text(file_path)
        
        ## passing it to the LLM 
        structured_output = parsing_helper(text,file_path)

        candidates.append(structured_output)
    return candidates