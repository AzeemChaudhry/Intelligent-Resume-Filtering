from LLM import LLM_call,OpenAI
from Vec_db import join_and_embed,embedding_model,required_fields,client
from collections import defaultdict
from cv_parser import cv_parser_pipeline
from Vec_db import create_vec_db



def job_description_parser(job_description) : 
    
    job_prompt = f"""
    You are an **Information Extraction Assistant**.

    Your task:
    - Parse the provided **job description**.
    - Extract **explicit information only** — ***do not infer, invent, or assume***.
    - Output a **valid JSON object** that matches the schema shown below.

    ### JSON schema:
    {{
    "skills": [ "list of required skills as short strings" ],
    "work_experience": "explicit description of required work experience, as a string",
    "education": "explicit education or qualification requirements, as a string",
    "projects": [ "list of explicitly mentioned types of projects or domains" ]
    }}

    ### Rules:
    - If a field is not present in the job description, use:
    - an empty list `[]` for list fields,
    - or `null` for string fields.
    - Do **not** add any extra text outside the JSON.
    - Do **not** add markdown or explanations.
    - Preserve the original wording of the job description when filling fields.

    ### Job Description:
    {job_description}
    ### Output(matches the schema):
 """

    return LLM_call(job_prompt)



def Searching_Qdrant(parsed__job_description,top_k) : 

    job_vectors = {
        field: join_and_embed(parsed__job_description.get(field, []), embedding_model)
        for field in required_fields
    }

    fields = ["skills", "education", "work_experience", "projects"]
    user_weights_raw = {}

    print("Please enter weight for each field. Total should sum to 1 (e.g. 0.4, 0.2, etc.)")

    for field in fields:
        while True:
            try:
                weight = float(input(f"Enter weight for '{field}': "))
                if weight < 0:
                    raise ValueError
                user_weights_raw[field] = weight
                break
            except ValueError:
                print("Invalid input. Please enter a non-negative number.")

    total_weight = sum(user_weights_raw.values())

    if abs(total_weight - 1.0) > 1e-6:
        print(f"\n Total weight entered is {total_weight:.3f}, normalizing to 1.")
        user_weights = {k: v / total_weight for k, v in user_weights_raw.items()}
    else:
        user_weights = user_weights_raw

    print("\n Normalized Weights:")
    for field, weight in user_weights.items():
        print(f"  {field}: {weight:.3f}")


    results = {}

    for field in required_fields:
        hits = client.search(
            collection_name="cv_data",
            query_vector=(field, job_vectors[field]), 
            limit=top_k,
            with_payload=True,
            with_vectors=False  
        )
        results[field] = hits


    score_board = defaultdict(float)

    for field in results:
        weight = user_weights.get(field, 0)
        for hit in results[field]:
            score_board[hit.id] += hit.score * weight
    return score_board





def sorting_candidates(score_board,top_k): 

    ranked = sorted(score_board.items(), key=lambda x: x[1], reverse=True)

    top_candidates = []  
    shown = 0

    for candidate_id, total_score in ranked:
        point = next(
            (pt for pt in client.scroll(
                collection_name="cv_data",
                with_payload=True,
                with_vectors=False,
                limit=100
            )[0] if pt.id == candidate_id),
            None
        )
        if point:
            candidate_info = {
                "name": point.payload.get("name"),
                "filepath": point.payload.get("filepath"),
                "score": round(total_score, 4),
                "id": candidate_id
            }
            top_candidates.append(candidate_info)

            # # Display
            # print(f"Name: {candidate_info['name']}")
            # print(f"Filepath: {candidate_info['filepath']}")
            # print(f"Score: {candidate_info['score']}\n")

            shown += 1
        if shown >= top_k:
            break

    return top_candidates


def analysis(job_description,top_candidates,candidates,top_k):
   prompt_3 = f""" You are an expert technical recruiter and AI career advisor. Use the information provided below to perform an in-depth candidate evaluation using semantic embeddings retrieved from a vector database (Qdrant).

---

### 🧾 Job Description

{job_description}
---

### 📌 Top {top_k} Candidates from Vector Similarity Search : {top_candidates}

These candidates were retrieved from the Qdrant vector database based on semantic similarity to the job description. Each candidate includes their resume filepath,a vector similarity score and id.
---

Here is all the candidates 
{candidates}

### 🎯 Task

Analyze the candidates above with respect to the job description and perform the following:

1. **Compare** each candidate's qualifications with the job description in terms of:
   - Skills
   - Work experience
   - Relevant projects
   - Educational background

2. **Rank** the candidates from most to least suitable based on the job description and vector match scores.

3. **Justify** your top 1–2 recommendations with detailed reasoning, focusing on fit for the role. Also the decision should not be eccentric to the match scores

4. **Highlight Gaps**:
   - Are there any key missing skills or misalignments?
   - Are there strengths that go beyond the role?

5. **Provide Insights**:
   - Strengths and weaknesses of each candidate
   - A comparison table showing each candidate's match to the job requirements
   - Visual ideas: skill coverage bar chart or score comparison chart
   - Suggest which candidate fits which kind of sub-role (e.g., research-focused, full-stack AI, deployment)

6. Write in a **professional tone** suitable for HR and technical hiring managers.

---

Your goal is to aid the hiring manager in making a well-informed and confident decision based on both semantic similarity and practical job fit. Dont disclose any scores to the user in the output 

 """
   
   client = OpenAI(
   base_url="http://172.16.2.214:8000/v1", 
   api_key="-" 
   )
   response = client.chat.completions.create(
   model="Qwen/Qwen2.5-32B-Instruct-AWQ",
   messages=[
      {"role": "user", "content": prompt_3}
      ],
      )

   print(response.choices[0].message.content)


   ### writing the main function for everything

def main(): 
    
    candidates = cv_parser_pipeline("Resumes")

    ## creating vec db 
    create_vec_db(candidates)

    job_description = input("***Please Enter the Job Description***")
    parsed__job_description = job_description_parser(job_description)

    top_k = int(input("Enter number of top candidates to display: "))
    score_board = Searching_Qdrant(parsed__job_description,top_k)

    top_candidates = sorting_candidates(score_board,top_k)

    return analysis(job_description,top_candidates,candidates,top_k)

if __name__ == "__main__": 
    main()