import json 
from pydantic import BaseModel
from openai import OpenAI


class ResumeInfo(BaseModel):
    name: str
    skills: list = []
    education: list = []
    work_experience: list = []
    projects: list = []
    filepath : str 


json_schema = ResumeInfo.model_json_schema()

def LLM_call(prompt):
    client = OpenAI(
    base_url="http://172.16.2.214:8000/v1", 
    api_key="-" 
    )
    response = client.chat.completions.create(
    model="Qwen/Qwen2.5-32B-Instruct-AWQ",
    messages=[
        {"role": "user", "content": prompt}
    ],
   
   extra_body={"guided_json": json_schema}
    )

    #print(response.choices[0].message.content)

    return json.loads(response.choices[0].message.content)
