*📄 CV-Parsing-Pipeline*

***🧠 Overview***
CV-Parsing-Pipeline is an end-to-end automation pipeline that:

Converts CVs (PDFs) into plain text.

Extracts structured candidate information (skills, education, work experience, projects) using a language model.

Generates vector embeddings for these extracted fields.

Stores the vectorized data in a vector database (e.g. Qdrant) for fast, scalable search and retrieval.

This pipeline simplifies CV screening, enhances querying by semantic similarity, and is easy to integrate into existing recruitment tools.

✨ Features
✅ PDF to text conversion — extract raw text content with minimal dependencies.

✅ LLM-powered parsing — structured extraction with a custom schema.

✅ Field-specific vector embeddings — represent skills, experience, etc. numerically.

✅ Efficient vector database operations — upsert and search vectors using Qdrant.

✅ Error handling & logging — gracefully skip failed CVs.

✅ Extensible and modular — easy to add new extractors or vector stores.

![image](https://github.com/user-attachments/assets/e6f383ab-8690-4dfd-91a7-2568e0e7fe7e)



🧠 Parsing Details
The pipeline uses a language model (parsing_helper.py) to parse CVs into a strict JSON schema:
{
  "skills": ["Python", "NLP", "Machine Learning"],
  "work_experience": "Explicit work history as a string",
  "education": "Degree, major, years as a string",
  "projects": ["Project1 title or description"]
}
Missing fields return null or empty lists as appropriate.

📊 Vector Storage
vector_embed.py encodes each field into a fixed-length vector.

vector_store.py handles Qdrant upserts and searches.

Weights for different fields can be tuned in get_user_requirements() or similar utility.


