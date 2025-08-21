# Intelligent‑Resume‑Filtering

A **Retrieval-Augmented System for Intelligent Resume Filtering**

---

## 🚀 Project Overview

This project implements an AI-powered resume screening platform. It enables:

- **OCR-based PDF parsing** using `pdf2image` and `pytesseract`
- **Information extraction** via LLMs (e.g., Qwen) guided by strict JSON schemas
- **Embedding extraction** through `sentence_transformers`
- **Vector similarity search** powered by **Qdrant**
- **Candidate ranking, analysis, and interactive chatbot interface** using LLM capabilities
- A **Streamlit-based frontend (`app3.py`)** to orchestrate the workflow seamlessly

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/AzeemChaudhry/Intelligent-Resume-Filtering.git
cd Intelligent-Resume-Filtering
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install & Configure Tesseract OCR

- Download and install Tesseract from its official release page.
- On Windows, a typical path is:
  ```
  C:\Program Files\Tesseract-OCR\tesseract.exe
  ```
- This path is already configured in `main.py`:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

### Start Qdrant (Vector Database)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Running the Application

To launch the Streamlit interface:

```bash
streamlit run app3.py
```

This will guide you through:

1. Uploading PDF resumes
2. Parsing and storing resumes (OCR → schema parsing → embeddings → Qdrant)
3. Submitting job descriptions
4. Retrieving and ranking candidates
5. Performing detailed candidate analysis and interactive chat

---

## Example Workflow

1. **Upload resumes** via the app interface.
2. System extracts text with OCR (`pdf_to_text`), parses it into structured JSON (`parsing_helper`), and logs results.
3. Embeddings are generated and stored in Qdrant (`initialize_collection` & `insert_candidate`).
4. Enter a **Job Description** → the system parses it and searches the vector store.
5. Top candidates are presented; choose selections for deeper analysis.
6. Get a recruiter-style breakdown, candidate insights, and chat-based assistance.

---

## 🛠️ Tech Stack

- **Python** (Pydantic, Transformers, Sentence-Transformers, pdf2image, pytesseract)
- **Streamlit** (Frontend)
- **Qdrant** (Vector Database)
- **OpenAI/Qwen LLMs** (Structured data extraction & chatbot)
