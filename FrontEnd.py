# import streamlit as st
# from  import (
#     cv_parser_pipeline,
#     create_vec_db,
#     job_description_parser,
#     Searching_Qdrant,
#     sorting_candidates,
#     analysis
# )
# import os
# import shutil

# def save_uploaded_files(uploaded_files, save_dir="Resumes"):
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.makedirs(save_dir, exist_ok=True)

#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(save_dir, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#     return save_dir


# def main():
#     st.set_page_config(page_title="AI Resume Matcher", layout="wide")
#     st.title("🤖 AI-Powered Resume Matching System")

#     st.markdown("Upload resumes (PDFs), enter the job description, and get the top-k matching candidates using vector search and LLM insights.")

#     # --- Upload Resumes ---
#     uploaded_files = st.file_uploader(
#         "📎 Upload Resume PDFs",
#         type=["pdf"],
#         accept_multiple_files=True
#     )

#     # --- Job Description Input ---
#     job_description = st.text_area("🧾 Enter the Job Description", height=250)

#     # --- Top-K Slider ---
#     top_k = st.slider("🎯 Number of Top Candidates to Display", min_value=1, max_value=20, value=5)

#     if st.button("🚀 Run Matching Pipeline"):
#         if not uploaded_files or not job_description.strip():
#             st.error("Please upload at least one resume and provide a job description.")
#             return

#         with st.spinner("Processing resumes and job description..."):
#             # Save files locally
#             save_dir = save_uploaded_files(uploaded_files)

#             # Parse resumes and build vector DB
#             candidates = cv_parser_pipeline(save_dir)
#             create_vec_db(candidates)

#             # Parse job description
#             parsed_job_description = job_description_parser(job_description)

#             # Vector similarity search
#             score_board = Searching_Qdrant(parsed_job_description, top_k)

#             # Sort and get top candidates
#             top_candidates = sorting_candidates(score_board, top_k)

#             # Run analysis (LLM-based, formatted)
#             report = analysis(job_description, top_candidates, candidates, top_k)

#         st.success("✅ Matching complete!")
#         st.markdown("### 📊 Candidate Analysis Report")
#         st.markdown(report, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
