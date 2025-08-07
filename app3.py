import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import io
import shutil
from datetime import datetime
import time

from main import (
    cv_parser_pipeline, 
    initialize_collection, 
    create_vec_db,
    normal_chatbot,
    maintaining_chat_history,
    jd_analysis_pipeline,
    filter_selected_candidates
)


st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide"
)

def initialize_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'upload'
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'selected_candidates' not in st.session_state:
        st.session_state.selected_candidates = []
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'auto_transition' not in st.session_state:
        st.session_state.auto_transition = True
    if 'analysis_shown_in_chat' not in st.session_state:
        st.session_state.analysis_shown_in_chat = False
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0


# Upload page
def render_upload_page():
    
    uploaded_files = st.file_uploader("Choose PDF files",type=['pdf'],accept_multiple_files=True,help="Upload multiple PDF resume files for batch processing")
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Process Resumes", use_container_width=True, type="primary"):
                process_resumes(uploaded_files)

def process_resumes(uploaded_files):
    
    try:
        save_dir = r"C:\Users\Azeem\Documents\CARE\Intelligent-Resume-Filtering\Uploaded_Files"

        # Clear the folder first
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Save uploaded files
        file_paths = []
        for file in uploaded_files:
            print(f"Processing file: {file.name}")
            file_path = os.path.join(save_dir, file.name)
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        initialize_collection()
        candidates = cv_parser_pipeline(save_dir)
        create_vec_db(candidates)
        st.session_state.candidates = candidates
        st.session_state.processing_complete = True
        
        st.markdown(f"""
        <div class="status-success">
            <h3> Processing Complete!</h3>
            <p>Successfully processed {len(candidates)} candidates</p>
            <p>Automatically moving to job description...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-transition to job description
        time.sleep(3)
        st.session_state.current_page = 'job_description'
        st.rerun()
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")

# Job Description page
def render_job_description_page():

    if not st.session_state.processing_complete:
        st.markdown("""
        <div class="status-info">
            <h3>Upload Required</h3>
            <p>Please upload and process resume files first.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("← Back to Upload", type="secondary"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return
    
    st.markdown("""
    <div class="main-card">
        <h1 style="text-align: center; color: #333;">📝 Job Description Analysis</h1>
        <p style="text-align: center; color: #666;">Enter job requirements to find matching candidates</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Job description input
    job_description = st.text_area(
        "Job Description",
        height=300,
        placeholder="""Enter detailed job description here...

Include:
• Required skills and technologies
• Experience requirements
• Educational qualifications
• Job responsibilities
• Company culture and values""",
        value=st.session_state.job_description
    )
    
    # Settings in a clean layout
    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.slider("Number of candidates to find", 1, len(st.session_state.candidates), 1)
    
    # Analyze button
    col1, col2 = st.columns([1, 2])
    with col2:
        if st.button("🔍 Find Matching Candidates", type="primary", use_container_width=True):
            if job_description.strip():
                analyze_job_description(job_description, top_k)
            else:
                st.error("Please enter a job description")




##helper main function to run analysis pipeline
def run_analysis_pipeline(job_description, top_k=5, selected_indexes=None, all_candidates=None, mode="selection"):
    """Unified function to run both initial and selected candidate analysis."""
    st.markdown(f"""
    <div class="status-processing">
        <h3>🔍 Analyzing {'Selected Candidates' if mode == 'analysis' else 'Job Description'}</h3>
        <p>{'Generating detailed analysis...' if mode == 'analysis' else 'Finding the best matching candidates...'}</p>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0)
    
    try:
        progress_bar.progress(30)

        result = jd_analysis_pipeline(
            chat_history=st.session_state.chat_history,
            user_prompt=job_description,
            selected_indexes=selected_indexes,
            all_candidates=all_candidates,
            top_k=top_k
        )

        progress_bar.progress(100)

        if result["stage"] == "selection":
            st.session_state.candidates = result["candidates"]
            st.session_state.job_description = result["parsed_job_description"]
            st.session_state.chat_history = result["chat_history"]

            st.markdown(f"""
            <div class="status-success">
                <h3>✅ Analysis Complete!</h3>
                <p>Found {len(result["candidates"])} matching candidates</p>
                <p>Moving to candidate selection...</p>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(2)
            st.session_state.current_page = 'candidates'
            st.rerun()

        elif result["stage"] == "analysis":
            st.session_state.analysis_result = result["response"]
            st.session_state.chat_history = result["chat_history"]
            st.session_state.analysis_shown_in_chat = True

            st.markdown("""
            <div class="status-success">
                <h3>Analysis Complete!</h3>
                <p>Detailed candidate analysis has been generated</p>
                <p>Moving to analysis results...</p>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(2)
            st.session_state.current_page = 'analysis'
            st.rerun()

        else:
            st.error(f"Analysis failed: {result.get('response', 'Unknown error')}")

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")




def analyze_job_description(job_description, top_k):
    run_analysis_pipeline(job_description=job_description, top_k=top_k, mode="selection")



def get_candidates_df():
    data = []
    for idx, c in enumerate(st.session_state.candidates, start=1):
        data.append({
            "ID": idx,
            "Name": c.get("name", "N/A"),
            "Experience (years)": c.get("work_experience_years", 0),
            "Skills": ", ".join(c.get("skills", [])),
            "Score": c.get("score", "N/A"),
            "Selected": idx in st.session_state.selected_candidates
        })
    return pd.DataFrame(data)


def render_candidates_page():
    import pandas as pd
    import streamlit as st

    if not st.session_state.candidates:
        st.warning("No candidates found. Please complete job description analysis first.")
        return

    def flatten_and_stringify(data, max_items=3):
        if data is None:
            return "N/A"
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return ", ".join(f"{k}: {v}" for k, v in list(data.items())[:max_items])
        if isinstance(data, (list, tuple, set)):
            return ", ".join(str(item) for item in list(data)[:max_items]) or "N/A"
        return str(data)

    if 'selected_candidates' not in st.session_state:
        st.session_state.selected_candidates = []
    data = []
    for i, c in enumerate(st.session_state.candidates, 1):
        data.append({
            "ID": i,
            "Name": flatten_and_stringify(c.get("name")),
            "Experience": c.get("work_experience_years", "N/A"),
            "Skills": flatten_and_stringify(c.get("skills")),
            "Education": flatten_and_stringify(c.get("education")),
            "Score": c.get("score", "N/A"),
            "Selected": False  # Default to not selected
        })

    sort_options = {
        "🔼 Experience (Low → High)": ("Experience", True),
        "🔽 Experience (High → Low)": ("Experience", False),
        "🔼 Score (Low → High)": ("Score", True),
        "🔽 Score (High → Low)": ("Score", False),
        "🔼 Name (A → Z)": ("Name", True),
        "🔽 Name (Z → A)": ("Name", False),
        "🔼 ID (Low → High)": ("ID", True),
        "🔽 ID (High → Low)": ("ID", False),
    }

    sort_choice = st.selectbox("📊 Sort Candidates By", list(sort_options.keys()))
    sort_by, sort_asc = sort_options[sort_choice]

    df = pd.DataFrame(data)
    df = df.sort_values(by=sort_by, ascending=sort_asc)

    edited_df = st.data_editor(
        df)


    # Just update the selected candidate IDs for now
    st.session_state.selected_candidates = [
        row["ID"] for _, row in edited_df.iterrows() if row["Selected"]
    ]

    if st.session_state.selected_candidates:
        st.markdown("Ready for Analysis")
        st.success(f"Selected {len(st.session_state.selected_candidates)} candidates: {', '.join(map(str, st.session_state.selected_candidates))}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze Selected Candidates", type="primary", use_container_width=True):
                updated_candidates = []
                for i, c in enumerate(st.session_state.candidates):
                    selected = (i + 1) in st.session_state.selected_candidates
                    updated_candidates.append({**c, "selected": selected})
                st.session_state.candidates = updated_candidates

                analyze_selected_candidates(len(st.session_state.selected_candidates))
    else:
        st.info("Select at least one candidate to proceed with analysis.")

def analyze_selected_candidates(final_top_k=5):
    selected_indexes = st.session_state.selected_candidates
    if not selected_indexes:
        st.error("Please select candidates before analyzing.")
        return
    run_analysis_pipeline(
        job_description=st.session_state.job_description,
        top_k=final_top_k,
        selected_indexes=selected_indexes,
        all_candidates=st.session_state.candidates,
        mode="analysis"
    )

# def analyze_selected_candidates(final_top_k=5):
#     """Analyze selected candidates with auto-transition"""
    
#     st.markdown("""
#     <div class="status-processing">
#         <h3>🔍 Analyzing Selected Candidates</h3>
#         <p>Generating detailed analysis...</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     progress_bar = st.progress(0)
    
#     try:
#         selected_candidates = filter_selected_candidates(
#             st.session_state.candidates, 
#             st.session_state.selected_candidates
#         )
#         progress_bar.progress(30)

#         if final_top_k < len(selected_candidates):
#             selected_candidates = selected_candidates[:final_top_k]
#         progress_bar.progress(60)

#         result = jd_analysis_pipeline(
#         chat_history=st.session_state.chat_history,
#         user_prompt=st.session_state.job_description,
#         selected_indexes=selected_candidates,
#         all_candidates=st.session_state.candidates,
#         top_k= final_top_k
#             )

#         analysis_result = result["response"]
#         updated_history = result["chat_history"]

#         progress_bar.progress(100)

#         st.session_state.analysis_result = analysis_result
#         st.session_state.chat_history = updated_history
#         # Mark that analysis has been shown in chat history
#         st.session_state.analysis_shown_in_chat = True

#         st.markdown("""
#         <div class="status-success">
#             <h3>✅ Analysis Complete!</h3>
#             <p>Detailed candidate analysis has been generated</p>
#             <p>Moving to analysis results...</p>
#         </div>
#         """, unsafe_allow_html=True)

#         # Auto-transition to analysis
#         time.sleep(2)
#         st.session_state.current_page = 'analysis'
#         st.rerun()

#     except Exception as e:
#         st.error(f"Error analyzing candidates: {str(e)}")

# Analysis page
def render_analysis_page():
    

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Continue to Chat", type="primary", use_container_width=True):
            st.session_state.current_page = 'chat'
            st.rerun()
    
    with col2:
        # Download functionality
        content = f"""# Candidate Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Job Description
{st.session_state.job_description}

## Analysis Results
{st.markdown(st.session_state.analysis_result or "_No analysis result available yet._")}
"""
        st.download_button(
            label="📥 Download Analysis",
            data=content,
            file_name=f"candidate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col3:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.current_page = 'job_description'
            st.session_state.selected_candidates = []
            st.session_state.analysis_result = ""
            st.session_state.analysis_shown_in_chat = False
            st.rerun()

# Chat page
def render_chat_page():
    display_history = []

    # Only show new messages that are added after reaching the chat page
    # This ensures a clean chat interface without showing backend conversation history
    if 'chat_display_history' not in st.session_state:
        st.session_state.chat_display_history = []

    display_history = st.session_state.chat_display_history

    # Chat history display
    if not display_history:
        st.markdown("""
        <div class="status-info">
            <h3>👋 Welcome to AI Assistant!</h3>
            <p>Ask me anything about the candidates or job requirements.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggested questions
        st.markdown("### 💡 Try asking:")
        suggestions = [
            "Which candidate has the most Python experience?",
            "Compare the top 3 candidates",
            "Who has machine learning skills?",
            "What are the most common skills among candidates?",
            "Which candidate would be best for a senior role?"
        ]

        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion}", use_container_width=True):
                process_chat_message(suggestion)
    else:
        # Display chat history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message_dict in display_history:
            role = message_dict["role"]
            message = message_dict["content"]

            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
            elif role == "assistant":
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>AI Assistant:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(" Ask a Question")
    user_input = st.text_area(
        "Your question:",
        height=100,
        placeholder="Ask about candidate skills, experience, comparisons, or recommendations...",
        key=f"chat_input_{st.session_state.chat_input_key}"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Send Message", type="primary", use_container_width=True):
            if user_input.strip():
                process_chat_message(user_input)
            else:
                st.error("Please enter a message")

    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_display_history = []
            st.session_state.chat_input_key += 1
            st.rerun()
        if st.session_state.chat_history:
            chat_md = ""
            for entry in st.session_state.chat_history:
                role = entry.get("role", "unknown").capitalize()
                content = entry.get("content", "").strip()
                chat_md += f"### {role}\n\n{content}\n\n---\n\n"

            buffer = io.StringIO(chat_md)
            st.download_button(
                label="Download Chat History (Markdown)",
                data=buffer.getvalue(),
                file_name="chat_history.md",
                mime="text/markdown",
                use_container_width=True
            )



def process_chat_message(user_input):
    """Send message to chatbot without duplicating messages"""

    try:
        temp_history = st.session_state.chat_history.copy()
        chat_addition = {
            "role": "user",
            "content": user_input
        }
        temp_history.append(chat_addition)

        with st.spinner("AI is thinking..."):
            print(st.session_state.selected_candidates)
            selected_candidates = filter_selected_candidates(
                st.session_state.candidates, 
                st.session_state.selected_candidates
            )

            print(f"Selected candidates for chat: {selected_candidates}")

            response = normal_chatbot(temp_history, user_input, st.session_state.job_description, selected_candidates)

        st.session_state.chat_history = maintaining_chat_history(
            st.session_state.chat_history, 
            user_input,
            "user"
        )
        st.session_state.chat_history = maintaining_chat_history(
            st.session_state.chat_history,
            response,
            "assistant"
        )

        if 'chat_display_history' not in st.session_state:
            st.session_state.chat_display_history = []

        st.session_state.chat_display_history.append({"role": "user", "content": user_input})
        st.session_state.chat_display_history.append({"role": "assistant", "content": response})
        st.session_state.chat_input_key += 1
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")


# Main app
def main():
    initialize_session_state()
    
    # Route to appropriate page
    if st.session_state.current_page == 'upload':
        render_upload_page()
    elif st.session_state.current_page == 'job_description':
        render_job_description_page()
    elif st.session_state.current_page == 'candidates':
        render_candidates_page()
    elif st.session_state.current_page == 'analysis':
        render_analysis_page()
    elif st.session_state.current_page == 'chat':
        render_chat_page()

if __name__ == "__main__":
    main()