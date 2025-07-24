import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil
from datetime import datetime

# Import backend functions
from main import (
    cv_parser_pipeline, 
    initialize_collection, 
    create_vec_db,
    update_cached_resumes,
    normal_chatbot,
    jd_analysis_pipeline,
    complete_jd_analysis,
    filter_selected_candidates
)

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .candidate-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 0.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Sidebar navigation
def render_sidebar():
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>🎯 AI Resume Screener</h3>
        <p>Intelligent candidate analysis powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    pages = ['upload', 'job_description', 'candidates', 'analysis', 'chat']
    current_idx = pages.index(st.session_state.current_page)
    
    st.sidebar.markdown("### 📋 Progress")
    for i, page in enumerate(pages):
        icon = "✅" if i < current_idx else "🔄" if i == current_idx else "⏳"
        status = "Completed" if i < current_idx else "Current" if i == current_idx else "Pending"
        st.sidebar.markdown(f"{icon} **{page.replace('_', ' ').title()}** - {status}")
    
    st.sidebar.markdown("---")
    
    # Navigation buttons
    st.sidebar.markdown("### 🧭 Navigation")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📁 Upload", use_container_width=True):
            st.session_state.current_page = 'upload'
            st.rerun()
        
        if st.button("👥 Candidates", use_container_width=True):
            if st.session_state.candidates:
                st.session_state.current_page = 'candidates'
                st.rerun()
    
    with col2:
        if st.button("📝 Job Desc", use_container_width=True):
            if st.session_state.processing_complete:
                st.session_state.current_page = 'job_description'
                st.rerun()
        
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.current_page = 'chat'
            st.rerun()
    
    if st.sidebar.button("📊 Analysis", use_container_width=True):
        if st.session_state.analysis_result:
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    # Statistics
    if st.session_state.candidates:
        st.sidebar.markdown("### 📊 Statistics")
        st.sidebar.metric("Total Candidates", len(st.session_state.candidates))
        if st.session_state.selected_candidates:
            st.sidebar.metric("Selected", len(st.session_state.selected_candidates))
        
        # Show top skills
        all_skills = []
        for candidate in st.session_state.candidates:
            all_skills.extend(candidate.get('skills', []))
        
        if all_skills:
            skill_counts = {}
            for skill in all_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            st.sidebar.markdown("**Top Skills:**")
            for skill, count in top_skills:
                st.sidebar.markdown(f"• {skill} ({count})")

# Upload page
def render_upload_page():
    st.markdown("""
    <div class="main-header">
        <h1>📁 Upload Resume Files</h1>
        <p>Upload PDF resume files for AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDF resume files for batch processing"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        # Display uploaded files
        st.markdown("### 📋 Uploaded Files")
        for i, file in enumerate(uploaded_files, 1):
            st.markdown(f"""
            <div class="candidate-card">
                <strong>{i}. {file.name}</strong><br>
                Size: {file.size / 1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Resumes", use_container_width=True, type="primary"):
                process_resumes(uploaded_files)

def process_resumes(uploaded_files):
    """Process uploaded resume files"""
    with st.spinner("Processing resumes... This may take a few minutes."):
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Save uploaded files
            file_paths = []
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, 'wb') as f:
                    f.write(file.getbuffer())
                file_paths.append(file_path)
            
            # Initialize vector database
            progress_bar = st.progress(0)
            st.info("Initializing vector database...")
            initialize_collection()
            progress_bar.progress(20)
            
            # Parse resumes
            st.info("Parsing resumes with AI...")
            candidates = cv_parser_pipeline(temp_dir)
            progress_bar.progress(60)
            
            # Create vector database
            st.info("Creating vector embeddings...")
            create_vec_db(candidates)
            progress_bar.progress(80)
            
            # Update cached data
            update_cached_resumes()
            progress_bar.progress(100)
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            # Store results
            st.session_state.candidates = candidates
            st.session_state.processing_complete = True
            
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Processing Complete!</strong><br>
                Successfully processed {len(candidates)} candidates.<br>
                Vector database created with {len(candidates)} entries.
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-navigate to job description
            if st.button("Continue to Job Description →", type="primary"):
                st.session_state.current_page = 'job_description'
                st.rerun()
                
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ Processing Failed</strong><br>
                Error: {str(e)}
            </div>
            """, unsafe_allow_html=True)

# Job Description page
def render_job_description_page():
    st.markdown("""
    <div class="main-header">
        <h1>📝 Job Description Analysis</h1>
        <p>Enter job requirements to find matching candidates</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.processing_complete:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ Upload Required</strong><br>
            Please upload and process resume files first.
        </div>
        """, unsafe_allow_html=True)
        return
    
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
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of candidates to find", 5, 20, 10)
    
    with col2:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific about technical skills
        - Include minimum experience requirements
        - Mention educational preferences
        - Describe role responsibilities clearly
        """)
    
    # Analyze button
    if st.button("🔍 Find Matching Candidates", type="primary", use_container_width=True):
        if job_description.strip():
            analyze_job_description(job_description, top_k)
        else:
            st.error("Please enter a job description")

def analyze_job_description(job_description, top_k):
    """Analyze job description and find candidates"""
    with st.spinner("Analyzing job description and finding candidates..."):
        try:
            result = jd_analysis_pipeline(
                chat_history=st.session_state.chat_history,
                user_prompt=job_description,
                top_k=top_k
            )

            if result["stage"] == "selection":
                st.session_state.candidates = result["candidates"]
                st.session_state.job_description = job_description
                st.session_state.chat_history = result["chat_history"]  # unified

                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ Analysis Complete!</strong><br>
                    Found {len(result["candidates"])} matching candidates.
                </div>
                """, unsafe_allow_html=True)

                st.session_state.current_page = 'candidates'
                st.rerun()
            else:
                st.error(f"Analysis failed: {result.get('response', 'Unknown error')}")

        except Exception as e:
            st.error(f"Error analyzing job description: {str(e)}")


def render_candidates_page():
    st.markdown("""
    <div class="main-header">
        <h1>👥 Candidate Selection</h1>
        <p>Review and select candidates for detailed analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.candidates:
        st.warning("ℹ️ No candidates found. Please upload resumes and complete job description analysis.")
        return

    # Display candidate stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Candidates", len(st.session_state.candidates))
    with col2:
        avg_exp = sum(c.get('work_experience_years', 0) or 0 for c in st.session_state.candidates) / len(st.session_state.candidates)
        st.metric("Avg Experience", f"{avg_exp:.1f} years")
    with col3:
        st.metric("Selected", len(st.session_state.selected_candidates))

    st.markdown("---")
    st.markdown("### 🎯 Selection Controls")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Select All"):
            st.session_state.selected_candidates = list(range(1, len(st.session_state.candidates) + 1))
    with col2:
        if st.button("Clear All"):
            st.session_state.selected_candidates = []
    with col3:
        if st.button("Top 5"):
            st.session_state.selected_candidates = list(range(1, min(6, len(st.session_state.candidates) + 1)))
    with col4:
        if st.button("Top 10"):
            st.session_state.selected_candidates = list(range(1, min(11, len(st.session_state.candidates) + 1)))
    with col5:
        final_top_k = st.selectbox("Final Top K", [3, 5, 7, 10], index=1)

    st.markdown("### 📋 Candidate Table")

    # Build table with checkboxes
    table_data = []
    for i, cand in enumerate(st.session_state.candidates, 1):
        skills = cand.get("skills", [])
        if isinstance(skills, list):
            skills = ", ".join(str(s) for s in skills[:3])
        elif isinstance(skills, str):
            pass
        else:
            skills = "N/A"

        edu = cand.get("education", [])
        if isinstance(edu, list):
            edu = ", ".join(str(e) for e in edu[:2])
        elif isinstance(edu, str):
            pass
        else:
            edu = "N/A"

        row = {
            "Select": i in st.session_state.selected_candidates,
            "Rank": i,
            "Name": str(cand.get("name", "Unknown")),
            "Experience": f'{cand.get("work_experience_years", 0)} years',
            "Score": f'{cand.get("score", 0.0):.3f}',
            "Top Skills": skills,
            "Education": edu
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    edited_df = st.data_editor(
        df,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Include this candidate")
        },
        disabled=["Rank", "Name", "Experience", "Score", "Top Skills", "Education"],
        hide_index=True,
        use_container_width=True
    )

    # Update selected candidates from checkbox
    st.session_state.selected_candidates = [
        row["Rank"] for _, row in edited_df.iterrows() if row["Select"]
    ]

    # Display analysis button
    if st.session_state.selected_candidates:
        st.markdown("### ✅ Ready for Analysis")
        if st.button("🔍 Analyze Selected Candidates", type="primary", use_container_width=True):
            analyze_selected_candidates(final_top_k)
    else:
        st.info("Select at least one candidate to proceed with analysis.")


def analyze_selected_candidates(final_top_k):
    """Analyze selected candidates"""
    with st.spinner("Analyzing selected candidates..."):
        try:
            selected_candidates = filter_selected_candidates(
                st.session_state.candidates, 
                st.session_state.selected_candidates
            )

            if final_top_k < len(selected_candidates):
                selected_candidates = selected_candidates[:final_top_k]

            analysis_result, updated_history = complete_jd_analysis(
                st.session_state.job_description,
                selected_candidates,
                st.session_state.chat_history
            )

            st.session_state.analysis_result = analysis_result
            st.session_state.chat_history = updated_history  # same history

            st.markdown("""
            <div class="success-box">
                <strong>✅ Analysis Complete!</strong><br>
                Detailed candidate analysis has been generated.
            </div>
            """, unsafe_allow_html=True)

            st.session_state.current_page = 'analysis'
            st.rerun()

        except Exception as e:
            st.error(f"Error analyzing candidates: {str(e)}")


# Analysis page
def render_analysis_page():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Analysis Results</h1>
        <p>Detailed AI-powered candidate analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analysis_result:
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ No Analysis Available</strong><br>
            Please select and analyze candidates first.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display analysis result
    st.markdown("### 📋 Detailed Analysis")
    st.markdown(st.session_state.analysis_result)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Continue to Chat", type="primary", use_container_width=True):
            st.session_state.current_page = 'chat'
            st.rerun()
    
    with col2:
        if st.button("📥 Download Analysis", use_container_width=True):
            # Create downloadable content
            content = f"""# Candidate Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Job Description
{st.session_state.job_description}

## Analysis Results
{st.session_state.analysis_result}
"""
            st.download_button(
                label="Download as Markdown",
                data=content,
                file_name=f"candidate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    with col3:
        if st.button("🔄 New Analysis", use_container_width=True):
            # Reset for new analysis
            st.session_state.current_page = 'job_description'
            st.session_state.selected_candidates = []
            st.session_state.analysis_result = ""
            st.rerun()

# Chat page
def render_chat_page():
    st.markdown("""
    <div class="main-header">
        <h1>💬 AI Assistant</h1>
        <p>Ask questions about candidates and get insights</p>
    </div>
    """, unsafe_allow_html=True)

    current_history = st.session_state.chat_history

    st.markdown("### 💭 Conversation")
    with st.container():
        if not current_history:
            st.markdown("""
            <div class="info-box">
                <strong>👋 Welcome to AI Assistant!</strong><br>
                Ask me anything about the candidates or job requirements.
            </div>
            """, unsafe_allow_html=True)
        else:
            for role, message in current_history:
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

    # Suggested questions
    if not current_history:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "Which candidate has the most Python experience?",
            "Compare the top 3 candidates",
            "Who has machine learning skills?",
            "What are the most common skills among candidates?",
            "Which candidate would be best for a senior role?"
        ]

        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    process_chat_message(suggestion)

    # Input box
    user_input = st.text_area(
        "Your question:",
        height=100,
        placeholder="Ask about candidate skills, experience, comparisons, or recommendations..."
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
            st.session_state.chat_history = []
            st.rerun()


def process_chat_message(user_input):
    """Send message to unified chatbot"""
    with st.spinner("Thinking..."):
        try:
            st.session_state.chat_history.append(("user", user_input))
            response = normal_chatbot(st.session_state.chat_history, user_input)
            st.session_state.chat_history.append(("assistant", response))
            st.rerun()
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")


# Main app
def main():
    initialize_session_state()
    render_sidebar()
    
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
