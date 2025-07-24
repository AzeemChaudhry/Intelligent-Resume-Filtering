import gradio as gr
import os
import shutil
import tempfile
from pathlib import Path
from main import cv_parser_pipeline, create_vec_db, Resume_data, update_cached_resumes, initialize_collection, normal_chatbot, complete_jd_analysis, get_top_candidates, analyze_selected_candidates

SAVE_DIR = "Uploaded_Files"
# Initialize global chat history as a list to maintain state
global global_chat_history
global_chat_history = []

# Global variables for state management
current_candidates = []
current_job_description = ""
jd_chat_state = {}

def save_files(filepaths):
    """Process uploaded files with proper error handling"""
    if not filepaths:
        return "No files selected"
    
    try:
        # Clean up existing directory
        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        processed_files = []
        
        # Process each file individually
        for filepath in filepaths:
            if filepath is None:
                continue
                
            # Ensure we have a valid file path, not a directory
            if os.path.isfile(filepath):
                filename = os.path.basename(filepath)
                destination = os.path.join(SAVE_DIR, filename)
                
                # Copy file with error handling
                try:
                    shutil.copy2(filepath, destination)
                    processed_files.append(filename)
                    print(f"Processed: {filename}")
                except Exception as file_error:
                    print(f"Error processing {filename}: {str(file_error)}")
                    continue
            else:
                print(f"Skipping invalid path: {filepath}")
        
        if not processed_files:
            return "No valid files were processed"
        
        # Process resumes
        candidates = cv_parser_pipeline(SAVE_DIR)
        initialize_collection()
        create_vec_db(candidates)
        print("VCDB created\n")
        update_cached_resumes()
        
        return f"Successfully processed {len(processed_files)} resumes: {', '.join(processed_files)}"
        
    except Exception as e:
        print(f"Error in save_files: {str(e)}")
        return f"Error processing files: {str(e)}"

def process_files_and_job_description(filepaths, job_description):
    """Process files and job description, then show candidates for selection"""
    global current_candidates, current_job_description, jd_chat_state, global_chat_history
    
    # Filter out None values and invalid paths
    valid_filepaths = [fp for fp in (filepaths or []) if fp is not None and os.path.isfile(fp)]
    
    if not valid_filepaths:
        return (
            "No valid files selected. Please upload PDF, DOC, or DOCX files.",
            "",
            gr.update(choices=[], value=[]),  # Fixed: Update CheckboxGroup properly
            gr.update(visible=False),   # Hide landing
            gr.update(visible=True),    # Show upload (stay on upload page)
            gr.update(visible=False),   # Hide selection
            gr.update(visible=False),   # Hide results
            gr.update(visible=False),   # Hide chat
            gr.update(value=None)       # Clear file input
        )
    
    if not job_description or job_description.strip() == "":
        return (
            "Please enter a job description.",
            "",
            gr.update(choices=[], value=[]),  # Fixed: Update CheckboxGroup properly
            gr.update(visible=False),   # Hide landing
            gr.update(visible=True),    # Show upload (stay on upload page)
            gr.update(visible=False),   # Hide selection
            gr.update(visible=False),   # Hide results
            gr.update(visible=False),   # Hide chat
            gr.update(value=None)       # Clear file input
        )
    
    # Process files first
    status = save_files(valid_filepaths)
    
    if "Successfully processed" not in status:
        return (
            status,
            "",
            gr.update(choices=[], value=[]),  # Fixed: Update CheckboxGroup properly
            gr.update(visible=False),   # Hide landing
            gr.update(visible=True),    # Show upload (stay on upload page)
            gr.update(visible=False),   # Hide selection
            gr.update(visible=False),   # Hide results
            gr.update(visible=False),   # Hide chat
            gr.update(value=None)       # Clear file input
        )
    
    try:
        # Reset global chat history for new analysis
        global_chat_history = []
        
        # Get initial candidates (using default top_k=10 to show more options)
        candidates, display_text = get_top_candidates(job_description, top_k=10)
        current_candidates = candidates
        current_job_description = job_description
        
        # Create checkbox choices for candidate selection
        candidate_choices = []
        for idx, candidate in enumerate(candidates, 1):
            name = candidate.get("name", "N/A")
            exp_years = candidate.get("work_experience_years", "N/A")
            skills = ", ".join(candidate.get("skills", [])[:3])  # Show first 3 skills
            if len(candidate.get("skills", [])) > 3:
                skills += "..."
            choice_label = f"#{idx} - {name} ({exp_years} yrs) - {skills}"
            candidate_choices.append(choice_label)
        
        return (
            status,
            display_text,
            gr.update(choices=candidate_choices, value=[]),  # Fixed: Update both choices and value
            gr.update(visible=False),   # Hide landing
            gr.update(visible=False),   # Hide upload
            gr.update(visible=True),    # Show selection
            gr.update(visible=False),   # Hide results
            gr.update(visible=False),   # Hide chat
            gr.update(value=None)       # Clear file input
        )
        
    except Exception as e:
        error_msg = f"Error processing job description: {str(e)}"
        return (
            error_msg,
            "",
            gr.update(choices=[], value=[]),  # Fixed: Update CheckboxGroup properly
            gr.update(visible=False),   # Hide landing
            gr.update(visible=True),    # Show upload (stay on upload page)
            gr.update(visible=False),   # Hide selection
            gr.update(visible=False),   # Hide results
            gr.update(visible=False),   # Hide chat
            gr.update(value=None)       # Clear file input
        )

def analyze_candidates_with_selection(selected_candidates, top_k_final):
    """Analyze selected candidates with final top_k"""
    global current_candidates, current_job_description, global_chat_history
    
    if not selected_candidates:
        return (
            "Please select at least one candidate for analysis.",
            gr.update(visible=False),   # Hide landing
            gr.update(visible=False),   # Hide upload
            gr.update(visible=True),    # Show selection (stay on selection page)
            gr.update(visible=False),   # Hide results
            gr.update(visible=False)    # Hide chat
        )
    
    try:
        # Extract candidate indices from selected labels
        selected_indices = []
        selected_candidate_objects = []
        
        for selected in selected_candidates:
            # Extract the number from "#1 - Name..." format
            try:
                idx = int(selected.split(" - ")[0].replace("#", ""))
                selected_indices.append(idx)
                if 1 <= idx <= len(current_candidates):
                    selected_candidate_objects.append(current_candidates[idx - 1])
            except (ValueError, IndexError) as e:
                print(f"Error parsing candidate selection: {selected}, Error: {e}")
                continue
        
        if not selected_candidate_objects:
            return (
                "Error: Could not parse selected candidates. Please try again.",
                gr.update(visible=False),   # Hide landing
                gr.update(visible=False),   # Hide upload
                gr.update(visible=True),    # Show selection (stay on selection page)
                gr.update(visible=False),   # Hide results
                gr.update(visible=False)    # Hide chat
            )
        
        # Limit to top_k_final if specified
        if top_k_final and top_k_final < len(selected_candidate_objects):
            selected_candidate_objects = selected_candidate_objects[:top_k_final]
            selected_indices = selected_indices[:top_k_final]
        
        # Analyze selected candidates
        analysis_result = analyze_selected_candidates(
            current_job_description, 
            current_candidates, 
            selected_indices
        )
        
        return (
            analysis_result,
            gr.update(visible=False),   # Hide landing
            gr.update(visible=False),   # Hide upload
            gr.update(visible=False),   # Hide selection
            gr.update(visible=True),    # Show results
            gr.update(visible=False)    # Hide chat
        )
        
    except Exception as e:
        error_msg = f"Error analyzing candidates: {str(e)}"
        print(f"Full error details: {e}")
        return (
            error_msg,
            gr.update(visible=False),   # Hide landing
            gr.update(visible=False),   # Hide upload
            gr.update(visible=True),    # Show selection (stay on selection page)
            gr.update(visible=False),   # Hide results
            gr.update(visible=False)    # Hide chat
        )

def go_to_chat():
    """Navigate to chat interface"""
    return (
        gr.update(visible=False),   # Hide landing
        gr.update(visible=False),   # Hide upload
        gr.update(visible=False),   # Hide selection
        gr.update(visible=False),   # Hide results
        gr.update(visible=True),    # Show chat
        []
    )

def handle_chat(user_message, history):
    """Handle chat interactions with proper format for Gradio chatbot"""
    global global_chat_history
    
    if not user_message or user_message.strip() == "":
        return history, ""
    
    # Initialize history if None
    if history is None:
        history = []
    
    try:
        # Call the normal chatbot function
        response = normal_chatbot(global_chat_history, user_message)
        
        # Update global chat history for backend consistency
        global_chat_history.append(("user", user_message))
        global_chat_history.append(("assistant", response))
        
        # Append to Gradio chatbot history in the correct format [user_msg, bot_response]
        history.append([user_message, response])
        
        return history, ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        # Even for errors, maintain the correct format
        history.append([user_message, error_msg])
        return history, ""

def clear_chat():
    """Clear chat history"""
    global global_chat_history
    global_chat_history = []
    return []

def trigger_transition():
    """Transition from landing to upload page"""
    return (
        gr.update(visible=False),   # Hide landing
        gr.update(visible=True),    # Show upload
        gr.update(visible=False),   # Hide selection
        gr.update(visible=False),   # Hide results
        gr.update(visible=False),   # Hide chat
        [],
        ""
    )

def go_back():
    """Go back to landing page"""
    return (
        gr.update(visible=True),    # Show landing
        gr.update(visible=False),   # Hide upload
        gr.update(visible=False),   # Hide selection
        gr.update(visible=False),   # Hide results
        gr.update(visible=False),   # Hide chat
        [],
        ""
    )

def restart():
    """Restart application"""
    global global_chat_history, current_candidates, current_job_description
    global_chat_history = []
    current_candidates = []
    current_job_description = ""
    return (
        gr.update(visible=True),    # Show landing
        gr.update(visible=False),   # Hide upload
        gr.update(visible=False),   # Hide selection
        gr.update(visible=False),   # Hide results
        gr.update(visible=False),   # Hide chat
        [],
        ""
    )

def select_all_candidates(current_choices):
    """Select all available candidates"""
    return current_choices

def clear_selection():
    """Clear all candidate selections"""
    return []

# Premium minimalistic CSS (enhanced with new styles)
premium_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

body, .gradio-container {
    background: #fafafa;
    margin: 0;
    padding: 0;
}

.main-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
}

/* Landing Page Styles */
.landing-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-text {
    text-align: center;
    margin-bottom: 4rem;
    z-index: 10;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 300;
    color: #1a1a1a;
    margin: 0 0 1rem 0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.2rem;
    font-weight: 400;
    color: #666;
    margin: 0;
    max-width: 500px;
    line-height: 1.5;
}

/* Ball Animation */
.ball-container {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
}

.animated-ball {
    width: 120px;
    height: 120px;
    background: linear-gradient(135deg, #1a1a1a 0%, #333 100%);
    border-radius: 50%;
    position: relative;
    animation: float 3s ease-in-out infinite;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.animated-ball::before {
    content: '';
    position: absolute;
    top: 20%;
    left: 20%;
    width: 30%;
    height: 30%;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    filter: blur(10px);
}

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
}

.ball-drop {
    animation: ballDrop 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
}

@keyframes ballDrop {
    0% {
        transform: translateY(0) scale(1);
        border-radius: 50%;
    }
    70% {
        transform: translateY(100vh) scale(0.8);
        border-radius: 50%;
    }
    100% {
        transform: translateY(100vh) scale(20);
        border-radius: 0%;
        opacity: 0;
    }
}

/* Modern Button */
.primary-btn {
    background: #1a1a1a !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 16px 32px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    box-shadow: 0 4px 20px rgba(26, 26, 26, 0.2) !important;
    position: relative !important;
    overflow: hidden !important;
    z-index: 10;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(26, 26, 26, 0.3) !important;
    background: #333 !important;
}

.primary-btn:active {
    transform: translateY(0) !important;
}

/* Upload Page */
.upload-container {
    max-width: 600px;
    margin: 0 auto;
    padding: 4rem 2rem;
    animation: slideUp 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.section-title {
    font-size: 2.5rem;
    font-weight: 300;
    color: #1a1a1a;
    text-align: center;
    margin-bottom: 1rem;
    letter-spacing: -0.01em;
}

.section-subtitle {
    font-size: 1.1rem;
    color: #666;
    text-align: center;
    margin-bottom: 3rem;
    line-height: 1.5;
}

/* File Upload Area */
.upload-area {
    border: 2px dashed #ddd !important;
    border-radius: 12px !important;
    padding: 3rem 2rem !important;
    text-align: center !important;
    background: white !important;
    transition: all 0.3s ease !important;
    margin-bottom: 2rem !important;
}

.upload-area:hover {
    border-color: #1a1a1a !important;
    background: #fafafa !important;
}

/* Job Description Textarea */
.job-description-area {
    background: white !important;
    border: 1px solid #ddd !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 2rem !important;
    min-height: 120px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    resize: vertical !important;
}

.job-description-area:focus {
    border-color: #1a1a1a !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(26, 26, 26, 0.1) !important;
}

/* Slider Styles */
.top-k-slider {
    margin-bottom: 2rem !important;
}

.top-k-slider .gr-slider {
    background: white !important;
    border: 1px solid #ddd !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Secondary Button */
.secondary-btn {
    background: white !important;
    color: #1a1a1a !important;
    border: 1px solid #ddd !important;
    border-radius: 50px !important;
    padding: 12px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.secondary-btn:hover {
    border-color: #1a1a1a !important;
    background: #f5f5f5 !important;
}

/* Selection helpers */
.selection-btn {
    background: #f8f9fa !important;
    color: #495057 !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 25px !important;
    padding: 8px 16px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    margin: 0 4px !important;
}

.selection-btn:hover {
    background: #e9ecef !important;
    border-color: #adb5bd !important;
}

/* Results Page */
.results-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 4rem 2rem;
    animation: slideUp 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.results-box {
    background: white !important;
    border: 1px solid #ddd !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    margin-bottom: 2rem !important;
    white-space: pre-wrap !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.6 !important;
    max-height: 400px !important;
    overflow-y: auto !important;
}

/* Chat Interface */
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    animation: fadeIn 0.6s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.chat-header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #eee;
}

/* Checkbox Group Styling */
.checkbox-group {
    background: white !important;
    border: 1px solid #ddd !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
    max-height: 300px !important;
    overflow-y: auto !important;
}

/* Status Messages */
.status-success {
    background: #f0f9f0 !important;
    color: #2d5a2d !important;
    border: 1px solid #c3e6c3 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin: 1rem 0 !important;
}

.status-error {
    background: #fdf2f2 !important;
    color: #c53030 !important;
    border: 1px solid #fed7d7 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin: 1rem 0 !important;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }
    
    .animated-ball {
        width: 80px;
        height: 80px;
    }
    
    .upload-container,
    .chat-container,
    .results-container {
        padding: 2rem 1rem;
    }
}

/* Hide Gradio Branding */
.gradio-container .footer {
    display: none !important;
}
"""

# Create the interface with improved error handling
with gr.Blocks(
    title="Resume Screener",
    css=premium_css,
    theme=gr.themes.Base(
        primary_hue="slate",
        secondary_hue="gray",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        border_color_primary="*neutral_200"
    )
) as app:
    
    # Landing Page
    with gr.Column(visible=True, elem_classes="page-transition") as landing_page:
        gr.HTML("""
            <div class="landing-container">
                <div class="ball-container">
                    <div class="animated-ball" id="animatedBall"></div>
                </div>
                <div class="hero-text">
                    <h1 class="hero-title">Resume Screener</h1>
                    <p class="hero-subtitle">
                        Intelligent resume analysis powered by AI. 
                        Upload, analyze, and discover the perfect candidates.
                    </p>
                </div>
            </div>
        """)
        
        start_button = gr.Button(
            "Begin",
            elem_classes="primary-btn",
            elem_id="startButton"
        )
    
    # Upload Page
    with gr.Column(visible=False, elem_classes="page-transition") as upload_page:
        gr.HTML("""
            <div class="upload-container">
                <h2 class="section-title">Upload & Analyze</h2>
                <p class="section-subtitle">
                    Upload resume files and provide job description for intelligent matching
                </p>
            </div>
        """)
        
        file_input = gr.File(
            file_count="multiple",
            type="filepath",
            label="Choose Resume Files",
            file_types=[".pdf", ".doc", ".docx"],
            elem_classes="upload-area"
        )
        
        job_description_input = gr.Textbox(
            label="Job Description",
            placeholder="Paste the job description here...",
            lines=6,
            elem_classes="job-description-area"
        )
        
        with gr.Row():
            back_btn = gr.Button("← Back", elem_classes="secondary-btn")
            process_btn = gr.Button("Analyze Candidates", elem_classes="primary-btn")
        
        status_output = gr.Textbox(
            label="Status",
            interactive=False,
            visible=True,
            value=""
        )
    
    # Candidate Selection Page
    with gr.Column(visible=False, elem_classes="page-transition") as selection_page:
        gr.HTML("""
            <div class="upload-container">
                <h2 class="section-title">Select Candidates</h2>
                <p class="section-subtitle">
                    Choose candidates you want to analyze and set final top_k
                </p>
            </div>
        """)
        
        candidates_display = gr.Textbox(
            label="Available Candidates",
            interactive=False,
            lines=15,
            elem_classes="results-box"
        )
        
        # Selection helper buttons
        with gr.Row():
            select_all_btn = gr.Button("Select All", elem_classes="selection-btn")
            clear_selection_btn = gr.Button("Clear Selection", elem_classes="selection-btn")
        
        candidate_checkboxes = gr.CheckboxGroup(
            label="Select Candidates for Analysis",
            choices=[],
            value=[],
            elem_classes="checkbox-group"
        )
        
        top_k_final_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Final Top K Candidates",
            info="Select how many of the selected candidates to analyze",
            elem_classes="top-k-slider"
        )
        
        with gr.Row():
            back_to_upload_btn = gr.Button("← Back to Upload", elem_classes="secondary-btn")
            analyze_selected_btn = gr.Button("Analyze Selected", elem_classes="primary-btn")
    
    # Results Page
    with gr.Column(visible=False, elem_classes="page-transition") as results_page:
        gr.HTML("""
            <div class="results-container">
                <h2 class="section-title">Analysis Results</h2>
                <p class="section-subtitle">
                    Top candidates matched against your job requirements
                </p>
            </div>
        """)
        
        results_output = gr.Textbox(
            label="Candidate Analysis",
            interactive=False,
            lines=15,
            elem_classes="results-box"
        )
        
        with gr.Row():
            back_to_upload_btn2 = gr.Button("← Back to Upload", elem_classes="secondary-btn")
            go_to_chat_btn = gr.Button("Continue to Chat", elem_classes="primary-btn")
    
    # Chat Page
    with gr.Column(visible=False, elem_classes="page-transition") as chat_page:
        gr.HTML("""
            <div class="chat-container">
                <div class="chat-header">
                    <h2 class="section-title">AI Assistant</h2>
                    <p class="section-subtitle">
                        Ask detailed questions about candidates or get more insights
                    </p>
                </div>
            </div>
        """)
        
        chatbot_interface = gr.Chatbot(
            height=400,
            show_label=False,
            container=True,
            bubble_full_width=False,
            value=[]  # Initialize with empty list
        )
        
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask about candidates...",
                show_label=False,
                scale=4
            )
            send_btn = gr.Button("Send", elem_classes="primary-btn", scale=1)
        
        with gr.Row():
            clear_chat_btn = gr.Button("Clear Chat", elem_classes="secondary-btn")
            new_upload_btn = gr.Button("New Analysis", elem_classes="secondary-btn")
            restart_btn = gr.Button("Restart", elem_classes="secondary-btn")
    
    # JavaScript for animations
    gr.HTML("""
        <script>
        function triggerBallDrop() {
            const ball = document.getElementById('animatedBall');
            if (ball) {
                ball.classList.add('ball-drop');
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            const startBtn = document.getElementById('startButton');
            if (startBtn) {
                startBtn.addEventListener('click', triggerBallDrop);
            }
        });
        </script>
    """)
    
    # Event Handlers
    start_button.click(
        fn=trigger_transition,
        inputs=[],
        outputs=[landing_page, upload_page, selection_page, results_page, chat_page, chatbot_interface, status_output]
    )
    
    process_btn.click(
        fn=process_files_and_job_description,
        inputs=[file_input, job_description_input],
        outputs=[status_output, candidates_display, candidate_checkboxes, landing_page, upload_page, selection_page, results_page, chat_page, file_input]
    )
    
    # Selection helper functions
    select_all_btn.click(
        fn=select_all_candidates,
        inputs=[candidate_checkboxes],
        outputs=[candidate_checkboxes]
    )
    
    clear_selection_btn.click(
        fn=clear_selection,
        inputs=[],
        outputs=[candidate_checkboxes]
    )
    
    analyze_selected_btn.click(
        fn=analyze_candidates_with_selection,
        inputs=[candidate_checkboxes, top_k_final_slider],
        outputs=[results_output, landing_page, upload_page, selection_page, results_page, chat_page]
    )
    
    back_btn.click(
        fn=go_back,
        inputs=[],
        outputs=[landing_page, upload_page, selection_page, results_page, chat_page, chatbot_interface, status_output]
    )
    
    back_to_upload_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", gr.update(choices=[], value=[])),
        inputs=[],
        outputs=[landing_page, upload_page, selection_page, results_page, chat_page, status_output, candidate_checkboxes]
    )
    
    back_to_upload_btn2.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", gr.update(choices=[], value=[])),
        inputs=[],
        outputs=[landing_page, upload_page, selection_page, results_page, chat_page, status_output, candidate_checkboxes]
    )
    
    go_to_chat_btn.click(
        fn=go_to_chat,
        inputs=[],
        outputs=[landing_page, upload_page, selection_page, results_page, chat_page, chatbot_interface]
    )
    
    send_btn.click(
        fn=handle_chat,
        inputs=[chat_input, chatbot_interface],
        outputs=[chatbot_interface, chat_input]
    )
    
    chat_input.submit(
        fn=handle_chat,
        inputs=[chat_input, chatbot_interface],
        outputs=[chatbot_interface, chat_input]
    )
    
    clear_chat_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot_interface]
    )
    
    new_upload_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", gr.update(choices=[], value=[])),
        inputs=[],
        outputs=[landing_page, upload_page, selection_page, results_page, chat_page, status_output, candidate_checkboxes]
    )
    
    restart_btn.click(
        fn=restart,
        inputs=[],
        outputs=[landing_page, upload_page, selection_page, results_page, chat_page, chatbot_interface, status_output]
    )

if __name__ == "__main__":
    app.launch()
