import gradio as gr
import os
import shutil
import tempfile
from pathlib import Path
from main import cv_parser_pipeline, chatbot, create_vec_db, Resume_data, update_cached_resumes,initialize_collection

SAVE_DIR = "Uploaded_Files"
chat_history = []

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

def process_and_show_chat(filepaths):
    """Process files and transition to chat interface"""
    # Filter out None values and invalid paths
    valid_filepaths = [fp for fp in (filepaths or []) if fp is not None and os.path.isfile(fp)]
    
    if not valid_filepaths:
        return (
            "No valid files selected. Please upload PDF, DOC, or DOCX files.",
            gr.update(visible=False),   # Hide landing
            gr.update(visible=True),    # Show upload (stay on upload page)
            gr.update(visible=False),   # Hide chat
            [],
            gr.update(value=None)       # Clear file input
        )
    
    status = save_files(valid_filepaths)
    
    if "Successfully processed" in status:
        return (
            status,
            gr.update(visible=False),   # Hide landing
            gr.update(visible=False),   # Hide upload
            gr.update(visible=True),    # Show chat
            [],
            gr.update(value=None)       # Clear file input
        )
    else:
        return (
            status,
            gr.update(visible=False),   # Hide landing
            gr.update(visible=True),    # Show upload (stay on upload page)
            gr.update(visible=False),   # Hide chat
            [],
            gr.update(value=None)       # Clear file input
        )

def handle_chat(user_message, history):
    """Handle chat interactions with error handling"""
    if not user_message or user_message.strip() == "":
        return history, ""
    
    try:
        history, response = chatbot(history, user_message)
        return history, ""
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if history is None:
            history = []
        history.append([user_message, error_msg])
        return history, ""

def trigger_transition():
    """Transition from landing to upload page"""
    return (
        gr.update(visible=False),   # Hide landing
        gr.update(visible=True),    # Show upload
        gr.update(visible=False),   # Hide chat
        [],
        ""
    )

def go_back():
    """Go back to landing page"""
    return (
        gr.update(visible=True),    # Show landing
        gr.update(visible=False),   # Hide upload
        gr.update(visible=False),   # Hide chat
        [],
        ""
    )

def restart():
    """Restart application"""
    return (
        gr.update(visible=True),    # Show landing
        gr.update(visible=False),   # Hide upload
        gr.update(visible=False),   # Hide chat
        [],
        ""
    )

# Premium minimalistic CSS (keeping your existing styles)
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
    .chat-container {
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
                <h2 class="section-title">Upload Resumes</h2>
                <p class="section-subtitle">
                    Select multiple resume files to begin the analysis process
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
        
        with gr.Row():
            back_btn = gr.Button("← Back", elem_classes="secondary-btn")
            process_btn = gr.Button("Process Files", elem_classes="primary-btn")
        
        status_output = gr.Textbox(
            label="Status",
            interactive=False,
            visible=True,
            value=""
        )
    
    # Chat Page
    with gr.Column(visible=False, elem_classes="page-transition") as chat_page:
        gr.HTML("""
            <div class="chat-container">
                <div class="chat-header">
                    <h2 class="section-title">AI Assistant</h2>
                    <p class="section-subtitle">
                        Ask questions about candidates or paste job requirements
                    </p>
                </div>
            </div>
        """)
        
        chatbot_interface = gr.Chatbot(
            height=400,
            show_label=False,
            container=True,
            bubble_full_width=False
        )
        
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask about candidates...",
                show_label=False,
                scale=4
            )
            send_btn = gr.Button("Send", elem_classes="primary-btn", scale=1)
        
        with gr.Row():
            new_upload_btn = gr.Button("New Upload", elem_classes="secondary-btn")
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
    
    # Event Handlers with improved error handling
    start_button.click(
        fn=trigger_transition,
        inputs=[],
        outputs=[landing_page, upload_page, chat_page, chatbot_interface, status_output]
    )
    
    process_btn.click(
        fn=process_and_show_chat,
        inputs=[file_input],
        outputs=[status_output, landing_page, upload_page, chat_page, chatbot_interface, file_input]
    )
    
    back_btn.click(
        fn=go_back,
        inputs=[],
        outputs=[landing_page, upload_page, chat_page, chatbot_interface, status_output]
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
    
    new_upload_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), [], ""),
        inputs=[],
        outputs=[landing_page, upload_page, chat_page, chatbot_interface, status_output]
    )
    
    restart_btn.click(
        fn=restart,
        inputs=[],
        outputs=[landing_page, upload_page, chat_page, chatbot_interface, status_output]
    )

if __name__ == "__main__":
    app.launch()
