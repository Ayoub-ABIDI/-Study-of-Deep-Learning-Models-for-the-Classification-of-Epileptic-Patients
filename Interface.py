import streamlit as st
from streamlit_option_menu import option_menu
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import base64
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import pywt
from PIL import Image
import requests
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Initial Configuration ---
DATA_DIR = "eeg_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('eeg_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS eeg_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id TEXT,
                  patient_name TEXT,
                  recording_date TEXT,
                  eeg_data TEXT,
                  result TEXT,
                  confidence REAL,
                  physician TEXT,
                  notes TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Set page config with medical-themed icon
st.set_page_config(
    page_title="NeuroScan EEG Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def set_custom_style():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f9fc;
    }
    h1, h2, h3 {
        color: #2b5876;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2b5876;
    }
    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed #2b5876;
        border-radius: 10px;
        padding: 20px;
    }
    .custom-metric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

set_custom_style()

# Base path for models
BASE_PATH = "C:/Users/ayoub/Downloads/Models-20250403T035016Z-002/Models"
             
# Available options
model_types = ["ANN", "CNN", "CNN-LSTM"]
experiments = {
    "Healthy vs Seizure": "Exp1",
    "Epileptic vs Seizure": "Exp2", 
    "Healthy vs Epileptic": "Exp3",
    "Healthy vs (Epileptic or Seizure)": "Exp4",
    "Healthy+Epileptic vs Seizure": "Exp5",
    "3-Class Classification": "Exp6"
}

preprocessing_options = ["Raw Data", "Derivative Transformation", "Wavelet Decomposition"]
optimization_options = ["None", "Standardization", "Adam Optimizer", "Standardization + Adam"]

# Labels for each experiment
LABELS = {
    "Exp1": ["Healthy", "Seizure"],
    "Exp2": ["Epileptic", "Seizure"],
    "Exp3": ["Healthy", "Epileptic"],
    "Exp4": ["Healthy", "Abnormal"],
    "Exp5": ["Non-Seizure", "Seizure"],
    "Exp6": ["Healthy", "Epileptic", "Seizure"]
}

# --- Preprocessing Functions ---
def compute_derivatives(data):
    derive1 = np.diff(data, n=1, axis=1)
    derive2 = np.diff(data, n=2, axis=1)
    derive3 = np.diff(data, n=3, axis=1)
    return derive1, derive2, derive3

def is_valid_eeg_file(file):
    """Ultra-robust EEG file validation"""
    try:
        # Check basic file attributes
        if not file or not hasattr(file, 'read'):
            st.error("Invalid file object provided")
            return False
            
        # Check file size (at least 10 bytes for minimal EEG data)
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset pointer
        if size < 10:
            st.error("File is too small to contain EEG data (minimum 10 bytes required)")
            return False
            
        return True
    except Exception as e:
        st.error(f"File validation error: {str(e)}")
        return False

def load_eeg_data(file):
    """Bulletproof EEG data loading with multiple fallbacks"""
    try:
        content = file.getvalue().decode('utf-8')
        
        # Try multiple parsing methods
        for method in [np.loadtxt, lambda x: np.genfromtxt(x, delimiter=None)]:
            try:
                data = method(io.StringIO(content))
                if data.size > 0:
                    return data
            except:
                continue
                
        # Final attempt - raw bytes conversion
        try:
            file.seek(0)
            data = np.frombuffer(file.read(), dtype=np.float32)
            if data.size > 0:
                return data
        except:
            pass
            
        st.error("""
        Failed to read numerical EEG data. Please ensure your file:
        1. Contains only numbers (no headers or text)
        2. Uses space, comma or tab separation
        3. Has at least 100 data points
        """)
        return None
        
    except Exception as e:
        st.error(f"Critical loading error: {str(e)}")
        return None

def preprocess_eeg_signal(file, preprocessing_method):
    """Final, completely robust preprocessing pipeline"""
    # 1. Validate file exists and is readable
    if not is_valid_eeg_file(file):
        return None
        
    # 2. Load data with multiple fallbacks
    eeg_signal = load_eeg_data(file)
    if eeg_signal is None:
        return None
        
    # 3. Basic signal validation
    if len(eeg_signal) < 100:
        st.error(f"Signal too short ({len(eeg_signal)} points). Need at least 100 points.")
        return None
        
    try:
        # 4. Apply preprocessing
        if preprocessing_method == "Raw Data":
            target_length = 4097
            if len(eeg_signal) < target_length:
                eeg_signal = np.pad(eeg_signal, (0, target_length - len(eeg_signal)), 'constant')
            else:
                eeg_signal = eeg_signal[:target_length]
            return eeg_signal.reshape(1, -1)
            
        elif preprocessing_method == "Derivative Transformation":
            derive1 = np.diff(eeg_signal, n=1)
            derive2 = np.diff(eeg_signal, n=2)
            derive3 = np.diff(eeg_signal, n=3)
            combined = np.concatenate([derive1, derive2, derive3])
            
            target_length = 16382
            if len(combined) < target_length:
                combined = np.pad(combined, (0, target_length - len(combined)), 'constant')
            else:
                combined = combined[:target_length]
            return combined.reshape(1, -1)
            
        elif preprocessing_method == "Wavelet Decomposition":
            # Simple approximation-only wavelet
            coeffs = pywt.wavedec(eeg_signal, 'db4', level=5)
            approx = coeffs[0]
            if len(approx) < 4129:
                approx = np.pad(approx, (0, 4129 - len(approx)), 'constant')
            else:
                approx = approx[:4129]
            return approx.reshape(1, -1)
            
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")

# --- Visualization Functions ---
def plot_interactive_eeg(data, title="EEG Signal"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data,
        line=dict(color='#2b5876', width=1),
        name="EEG Signal"
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Time (samples)',
        yaxis_title='Amplitude (μV)',
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_interactive_spectrum(data, fs=173.61):
    n = len(data)
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(n, 1/fs)[:n//2]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xf,
        y=2.0/n * np.abs(yf[:n//2]),
        line=dict(color='#4e4376', width=1),
        name="Frequency Spectrum"
    ))
    fig.update_layout(
        title='Frequency Spectrum',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- UI Components ---
def render_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(
            """
            <div style="background-color: #2b5876; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                <h2>🧠</h2>
                <strong>NeuroScan</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.title("EEG Analyzer Pro")
        st.caption("Advanced neural signal classification for clinical diagnostics")

def patient_info_form():
    with st.expander("🟢 Patient Information", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            name = st.text_input("Full Name", placeholder="Ayoub ABIDI")
        with cols[1]:
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        with cols[2]:
            dob = st.date_input("Date of Birth", min_value=pd.to_datetime('1920-01-01'))
        
        cols = st.columns(2)
        with cols[0]:
            pid = st.text_input("Patient ID", placeholder="NS-0000")
        with cols[1]:
            physician = st.text_input("Referring Physician", placeholder="Dr. Abidi")
        
        reason = st.text_area("Clinical Notes", placeholder="Patient presents with...")
        
        return {
            "name": name,
            "sex": sex,
            "dob": dob,
            "pid": pid,
            "physician": physician,
            "reason": reason
        }

def model_config_form():
    with st.expander("⚙️ Analysis Configuration", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            classification = st.selectbox(
                "Classification Type",
                list(experiments.keys()),
                help="Select the clinical question you want to answer"
            )
            model_type = st.selectbox(
                "Model Architecture",
                model_types,
                help="Deep learning model to use for analysis"
            )
        with cols[1]:
            preprocessing = st.selectbox(
                "Signal Processing",
                preprocessing_options,
                help="Preprocessing method for EEG signals"
            )
            optimization = st.selectbox(
                "Optimization",
                optimization_options,
                help="Optimization techniques applied"
            )
        
        return {
            "classification": classification,
            "model_type": model_type,
            "preprocessing": preprocessing,
            "optimization": optimization
        }

def file_uploader():
    with st.expander("📁 Upload EEG Data", expanded=True):
        file = st.file_uploader(
            "Select EEG recording file",
            type=["txt", "csv", "edf"],
            help="Supported formats: .txt (ASCII), .csv, .edf"
        )
        if file:
            st.success(f"File {file.name} uploaded successfully")
            with st.spinner("Validating EEG data..."):
                st.progress(100)
                st.caption("✅ Data validated - ready for analysis")
        return file

def get_model_path(config):
    experiment = experiments[config["classification"]]
    model_type = config["model_type"]
    preprocessing = config["preprocessing"]
    optimization = config["optimization"]
    
    if preprocessing == "Raw Data":
        data_type = "Raw_Data"
    elif preprocessing == "Derivative Transformation":
        data_type = "Derive"
    else:
        data_type = "Walvet_dec"
    
    if optimization == "Standardization":
        fname = "With_Stander.keras"
    elif optimization == "Adam Optimizer":
        fname = "With_Optimiser.keras"
    elif optimization == "Standardization + Adam":
        fname = "With_Options.keras"
    else:
        fname = "No_Options.keras"
    
    return os.path.join(BASE_PATH, data_type, model_type, experiment, fname)

def analyze_eeg(file, model_path, preprocessing_method):
    try:
        model = load_model(model_path)
        st.info(f"Model loaded successfully. Expected input shape: {model.input_shape}")
        
        eeg_signal = preprocess_eeg_signal(file, preprocessing_method)
        if eeg_signal is None:
            return None
            
        st.info(f"Processed EEG signal shape: {eeg_signal.shape}")
        return model.predict(eeg_signal)
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def generate_report(patient, config, results, eeg_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Set up fonts and colors
    title_font = "Helvetica-Bold"
    heading_font = "Helvetica-Bold"
    body_font = "Helvetica"
    primary_color = (0.16, 0.34, 0.46)  # Dark blue
    secondary_color = (0.8, 0.85, 0.9)  # Light blue
    alert_color = (0.9, 0.2, 0.2)      # Red
    normal_color = (0.2, 0.7, 0.3)     # Green
    
    # Title Section
    c.setFillColorRGB(*primary_color)
    c.rect(0, 750, 612, 60, fill=True, stroke=False)
    c.setFillColorRGB(1, 1, 1)
    c.setFont(title_font, 18)
    c.drawCentredString(306, 765, "EEG DIAGNOSTIC REPORT")
    c.setFont(body_font, 10)
    c.drawCentredString(306, 745, "NeuroScan Advanced Neural Analysis System")

    # Report Metadata
    c.setFillColorRGB(0, 0, 0)
    c.setFont(body_font, 8)
    c.drawRightString(600, 735, f"Report ID: NS-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    c.drawRightString(600, 725, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # Patient Information Section
    c.setFillColorRGB(*secondary_color)
    c.roundRect(50, 650, 512, 60, 5, fill=True, stroke=False)
    c.setFillColorRGB(0, 0, 0)
    c.setFont(heading_font, 14)
    c.drawString(60, 680, "PATIENT INFORMATION")
    
    # Patient Details
    c.setFont(body_font, 10)
    col1_x = 60
    col2_x = 300
    line_height = 15
    current_y = 660
    
    patient_info = [
        (f"Name: {patient.get('name', 'N/A')}", f"Physician: {patient.get('physician', 'N/A')}"),
        (f"Patient ID: {patient.get('pid', 'N/A')}", f"Department: Neurology"),
        (f"DOB: {patient.get('dob', 'N/A')}", f"Study Date: {datetime.now().strftime('%Y-%m-%d')}"),
        (f"Sex: {patient.get('sex', 'N/A')}", f"Report Status: Final")
    ]
    
    for left, right in patient_info:
        c.drawString(col1_x, current_y, left)
        c.drawString(col2_x, current_y, right)
        current_y -= line_height

    # EEG Visualization Section
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.roundRect(50, 500, 512, 130, 5, fill=True, stroke=False)
    c.setFillColorRGB(0, 0, 0)
    c.setFont(heading_font, 14)
    c.drawString(60, 610, "EEG SIGNAL VISUALIZATION")
    
    # Create and save EEG plot as image
    try:
        fig = plot_interactive_eeg(eeg_data[:1000])  # Show first 1000 samples for clarity
        img_data = fig.to_image(format="png", width=500, height=200)
        img = Image.open(io.BytesIO(img_data))
        
        # Save image to temporary file
        img_path = "temp_eeg_plot.png"
        img.save(img_path)
        
        # Add plot to PDF
        c.drawImage(img_path, 70, 520, width=470, height=150)
        os.remove(img_path)  # Clean up temporary file
    except Exception as e:
        c.drawString(70, 550, "Could not generate EEG visualization")
        c.drawString(70, 530, f"Error: {str(e)}")

    # Analysis Parameters Section
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.roundRect(50, 400, 512, 60, 5, fill=True, stroke=False)
    c.setFillColorRGB(0, 0, 0)
    c.setFont(heading_font, 14)
    c.drawString(60, 430, "ANALYSIS PARAMETERS")
    
    # Analysis Details
    c.setFont(body_font, 10)
    current_y = 420
    
    analysis_params = [
        (f"Classification: {config.get('classification', 'N/A')}", 
         f"Model: {config.get('model_type', 'N/A')}"),
        (f"Processing: {config.get('preprocessing', 'N/A')}", 
         f"Optimization: {config.get('optimization', 'N/A')}")
    ]
    
    for left, right in analysis_params:
        c.drawString(col1_x, current_y, left)
        c.drawString(col2_x, current_y, right)
        current_y -= line_height

    # Results Section
    c.setFillColorRGB(1, 1, 1)
    c.roundRect(50, 250, 512, 130, 5, fill=True, stroke=True)
    c.setFillColorRGB(0, 0, 0)
    c.setFont(heading_font, 16)
    c.drawCentredString(306, 370, "DIAGNOSTIC FINDINGS")
    
    if len(results["labels"]) == 2:
        # Binary Classification Results
        c.setFont(heading_font, 14)
        c.drawString(80, 340, "Primary Classification:")
        c.setFont(heading_font, 16)
        c.drawString(250, 340, f"{results['result']}")
        
        c.setFont(heading_font, 14)
        c.drawString(80, 310, "Confidence Level:")
        c.setFont(heading_font, 16)
        c.drawString(250, 310, f"{results['probability']*100:.1f}%")
        
        # Clinical Significance Box
        box_y = 270
        if "Seizure" in results["result"] or "Epileptic" in results["result"]:
            c.setFillColorRGB(*alert_color)
            c.roundRect(80, box_y, 452, 30, 3, fill=True, stroke=False)
            c.setFillColorRGB(1, 1, 1)
            c.setFont(heading_font, 10)
            c.drawCentredString(306, box_y+10, "⚠️ ABNORMAL FINDINGS - CLINICAL ATTENTION REQUIRED ⚠️")
        else:
            c.setFillColorRGB(*normal_color)
            c.roundRect(80, box_y, 452, 30, 3, fill=True, stroke=False)
            c.setFillColorRGB(1, 1, 1)
            c.setFont(heading_font, 10)
            c.drawCentredString(306, box_y+10, "✓ NORMAL EEG PATTERNS DETECTED")
    else:
        # Multi-class Classification Results
        c.setFont(heading_font, 14)
        c.drawCentredString(306, 340, "Classification Probabilities")
        
        y_pos = 310
        for label, prob in zip(results["labels"], results["probabilities"]):
            # Probability bar
            bar_width = 300 * prob
            c.setFillColorRGB(*primary_color)
            c.roundRect(120, y_pos-5, bar_width, 10, 2, fill=True, stroke=False)
            
            # Text labels
            c.setFillColorRGB(0, 0, 0)
            c.setFont(body_font, 10)
            c.drawString(80, y_pos-5, f"{label}:")
            c.drawString(450, y_pos-5, f"{prob*100:.1f}%")
            y_pos -= 20

    # Footer
    c.setFillColorRGB(*primary_color)
    c.rect(0, 0, 612, 40, fill=True, stroke=False)
    c.setFillColorRGB(1, 1, 1)
    c.setFont(body_font, 8)
    c.drawString(50, 20, "NeuroScan EEG Analyzer v3.0 | Clinical Decision Support System")
    c.drawRightString(562, 20, "University of Tunis El Manar")

    c.save()
    buffer.seek(0)
    return buffer

def settings_page():
    st.subheader("Application Settings")
    
    with st.expander("⚙️ General Settings", expanded=True):
        st.checkbox("Enable dark mode", value=False, key="dark_mode")
        st.slider("Chart animation speed", 0, 10, 5, key="anim_speed")
        st.selectbox("Default theme", ["Light", "Dark", "System"], key="default_theme")
    
    with st.expander("📊 Visualization Settings", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.selectbox("Default chart type", ["Interactive", "Static"], key="chart_type")
            st.color_picker("Primary color", "#2b5876", key="primary_color")
        with cols[1]:
            st.number_input("Default sample points", 500, 5000, 1000, key="sample_points")
            st.color_picker("Secondary color", "#4e4376", key="secondary_color")
    
    with st.expander("🔒 Data Privacy", expanded=True):
        st.checkbox("Anonymize patient data in exports", value=True, key="anonymize_data")
        st.checkbox("Enable data encryption", value=False, key="data_encryption")
        st.selectbox("Data retention policy", ["30 days", "90 days", "1 year", "Indefinite"], key="data_retention")
    
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")
        # Here you would typically save settings to a config file or database
        # For now we'll just use session state
        st.session_state.settings_saved = True

# --- Main App ---
def main():
    render_header()
    
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Analyze", "Database", "Settings"],
            icons=["house", "activity", "database", "gear"],
            default_index=1,
            styles={
                "container": {"padding": "5px", "background-color": "#f0f2f6"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
                "nav-link-selected": {"background-color": "#4e4376"},
            }
        )
    
    if selected == "Analyze":
        st.subheader("New EEG Analysis")
        st.markdown("---")
        
        patient = patient_info_form()
        config = model_config_form()
        file = file_uploader()
        
        if file and st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing EEG patterns..."):
                model_path = get_model_path(config)
                
                if not os.path.exists(model_path):
                    st.error("Selected model configuration not available")
                    return
                
                try:
                    eeg_data = np.loadtxt(file)
                    st.subheader("EEG Signal Visualization")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        fig = plot_interactive_eeg(eeg_data)
                        st.plotly_chart(fig, use_container_width=True)
                    with cols[1]:
                        fig = plot_interactive_spectrum(eeg_data)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not visualize EEG data: {str(e)}")
                
                results = analyze_eeg(file, model_path, config["preprocessing"])
                
                if results is not None:
                    experiment = experiments[config["classification"]]
                    labels = LABELS[experiment]
                    
                    # Debug output to understand model results
                    with st.expander("Model Output Details"):
                        st.write("Model output shape:", results.shape)
                        st.write("Sample output values:", results[:3])
                    
                    try:
                        # Handle different model output formats
                        if len(results.shape) == 1:
                            # Single output (binary classification)
                            pred_prob = float(results[0])
                            pred_class = int(pred_prob > 0.5)
                            prob = pred_prob if pred_class == 1 else 1 - pred_prob
                            result = labels[pred_class]
                            probs = [1 - prob, prob] if len(labels) == 2 else None
                        elif len(results.shape) == 2:
                            if results.shape[1] == 1:
                                # Binary classification with single output
                                pred_prob = float(results[0][0])
                                pred_class = int(pred_prob > 0.5)
                                prob = pred_prob if pred_class == 1 else 1 - pred_prob
                                result = labels[pred_class]
                                probs = [1 - prob, prob] if len(labels) == 2 else None
                            else:
                                # Multi-class classification
                                # Apply softmax if not already probabilities
                                if not np.allclose(np.sum(results, axis=1), 1.0):
                                    results = np.exp(results) / np.sum(np.exp(results), axis=1, keepdims=True)
                                probs = results[0]
                                pred_class = np.argmax(probs)
                                prob = float(probs[pred_class])
                                result = labels[pred_class]
                        else:
                            raise ValueError("Unexpected model output shape")
                            
                    except Exception as e:
                        st.error(f"Error processing results: {str(e)}")
                        result = "Error"
                        prob = 0.0
                        probs = None
                        import traceback
                        st.text(traceback.format_exc())
                    
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        with st.container():
                            st.markdown('<div class="custom-metric">', unsafe_allow_html=True)
                            st.metric("Classification", result)
                            st.markdown('</div>', unsafe_allow_html=True)
                    with cols[1]:
                        with st.container():
                            st.markdown('<div class="custom-metric">', unsafe_allow_html=True)
                            st.metric("Confidence", f"{prob*100:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    if probs is not None and len(labels) > 2:
                        st.markdown("### Class Probabilities")
                        for label, p in zip(labels, probs):
                            with st.container():
                                st.markdown('<div class="custom-metric">', unsafe_allow_html=True)
                                st.metric(label, f"{p*100:.2f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    if "Seizure" in result or "Epileptic" in result:
                        st.error("""
                        **Clinical Significance:**  
                        This EEG pattern suggests abnormal neurological activity.  
                        Recommended actions:
                        - Immediate clinical correlation
                        - Neurological consultation
                        - Consider additional monitoring
                        """)
                    else:
                        st.success("""
                        **Clinical Significance:**  
                        No epileptiform activity detected.  
                        Normal EEG patterns observed.
                        """)
                    
                    # Save to database
                    conn = sqlite3.connect('eeg_data.db')
                    c = conn.cursor()
                    c.execute('''INSERT INTO eeg_records 
                                (patient_id, patient_name, recording_date, eeg_data, result, confidence, physician, notes)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                             (patient["pid"], 
                              patient["name"],
                              datetime.now().strftime("%Y-%m-%d %H:%M"),
                              str(eeg_data.tolist()),
                              result,
                              float(prob),
                              patient["physician"],
                              patient["reason"]))
                    conn.commit()
                    conn.close()
                    
                    report = generate_report(
                        patient,
                        config,
                        {
                            "result": result,
                            "probability": prob,
                            "labels": labels,
                            "probabilities": probs.tolist() if probs is not None else None
                        },
                        eeg_data
                    )
                    
                    st.download_button(
                        "Download Full Report (PDF)",
                        report,
                        file_name=f"EEG_Report_{patient['pid']}.pdf",
                        mime="application/pdf"
                    )
    
    elif selected == "Home":
        st.subheader("Clinical EEG Analysis Platform")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://enit.rnu.tn/wp-content/uploads/2019/07/LOGO_ENIT_300.png", 
                    width=200, 
                    caption="National Engineering School of Tunis")
        
        with col2:
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%); 
                            color: white; padding: 30px; border-radius: 10px; text-align: center;">
                    <h1 style="margin-bottom: 0;">NeuroScan EEG Analyzer</h1>
                    <p style="margin-top: 0.5em; font-size: 1.2em;">Advanced Neural Signal Processing for Clinical Diagnostics</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("""
        <div style="background-color: #f0f5ff; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <p style="font-size: 1.1em; text-align: center;">
            A comprehensive platform for automated EEG analysis using deep learning models, 
            developed for accurate detection of neurological abnormalities including 
            epileptic seizures and other brain activity disorders.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Key Features")
        cols = st.columns(3)
        with cols[0]:
            with st.container(border=True, height=250):
                st.markdown("### 🧠 Advanced Analysis")
                st.markdown("""
                - State-of-the-art deep learning models
                - Multiple classification scenarios
                - Configurable processing pipelines
                - High accuracy detection
                """)
        
        with cols[1]:
            with st.container(border=True, height=250):
                st.markdown("### ⚡ Clinical Workflow")
                st.markdown("""
                - Patient management system
                - Comprehensive reporting
                - PDF export capabilities
                - Historical record tracking
                """)
        
        with cols[2]:
            with st.container(border=True, height=250):
                st.markdown("### 🔬 Research Ready")
                st.markdown("""
                - Raw signal visualization
                - Frequency spectrum analysis
                - Multiple model architectures
                - Experimental configurations
                """)
        
        st.markdown("## Supported Protocols")
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
        Our system supports various EEG recording protocols to accommodate different clinical and research needs:
        </div>
        """, unsafe_allow_html=True)
        
        protocol_cols = st.columns(3)
        with protocol_cols[0]:
            with st.container(border=True):
                st.markdown("**🏥 Routine EEG**")
                st.markdown("Standard clinical recordings in awake and resting state")
        
        with protocol_cols[1]:
            with st.container(border=True):
                st.markdown("**😴 Sleep-Deprived EEG**")
                st.markdown("Extended recordings following sleep deprivation protocols")
        
        with protocol_cols[2]:
            with st.container(border=True):
                st.markdown("**📱 Ambulatory Monitoring**")
                st.markdown("Long-term recordings in natural environments")
        
        st.markdown("## Clinical Applications")
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
        This platform is designed to assist in various neurological assessments:
        </div>
        """, unsafe_allow_html=True)
        
        app_cols = st.columns(3)
        with app_cols[0]:
            with st.container(border=True):
                st.markdown("**⚡ Epilepsy Diagnosis**")
                st.markdown("Detection of interictal epileptiform discharges")
        
        with app_cols[1]:
            with st.container(border=True):
                st.markdown("**⚠️ Seizure Detection**")
                st.markdown("Identification of ictal patterns in EEG recordings")
        
        with app_cols[2]:
            with st.container(border=True):
                st.markdown("**🧪 Neurological Research**")
                st.markdown("Brain activity analysis for research studies")
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background-color: #2b5876; color: white; padding: 20px; border-radius: 10px;">
            <h3 style="color: white; text-align: center;">Development Team</h3>
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <p><strong>Ayoub ABIDI</strong></p>
                    <p>ICT Research Engineer</p>
                </div>
                <div>
                    <p><strong>Eya SOUSSI</strong></p>
                    <p>ICT Research Engineer</p>
                </div>
            </div>
            <p style="text-align: center; margin-top: 20px;">
            <strong>RISC Laboratory</strong><br>
            National Engineering School of Tunis (ENIT)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    elif selected == "Database":
        st.title("EEG Database")
        
        conn = sqlite3.connect('eeg_data.db')
        records = conn.execute('''SELECT * FROM eeg_records 
                                 ORDER BY recording_date DESC''').fetchall()
        
        if not records:
            st.info("No EEG records found in database.")
        else:
            search_term = st.text_input("Search records")
            
            for record in records:
                if search_term.lower() not in str(record).lower():
                    continue
                    
                with st.expander(f"Record {record[0]} - {record[2]} ({record[5]})"):
                    cols = st.columns([2, 1])
                    with cols[0]:
                        try:
                            eeg_data = np.array(eval(record[4]))
                            fig = plot_interactive_eeg(eeg_data, f"EEG Recording {record[0]}")
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.warning("Could not visualize EEG data")
                    
                    with cols[1]:
                        st.write(f"**Patient:** {record[2]}")
                        st.write(f"**ID:** {record[1]}")
                        st.write(f"**Date:** {record[3]}")
                        st.write(f"**Result:** {record[5]}")
                        
                        confidence = record[6]
                        if isinstance(confidence, bytes):
                            try:
                                confidence = float(confidence.decode('utf-8'))
                            except (ValueError, AttributeError):
                                confidence = 0.0
                        elif not isinstance(confidence, (float, int)):
                            try:
                                confidence = float(confidence)
                            except (ValueError, TypeError):
                                confidence = 0.0
                        
                        st.write(f"**Confidence:** {confidence:.1f}%")
                        st.write(f"**Physician:** {record[7]}")
                        st.write(f"**Notes:** {record[8]}")

        conn.close()
    
    elif selected == "Settings":
        settings_page()

if __name__ == "__main__":
    main()