import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io
import joblib as jb
from langchain_groq import ChatGroq

# Set page configuration
st.set_page_config(
    page_title="Farmer's Hub",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2E7D32;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .subtitle {
        text-align: center;
        color: #558B2F;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    .section-header {
        color: #2E7D32;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #66BB6A;
    }

    .info-box {
        background: linear-gradient(135deg, #0D47A1 0%, #1976D2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #64B5F6;
        margin: 1rem 0;
    }


    .result-box {
        background: linear-gradient(135deg, #FFF176 0%, #FDD835 100%);
        color: #212121; /* Dark gray text for better contrast */
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FBC02D;
        margin: 1rem 0;
        font-size: 1.15rem;
        font-weight: 500;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1); /* Soft shadow for depth */
    }

    .stButton>button {
        background: linear-gradient(135deg, #66BB6A 0%, #43A047 100%);
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 28px;
        border: none;
        border-radius: 25px;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
        transform: scale(1.02);
    }

    .divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #66BB6A, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio(
    "Choose a Feature:",
    ["🏠 Home", "🌱 Crop Recommendation", "🥔 Potato Disease Detection", "💬 Farmer's Chatbot"]
)

# Helper Functions
@st.cache_resource
def load_crop_model():
    """Load the crop recommendation model"""
    try:
        with open('C:\\Users\\shoai\\Downloads\\Machine Learning Projects\\Real-World-Projects-ML+DL+Logic\\Crop-Recommendation-System\\crop_rec.pickle', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

@st.cache_resource
def load_potato_model():
    """Load the potato disease detection model"""
    try:
        model = jb.load(r'C:\Users\shoai\Downloads\Machine Learning Projects\All-3-Projects-Combined\Potato_Image_classification\rf_model_mobilenetv2_potato.joblib')
        return model
    except FileNotFoundError:
        return None

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Predict the best crop based on input parameters"""
    model = load_crop_model()
    if model is None:
        return "Model not found. Please upload crop_rec.pickle"

    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        return f"Error: {str(e)}"

# Load MobileNetV2 feature extractor once
@st.cache_resource
def load_feature_extractor():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )
    return base_model

def predict_potato_disease(image: Image.Image) -> str:
    """Predict potato disease from image using pre-trained model"""
    model = load_potato_model()
    feature_extractor = load_feature_extractor()

    if model is None:
        return "Model not found. Please upload rf_model_mobilenetv2_potato.joblib"

    try:
        # Resize and preprocess image
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        # Extract features using MobileNetV2
        features = feature_extractor.predict(image_array)  # shape: (1, 1280)

        # Predict using RandomForest
        prediction = model.predict(features)[0]

        # Map numeric prediction to label
        label_map = {0: "Healthy", 1: "Early Blight", 2: "Late Blight"}
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        return f"Error: {str(e)}"

GROQ_API_KEY = "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"

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>🌾 Welcome to Farmer's Hub 🌾</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p class='subtitle'>
            Your one-stop solution for agricultural innovation powered by AI and machine learning.<br>
            Explore cutting-edge technologies designed to empower farmers and optimize crop management.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Feature Overview
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("### 🌱 Crop Recommendation")
        st.write("""
            Get intelligent crop suggestions based on soil nutrients (NPK),
            temperature, humidity, pH levels, and rainfall patterns.
        """)
        st.info("📊 Uses Machine Learning to analyze soil and weather data")

    with col2:
        st.markdown("### 🥔 Disease Detection")
        st.write("""
            Upload potato leaf images to identify diseases instantly using
            AI-powered image recognition technology.
        """)
        st.info("🔬 Identifies Early Blight, Late Blight, and Healthy leaves")

    with col3:
        st.markdown("### 💬 AI Assistant")
        st.write("""
            Chat with an intelligent AI assistant for personalized farming
            advice and answers to your agricultural questions.
        """)
        st.info("🤖 24/7 support for all your farming queries")

    # Quick Stats
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### ✨ Why Choose Farmer's Hub?")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Accuracy", "95%+", "AI Models")
    with col_b:
        st.metric("Response Time", "<2 sec", "Fast")
    with col_c:
        st.metric("Crops Supported", "22+", "Growing")
    with col_d:
        st.metric("Diseases Detected", "3", "Potato")

# ==================== CROP RECOMMENDATION PAGE ====================
elif page == "🌱 Crop Recommendation":
    st.markdown("<h2 class='section-header'>🌱 Crop Recommendation System</h2>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    .info-box {
        background: linear-gradient(135deg, #1B5E20 0%, #388E3C 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #81C784;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ How it works:</strong><br>
        Enter soil parameters like nitrogen, phosphorus, potassium, pH, and rainfall. Our model will recommend the most suitable crop for your conditions.
    </div>
    """, unsafe_allow_html=True)

    # Input Form
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Soil Nutrients")
        N = st.number_input("Nitrogen (N) - kg/ha", min_value=0, max_value=200, value=50, help="Nitrogen content in soil")
        P = st.number_input("Phosphorous (P) - kg/ha", min_value=0, max_value=200, value=50, help="Phosphorous content in soil")
        K = st.number_input("Potassium (K) - kg/ha", min_value=0, max_value=200, value=50, help="Potassium content in soil")
        ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1, help="Soil pH level")

    with col2:
        st.subheader("Environmental Conditions")
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Get Crop Recommendation", type="primary"):
        with st.spinner("Analyzing your data..."):
            result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)

            st.markdown(f"""
                <div class='result-box'>
                    <strong>🎯 Recommended Crop:</strong><br>
                    <h2 style='color: #2E7D32; margin-top: 0.5rem;'>{result}</h2>
                </div>
            """, unsafe_allow_html=True)

            # Additional Information
            st.success("✅ This crop is best suited for your current soil and climate conditions!")

            with st.expander("📖 View Input Summary"):
                st.write(f"""
                - **Nitrogen:** {N} kg/ha
                - **Phosphorous:** {P} kg/ha
                - **Potassium:** {K} kg/ha
                - **Temperature:** {temperature}°C
                - **Humidity:** {humidity}%
                - **pH:** {ph}
                - **Rainfall:** {rainfall} mm
                """)

# ==================== POTATO DISEASE DETECTION PAGE ====================
elif page == "🥔 Potato Disease Detection":
    st.markdown("<h2 class='section-header'>🥔 Potato Disease Detection</h2>", unsafe_allow_html=True)

    # Replace the existing complex inline HTML block with the simple, class-based one:
    st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ How it works:</strong><br>
            Upload a clear image of a <strong>potato leaf</strong>. Our AI model will analyze it and
            identify if it's <strong>healthy</strong> or affected by <strong>Early Blight</strong> or <strong>Late Blight</strong> disease.
        </div>
        """, unsafe_allow_html=True)

    # File Upload
    uploaded_file = st.file_uploader("Upload Potato Leaf Image", type=['jpg', 'jpeg', 'png'])

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Analysis")
            if st.button("🔬 Analyze Disease", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = predict_potato_disease(image)

                    disease_info = {
                        "Early Blight": {
                            "severity": "Moderate",
                            "color": "#FF9800",
                            "treatment": "Apply fungicides containing chlorothalonil or mancozeb. Remove affected leaves."
                        },
                        "Late Blight": {
                            "severity": "Severe",
                            "color": "#F44336",
                            "treatment": "Immediate action required! Apply copper-based fungicides. Destroy infected plants."
                        },
                        "Healthy": {
                            "severity": "None",
                            "color": "#4CAF50",
                            "treatment": "No treatment needed. Continue regular crop maintenance."
                        }
                    }

                    if result in disease_info:
                        info = disease_info[result]
                        st.markdown(f"""
                            <div style='background-color: {info['color']}; padding: 1.5rem; border-radius: 10px; color: white;'>
                                <h3>Diagnosis: {result}</h3>
                                <p><strong>Severity:</strong> {info['severity']}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        with st.expander("📋 Treatment Recommendations"):
                            st.write(info['treatment'])

                        with st.expander("🔬 About this Disease"):
                            if result == "Early Blight":
                                st.write("""
                                **Early Blight** is caused by the fungus *Alternaria solani*.
                                Symptoms include dark brown spots with concentric rings on leaves.
                                It typically occurs in warm, humid conditions.
                                """)
                            elif result == "Late Blight":
                                st.write("""
                                **Late Blight** is caused by *Phytophthora infestans* and is
                                highly destructive. It can rapidly spread in cool, wet weather
                                and destroy entire crops within days.
                                """)
                            else:
                                st.write("Your potato plant appears healthy! Continue regular care and monitoring.")
                    else:
                        st.error(result)  # Show error if result is not in disease_info
    else:
        st.info("👆 Please upload an image to get started")

# ==================== CHATBOT PAGE ====================
elif page == "💬 Farmer's Chatbot":
    st.markdown("<h2 class='section-header'>💬 Farmer's AI Assistant</h2>", unsafe_allow_html=True)

    st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ How it works:</strong><br>
            Ask any farming-related questions and get instant answers from our AI assistant.
            Topics include crop management, pest control, irrigation, fertilizers, and more.
        </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! 👋 I'm your Farmer's AI Assistant. How can I help you today?"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about farming..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response (placeholder - integrate your actual chatbot model)
        # Generate response using Groq LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = llm.invoke(prompt)
                    st.write(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                    error_msg = f"❌ Something went wrong: {e}"
                    st.write(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Suggested questions
    st.markdown("### 💡 Suggested Questions:")
    col1, col2 = st.columns(2)

    # Define suggested prompts
    suggested_prompts = [
        ("🌾 What's the best irrigation method?", "What's the best irrigation method?"),
        ("🐛 How to control pests organically?", "How to control pests organically?"),
        ("🌱 When to apply fertilizers?", "When to apply fertilizers?"),
        ("☀️ Crop rotation benefits?", "What are the benefits of crop rotation?")
    ]

    # Render buttons and handle clicks
    for i, (label, prompt_text) in enumerate(suggested_prompts):
        with [col1, col2][i % 2]:
            if st.button(label, key=f"suggested_{i}"):
                st.session_state.messages.append({"role": "user", "content": prompt_text})
                response = llm.invoke(prompt_text)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                st.rerun()

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #2E7D32; padding: 1rem;'>
        <strong>Developed with 💚 for the farming community</strong><br>
        Empowering farmers through technology and innovation
    </div>
""", unsafe_allow_html=True)
