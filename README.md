## 🌾 Farmer's Hub: AI-Powered Agricultural Assistance

**Farmer's Hub** is a comprehensive, all-in-one web application built with **Streamlit** that leverages **Machine Learning** and **Large Language Models (LLMs)** to provide essential services to farmers. It offers intelligent crop recommendations, instant potato disease detection, and a dedicated AI assistant for general farming inquiries.

-----

### ✨ Features

The application is structured around four main pages, accessible via the sidebar navigation:

1.  **🏠 Home**: A welcome page providing an overview of the application's capabilities.
2.  **🌱 Crop Recommendation**: Recommends the most suitable crop based on soil and environmental parameters.
3.  **🥔 Potato Disease Detection**: Instantly diagnoses potato leaf diseases from uploaded images.
4.  **💬 Farmer's Chatbot**: An AI assistant for real-time answers to farming-related questions.

#### 1\. Crop Recommendation System

  * **Input Parameters**: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall.
  * **Technology**: Uses a pre-trained **Machine Learning model (likely a classifier, stored in `crop_rec.pickle`)** to predict one of 22+ crop types.

#### 2\. Potato Disease Detection

  * **Input**: A clear image of a potato leaf (`.jpg`, `.jpeg`, or `.png`).
  * **Technology**: Uses a model leveraging **MobileNetV2** for feature extraction and a **Random Forest classifier (stored in `rf_model_mobilenetv2_potato.joblib`)** for diagnosis.
  * **Diagnosis**: Identifies the leaf as **Healthy**, **Early Blight**, or **Late Blight**, and provides immediate treatment recommendations.

#### 3\. Farmer's AI Assistant

  * **Technology**: Integrates with the **Groq API** using the `ChatGroq` library, utilizing the **`meta-llama/llama-4-scout-17b-16e-instruct`** model for conversational farming advice.
  * **Functionality**: Provides 24/7 support for questions on crop management, pest control, irrigation, fertilizers, and more.

-----

### ⚙️ Technologies Used

  * **Frontend**: [Streamlit](https://streamlit.io/) (for web app creation)
  * **Machine Learning**:
      * **Scikit-learn/Pickle** for Crop Recommendation Model
      * **TensorFlow/Keras (MobileNetV2)** and **Joblib (Random Forest)** for Disease Detection Model
  * **AI/LLM**: [Groq API](https://groq.com/) for the Chatbot functionality
  * **Core Libraries**: `numpy`, `pandas`, `PIL` (Pillow)

-----

### 🚀 Getting Started

To set up and run the Farmer's Hub application locally, follow these steps.

#### Prerequisites

  * Python 3.8+
  * A Groq API key (for the chatbot feature).

#### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/farmers-hub.git
cd farmers-hub
```

#### 2\. Install Dependencies

Install all necessary Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

*(Note: You will need to create a `requirements.txt` file listing all dependencies like `streamlit`, `pickle`, `tensorflow`, `numpy`, `pandas`, `pillow`, `joblib`, and `langchain-groq`.)*

#### 3\. Setup Environment Variable

Set your Groq API key as an environment variable or directly in the `home2.py` file.

**Using Environment Variable (Recommended):**

```bash
# For Linux/macOS
export GROQ_API_KEY="YOUR_GROQ_API_KEY"

# For Windows (Command Prompt)
set GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

**Note:** The provided code has the key hardcoded (`GROQ_API_KEY = "gsk_..."`). For security, it is highly recommended to use environment variables instead.

#### 4\. Place Model Files

The application relies on three external files. Ensure these files are placed in the correct path or update the paths in `home2.py`:

| Feature | File Name | Required Path/Update |
| :--- | :--- | :--- |
| Crop Recommendation | `crop_rec.pickle` | Update path in `load_crop_model()` |
| Disease Detection | `rf_model_mobilenetv2_potato.joblib` | Update path in `load_potato_model()` |

*You will need to acquire these pre-trained model files and place them in your repository.*

#### 5\. Run the Application

Execute the Streamlit app from your terminal:

```bash
streamlit run home2.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

-----

### 🛠️ Customization and Model Paths

If you are encountering `FileNotFoundError` for the models, you must update the hardcoded absolute paths in the helper functions:

```python
# In home2.py

@st.cache_resource
def load_crop_model():
    """Load the crop recommendation model"""
    try:
        # **UPDATE THIS PATH**
        with open('path/to/your/crop_rec.pickle', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

@st.cache_resource
def load_potato_model():
    """Load the potato disease detection model"""
    try:
        # **UPDATE THIS PATH**
        model = jb.load(r'path/to/your/rf_model_mobilenetv2_potato.joblib')
        return model
    except FileNotFoundError:
        return None
```

A common practice is to place these files in a local directory (e.g., `models/`) and use relative paths for better portability.

-----

### 🤝 Contribution

Contributions are welcome\! If you have suggestions for new features, better UI/UX, or bug fixes, please open an issue or submit a pull request.
OR
Contact me here. 

-----

### 🧑‍💻 Developer

Developed with 💚 for the farming community.
*Lokesh Sohanda*
