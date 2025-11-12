import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

# Load the saved models
feature_extractor = tf.keras.models.load_model('feature_extractor.h5')
rf_classifier = joblib.load('rf_classifier.pkl')

# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Enhanced eradication methods with image and rich descriptions
eradication_methods = {
    'Potato___Early_blight': {
        'image': "potato-early-blight-leaves.jpg",
        'description': """
### 🐛 Early Blight - *Alternaria solani*

**Symptoms:**
- Brown spots with concentric rings on older leaves.
- Yellowing and drying leaves starting from the bottom.

**Pesticide Treatment:**
- Apply fungicides like **Chlorothalonil**, **Mancozeb**, or **Azoxystrobin** early in the season.
- Repeat every 7–10 days during humid conditions.

**Organic Treatment:**
- Use **neem oil** or **compost tea spray** weekly.
- Remove infected leaves manually.
- Improve air circulation through proper spacing.

**Prevention Tips:**
- Rotate crops every 2–3 years.
- Avoid overhead watering.
- Ensure good drainage and soil health.
        """
    },

    'Potato___Late_blight': {
        'image': "LateBlight03.JPG",
        'description': """
### 🌧️ Late Blight - *Phytophthora infestans*

**Symptoms:**
- Large, dark brown to black lesions on leaves and stems.
- White fungal growth on undersides of leaves in humid weather.
- Rapid plant collapse in severe infections.

**Pesticide Treatment:**
- Use **Metalaxyl**, **Copper-based fungicides**, or **Cyazofamid**.
- Apply before disease spreads, especially in wet weather.

**Organic Treatment:**
- Remove and destroy infected plants immediately.
- Apply **baking soda + oil + soap** spray.
- Use resistant varieties if possible.

**Prevention Tips:**
- Avoid waterlogged soils.
- Disinfect tools after contact.
- Keep leaves dry—water in early morning.
        """
    },

    'Potato___healthy': {
        'image': "healthy_leaf.jpg",
        'description': """
### 🌿 Healthy Potato Plant

Your plant shows no signs of disease. Keep up the good work!

**Care Tips:**
- Regularly check leaves for early signs of spots or lesions.
- Fertilize with balanced NPK fertilizers.
- Mulch around the base to retain moisture and control weeds.
- Keep good air flow between plants.

**Preventative Measures:**
- Crop rotation.
- Use certified disease-free seed potatoes.
- Water early in the day to avoid prolonged moisture.
        """
    }
}

# Awareness links
disease_awareness_links = {
    'Potato___Early_blight': [
        {
            "title": "📝 Understanding Early Blight in Potatoes (Symptoms, Lifecycle & Damage)",
            "url": "https://plantvillage.psu.edu/topics/potato/infos/disease-early-blight"
        },
        {
            "title": "🌱 Organic Control of Early Blight - University of Minnesota Extension",
            "url": "https://extension.umn.edu/disease-management/early-blight-potato"
        },
        {
            "title": "🎥 Video Guide: Identifying & Managing Early Blight (YouTube)",
            "url": "https://www.youtube.com/watch?v=J1KefcszLxA"
        },
        {
            "title": "👩‍🌾 Farmer Forum: Real Experiences with Early Blight Treatments",
            "url": "https://www.growveg.com/guides/early-blight-in-potatoes/"
        }
    ],

    'Potato___Late_blight': [
        {
            "title": "🧬 Late Blight: Causes, Spread, and Devastation in Crops",
            "url": "https://cipotato.org/pressroom/blogs/late-blight-of-potato/"
        },
        {
            "title": "🔬 Management of Late Blight - UC Integrated Pest Management",
            "url": "https://ipm.ucanr.edu/PMG/r607101311.html"
        },
        {
            "title": "🎥 Late Blight: How It Looks & How to Stop It (YouTube)",
            "url": "https://www.youtube.com/watch?v=6F_Wv0U3Lf4"
        },
        {
            "title": "📚 Late Blight Research Publications - PotatoPro",
            "url": "https://www.potatopro.com/topics/late-blight"
        }
    ],

    'Potato___healthy': [
        {
            "title": "🌿 Best Practices for Growing Healthy Potato Plants",
            "url": "https://www.ag.ndsu.edu/publications/crops/potato-production-and-management"
        },
        {
            "title": "🚜 10 Tips to Prevent Common Potato Diseases",
            "url": "https://www.planetnatural.com/pest-problem-solver/plant-disease/potato-disease/"
        },
        {
            "title": "🔄 Crop Rotation & Soil Health Strategies",
            "url": "https://www.canr.msu.edu/news/crop_rotation_and_disease_control_in_potato_fields"
        },
        {
            "title": "👨‍🔬 Healthy Plant Checklist (PDF Download)",
            "url": "https://edis.ifas.ufl.edu/pdf/CV/CV29800.pdf"
        }
    ]
}


# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Prediction function
def predict(image):
    preprocessed_image = preprocess_image(image)
    features = feature_extractor.predict(preprocessed_image)
    prediction = rf_classifier.predict(features)
    predicted_class = class_names[prediction[0]]
    return predicted_class

# Streamlit App UI
st.set_page_config(page_title="Potato Disease Classifier", layout="centered")
st.title("🥔 Potato Disease Classification")
st.write("Upload a potato leaf image to classify it as **Early Blight**, **Late Blight**, or **Healthy**.")

# Image upload
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    st.write("🔍 Classifying...")
    predicted_class = predict(image)
    display_class = predicted_class.split('___')[-1].replace('_', ' ')
    st.write(f"### ✅ Prediction: **{display_class}**")

    # Display eradication methods
    st.subheader("💊 Eradication Methods")
    methods = eradication_methods[predicted_class]
    with st.expander("📖 View Eradication Guide", expanded=True):
        st.image(methods['image'], use_column_width=True, caption="Visual Reference")
        st.markdown(methods['description'], unsafe_allow_html=True)

    # Awareness resources
    st.subheader("🧠 Disease Awareness Resources")
    for link in disease_awareness_links[predicted_class]:
        st.markdown(f"- [{link['title']}]({link['url']})")

