import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Page setup
st.set_page_config(page_title="COVID-19 X-ray Classifier", layout="wide")

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('covid_model1.keras')
        return model
    except:
        st.error("Model loading failed")
        return None

# Preprocess image
def preprocess_image(image):
    image = image.convert('RGB').resize((128, 128))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Main app
def main():
    st.title("🩺 COVID-19 Chest X-ray Classifier")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("Classifies X-rays into: COVID-19, Normal, or Pneumonia")
        st.write("**Accuracy:** 95.65%")
        
        st.header("Instructions")
        st.write("1. Upload X-ray image")
        st.write("2. Click Analyze")
        st.write("3. View results")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload X-ray")
        uploaded_file = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            if st.button("Analyze X-ray", type="primary"):
                with st.spinner("Analyzing..."):
                    model = load_model()
                    
                    if model:
                        # Process image
                        processed_img = preprocess_image(image)
                        
                        # Predict
                        predictions = model.predict(processed_img, verbose=0)
                        
                        # Get results
                        classes = ['COVID19', 'NORMAL', 'PNEUMONIA']
                        pred_idx = np.argmax(predictions[0])
                        pred_class = classes[pred_idx]
                        confidence = float(predictions[0][pred_idx])
                        
                        # Show results in col2
                        with col2:
                            st.subheader("Results")
                            
                            if pred_class == 'COVID19':
                                st.error(f"🦠 COVID-19 Detected")
                            elif pred_class == 'PNEUMONIA':
                                st.warning(f"😷 Pneumonia Detected")
                            else:
                                st.success(f"✅ Normal")
                            
                            st.write(f"**Confidence:** {confidence:.1%}")
                            
                            # Probabilities
                            st.subheader("Probabilities")
                            for i, cls in enumerate(classes):
                                prob = predictions[0][i]
                                st.write(f"{cls}: {prob:.1%}")
                            
                            # Recommendation
                            st.subheader("Recommendation")
                            if pred_class == 'COVID19':
                                st.info("Consult doctor for COVID testing")
                            elif pred_class == 'PNEUMONIA':
                                st.info("Seek medical attention")
                            else:
                                st.info("No concerns detected")
                            
                            st.warning("For screening only - consult doctor for diagnosis")

if __name__ == "__main__":
    main()