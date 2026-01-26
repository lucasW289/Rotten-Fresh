import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# --- 1. SETUP & IMPORTS ---
# Add current directory to path so we can import model.py
# This ensures python can find 'model.py' inside the src/ folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import FruitCNN
except ImportError:
    st.error("⚠️ Error: Could not find 'model.py'. Please make sure it is in the same folder as app.py.")
    st.stop()

# --- CONFIGURATION ---
IMG_HEIGHT = 128 
IMG_WIDTH = 128
MODEL_PATH = "../fruit_model.pth"  # Path relative to src/ folder

# ⚠️ CRITICAL: This order is determined by folder names alphabetically!
# 'fresh' starts with F, 'rotten' starts with R. 
# So Index 0 is Fresh, Index 1 is Rotten.
CLASSES = ['Fresh Fruit', 'Rotten Fruit'] 

st.set_page_config(page_title="Fresh vs Rotten Detector", page_icon="🍎")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    """Loads the model architecture and weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the architecture
    model = FruitCNN() 
    
    # Load the weights
    # We check if the file exists to prevent crashing
    if os.path.exists(MODEL_PATH):
        try:
            # map_location ensures it loads on CPU even if trained on GPU
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            st.error(f"❌ Failed to load model weights: {e}")
            return None, device
    else:
        st.warning(f"⚠️ Model file not found at `{MODEL_PATH}`.\n\nPlease place `fruit_model.pth` inside the `models/` folder.")
        return None, device
        
    model.to(device)
    model.eval()
    return model, device

# --- 3. PREPROCESSING ---
def process_image(image_file):
    """
    Transforms the image exactly how the training loader did.
    Resize -> ToTensor -> Normalize(0.5)
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        # This Normalization must match your training code!
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Open image and convert to RGB (fixes issues with PNG transparency)
    image = Image.open(image_file).convert("RGB")
    
    # Apply transforms and add batch dimension (1, 3, 128, 128)
    return transform(image).unsqueeze(0), image

# --- 4. MAIN APP UI ---
st.title("🍎 Fresh vs Rotten Detector")
st.markdown("Upload a fruit image, and the AI will check its quality.")

model, device = load_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Process
        tensor, original_image = process_image(uploaded_file)
        tensor = tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            # Softmax converts raw scores to percentages (e.g. 0.9 and 0.1)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get Results
        class_name = CLASSES[predicted_idx.item()]
        score = confidence.item() * 100
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, caption="Uploaded Image", use_column_width=True)
            
        with col2:
            st.subheader("Analysis Result:")
            
            # Index 0 = Fresh, Index 1 = Rotten
            if predicted_idx.item() == 0:
                st.success(f"✅ **{class_name}**")
                st.write("This fruit looks good to eat!")
            else:
                st.error(f"🤢 **{class_name}**")
                st.write("This fruit is spoiled.")
                
            st.metric("Confidence Score", f"{score:.2f}%")
            
            # Debug info (optional)
            with st.expander("See Raw Data"):
                st.write(f"Fresh Probability: {probabilities[0][0].item():.4f}")
                st.write(f"Rotten Probability: {probabilities[0][1].item():.4f}")