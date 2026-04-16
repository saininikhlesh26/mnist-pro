import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from streamlit_drawable_canvas import st_canvas

# --- Page Config ---
st.set_page_config(
    page_title="MNIST Pro - AI Handwritten Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    :root {
        --primary: #6366f1;
        --secondary: #ec4899;
        --bg-dark: #020617;
        --card-bg: rgba(15, 23, 42, 0.8);
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #020617);
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }

    /* Prediction Result Styling */
    .pred-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1));
        border: 2px solid var(--primary);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        animation: fadeIn 0.5s ease-out;
    }

    .result-digit {
        font-size: 150px;
        font-weight: 700;
        background: linear-gradient(to bottom, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }

    /* Custom Input Fields */
    .stFileUploader section {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
    }

    /* Hide Streamlit components that break immersion */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Model Definition ---
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

MODEL_PATH = 'models/mnist_ann_model.pth'

@st.cache_resource
def load_mnist_model():
    if os.path.exists(MODEL_PATH):
        model = ANNModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

def process_canvas_img(canvas_data):
    if canvas_data.image_data is not None:
        # canvas_data.image_data is (W, H, 4) RGBA
        img = canvas_data.image_data.astype(np.uint8)
        # Convert to Grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        # Resize to 28x28
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        img_tensor = transform(img_resized).unsqueeze(0)
        return img_tensor, img_resized
    return None, None

def process_upload_img(file):
    image = Image.open(file).convert('L')
    img_array = np.array(image)
    
    # Resize to 28x28
    img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert if light background
    if np.mean(img_resized) > 127:
        img_resized = cv2.bitwise_not(img_resized)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)
    return img_tensor, img_resized

# --- Main App ---
def main():
    st.sidebar.markdown("<h2 style='text-align: center;'>⚡ MNIST PRO</h2>", unsafe_allow_html=True)
    app_mode = st.sidebar.radio("Navigation", ["Prediction Interface", "Dataset Visuals", "Model Analytics"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Details")
    st.sidebar.write("**Type:** ANN (PyTorch)")
    st.sidebar.write("**Layers:** 64, 64, 32")
    st.sidebar.write("**Status:** Optimized ✅")

    model = load_mnist_model()
    if model is None:
        st.error("Model not found. Run train.py first.")
        return

    if app_mode == "Prediction Interface":
        st.markdown("<div class='glass-card'><h1>🎨 Intuitive Prediction Studio</h1><p>Draw a digit below or upload a handwritten image for instant neural processing.</p></div>", unsafe_allow_html=True)
        
        col_input, col_output = st.columns([1.2, 1])

        with col_input:
            tabs = st.tabs(["🖌️ Live Drawing", "📤 Manual Upload"])
            
            with tabs[0]:
                st.write("Draw a single digit (0-9)")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1)",
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=300,
                    width=300,
                    drawing_mode="freedraw",
                    key="canvas",
                )
                
            with tabs[1]:
                uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        with col_output:
            st.markdown("<div class='pred-box'>", unsafe_allow_html=True)
            
            tensor = None
            debug_img = None
            
            if tabs[0]._is_active and canvas_result.image_data is not None:
                # Check if user actually drew something
                if np.sum(canvas_result.image_data) > 0:
                    tensor, debug_img = process_canvas_img(canvas_result)
            elif uploaded_file:
                tensor, debug_img = process_upload_img(uploaded_file)

            if tensor is not None:
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    pred = torch.argmax(probs).item()
                    conf = torch.max(probs).item() * 100
                    
                st.markdown(f"<h3>Neural Inference Result</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-digit'>{pred}</div>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color: #ec4899;'>{conf:.2f}% Match</h2>", unsafe_allow_html=True)
                
                st.write("---")
                st.write("**Model Vision (28x28 Preprocessed)**")
                st.image(debug_img, width=150)
                
                # Probs
                st.bar_chart(probs.numpy()[0])
            else:
                st.markdown("<h3>Ready for Input</h3><p>Start drawing on the canvas or upload an image to see the result.</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    elif app_mode == "Dataset Visuals":
        st.markdown("<div class='glass-card'><h1>🖼️ Dataset Visualization</h1></div>", unsafe_allow_html=True)
        test_set = datasets.MNIST(root='./data', train=False, download=True)
        
        num_samples = st.slider("Samples to view", 5, 20, 10)
        fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
        plt.style.use('dark_background')
        for i, ax in enumerate(axes.flat):
            img, label = test_set[i]
            ax.imshow(img, cmap='magma')
            ax.set_title(f"Digit: {label}")
            ax.axis('off')
        st.pyplot(fig)

    elif app_mode == "Model Analytics":
        st.markdown("<div class='glass-card'><h1>📊 Performance Metrics</h1></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Training Trends")
            if os.path.exists('plots/training_history.png'):
                st.image('plots/training_history.png')
        with c2:
            st.write("### Error Distribution (Confusion Matrix)")
            if os.path.exists('plots/confusion_matrix.png'):
                st.image('plots/confusion_matrix.png')

if __name__ == "__main__":
    main()
