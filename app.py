# pip install streamlit-drawable-canvas
# pip install opencv-python

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2
import tensorflow as tf

# Load the model safely with legacy fallback
@st.cache_resource
def load_my_model():
    try:
        # standard load
        return tf.keras.models.load_model("digit_model.keras", compile=False)
    except Exception:
        try:
            # legacy format load if standard fails due to Keras 3 issues
            return tf.keras.src.legacy.saving.models.load_model("digit_model.keras", compile=False)
        except Exception as inner_e:
            raise inner_e

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure that 'digit_model.keras' is uploaded to the main folder of your GitHub repository.")
    st.stop()

# --- हे फंक्शन डिलीट झाल्यामुळे एरर येत होता, ते आता फिक्स केले आहे ---
def preprocess_canvas(img_rgba):
    # Convert to uint8 0..255
    arr = (img_rgba * 255).astype(np.uint8) if img_rgba.max() <= 1.0 else img_rgba.astype(np.uint8)
    
    # Composite on black background using alpha channel if it exists
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3] / 255.0
        for c in range(3):
            arr[:, :, c] = (arr[:, :, c] * alpha + 0 * (1 - alpha)).astype(np.uint8)
        rgb = arr[:, :, :3]
    else:
        rgb = arr[:, :, :3]

    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Invert background if it is light
    mean_val = gray.mean()
    if mean_val > 127:
        gray = 255 - gray

    # Binary threshold
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box
    coords = cv2.findNonZero(th)
    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)

    x, y, w_box, h_box = cv2.boundingRect(coords)

    # Crop with padding
    padding = int(0.15 * max(w_box, h_box))
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w_box + padding, gray.shape[1])
    y2 = min(y + h_box + padding, gray.shape[0])
    cropped = th[y1:y2, x1:x2]

    # Resize keeping aspect ratio
    h_c, w_c = cropped.shape
    if h_c > w_c:
        new_h = 20
        new_w = int(round((w_c / h_c) * 20))
    else:
        new_w = 20
        new_h = int(round((h_c / w_c) * 20))
    if new_w == 0: new_w = 1
    if new_h == 0: new_h = 1
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Smooth and normalize
    canvas = cv2.GaussianBlur(canvas, (3,3), 0)
    canvas = canvas.astype(np.float32) / 255.0

    return canvas

st.title("Handwritten Digit Identification")

canvas_result = st_canvas(
    width=280,
    height=280,
    background_color="#000000",
    stroke_color="#FFFFFF",
    drawing_mode="freedraw",
    stroke_width=20
)

def is_canvas_empty(img_rgba, threshold=10):
    if img_rgba is None:
        return True
    arr = (img_rgba * 255).astype(np.uint8) if img_rgba.max() <= 1.0 else img_rgba.astype(np.uint8)
    if arr.shape[2] == 4:
        if np.all(arr[:, :, 3] == 0):
            return True
    return arr[:, :, :3].max() < threshold

def prepare_input_for_model(img28, model):
    input_shape = model.input_shape
    if len(input_shape) == 2:
        x = img28.reshape(1, 28*28).astype(np.float32)
    elif len(input_shape) == 3:
        x = img28.reshape(1, 28, 28).astype(np.float32)
    else:
        x = img28.reshape(1, 28, 28, 1).astype(np.float32)
    return x

if st.button("Recognize"):
    img = canvas_result.image_data
    if is_canvas_empty(img):
        st.error("No image is drawn")
    else:
        st.write("Recognizing...")
        proc = preprocess_canvas(img)
        
        st.subheader("Processed Image (28x28)")
        st.image((proc * 255).astype(np.uint8), width=140)
        
        x = prepare_input_for_model(proc, model)
        preds = model.predict(x)
        probs = preds.ravel()
        top3_idx = probs.argsort()[::-1][:3]

        st.subheader("Top Predictions")
        for i in top3_idx:
            st.write(f"Digit {i} — Probability: {probs[i]:.4f}")

        predicted_class = int(top3_idx[0])
        st.success(f"🎯 Predicted Digit: **{predicted_class}** (p={probs[predicted_class]:.3f})")
