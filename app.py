import gradio as gr
import numpy as np
import cv2
import pickle

# ===============================
# LOAD MODEL & LABEL ENCODER
# ===============================
with open("dt_Space_data_project_best_2.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ===============================
# IMAGE PREPROCESSING (MATCH TRAINING)
# ===============================
def preprocess_image(image):
    if image is None:
        return None

    # PIL → NumPy
    image = np.array(image)

    # ✅ Convert to GRAYSCALE (MANDATORY)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ✅ Resize
    image = cv2.resize(image, (64, 64))

    # ✅ Normalize
    image = image.astype("float32") / 255.0

    # ✅ Flatten → 4096
    image = image.flatten().reshape(1, -1)

    return image


# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_space_object(image):
    try:
        img = preprocess_image(image)

        if img is None:
            return "Please upload an image."

        pred_encoded = model.predict(img)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        return f"Predicted Space Object: {pred_label}"

    except Exception as e:
        return f"Prediction Error: {str(e)}"


# ===============================
# BACKGROUND IMAGE CSS
# ===============================
custom_css = """
body {
    background-image: url("file=Hubble Space Telescope.jpeg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
.gradio-container {
    background: rgba(0, 0, 0, 0.65) !important;
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
"""

# ===============================
# GRADIO APP
# ===============================
app = gr.Interface(
    fn=predict_space_object,
    inputs=gr.Image(type="pil", label="Upload Astronomical Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Astronomical Image Classification",
    description="Classifies images as Galaxy, Nebula, or Star Cluster using Machine Learning.",
    css=custom_css
)

if __name__ == "__main__":
    app.launch()
