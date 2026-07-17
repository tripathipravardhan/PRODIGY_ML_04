"""Local Streamlit application for hand gesture prediction."""
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image

from gesture_utils import load_labels, prepare_image

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "gesture_model.keras"
LABELS_PATH = ARTIFACTS / "labels.json"


st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
)


@st.cache_resource
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(image):
    scores = get_model().predict(prepare_image(image), verbose=0)[0]
    labels = load_labels(LABELS_PATH)

    return [
        (labels[index], float(scores[index]))
        for index in scores.argsort()[-3:][::-1]
    ]


st.title("✋ Hand Gesture Recognition")
st.write("Upload a hand image or take a picture to classify its gesture.")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.warning(
        "First copy gesture_model.keras and labels.json into the artifacts folder."
    )
    st.stop()

source = st.radio(
    "Choose input",
    ["Upload image", "Camera"],
    horizontal=True,
)

if source == "Upload image":
    file = st.file_uploader(
        "Choose a hand image",
        type=["png", "jpg", "jpeg"],
    )
else:
    file = st.camera_input("Take a picture")

if file:
    image = Image.open(file)

    st.image(
        image,
        caption="Image to classify",
        use_container_width=True,
    )

    results = predict(image)

    st.success(
        f"Prediction: {results[0][0].title()} "
        f"({results[0][1]:.1%})"
    )

    st.subheader("Top predictions")

    for label, confidence in results:
        st.write(f"{label.title()}: {confidence:.1%}")
        st.progress(confidence) 