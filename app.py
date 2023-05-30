import streamlit as st
from PIL import Image, ImageOps
import io
from img_classification import teachable_machine_classification, load_model

from tensorflow import keras


st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Detecting presence of Poison Oak")
st.header("Poison Oak Classification Example")
st.text("Upload an image for classification as poison oak or no poison oak")


# Load trained model
model = load_model("./best_model.h5")

print("Starting Streamlit app")
uploaded_file = st.file_uploader("Select an image ...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(img=image, model=model)
    if label <= 0.2:
        st.write("Very unlikely that this is poison oak.")
    elif (label > 0.2) & (label <= 0.6):
        st.write(
            "Unsure from this picture. You may need to retake a closer/clearer picture."
        )
    elif (label > 0.6) & (label <= 0.7):
        st.write("Decent chance that this is poison oak.")
    else:
        st.write("{:.1f}% chance that this might be poison oak".format(label * 100))
else:
    st.write("No file uploaded")
