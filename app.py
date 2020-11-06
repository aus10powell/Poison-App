import streamlit as st
from PIL import Image, ImageOps
import io
from img_classification import teachable_machine_classification
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Presence of Poison Oak in normalized_image_array")
st.header("Poison Oak Classification Example")
st.text("Upload an image for classification as poison oak or no poison oak")

print('Starting Streamlit app')
uploaded_file = st.file_uploader("Select an image ...", type=["jpg","png","jpeg"])
# uploaded_file = io.TextIOWrapper(uploaded_file)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image,'./best_model.h5')
    if label <= 0.2:
        st.write("Very unlikely that this is poison oak.")
    elif (label > 0.2) & (label <= 0.4):
        st.write("Unsure from this picture. You may need to retake a closer/clearer picture.")
    elif (label > 0.4) & (label <= 0.7):
        st.write("Decent chance that this is poison oak.")
    else:
        st.write("{:.1f}% chance that this might be poison oak".format(label * 100))
else:
    st.write('No file uploaded')
