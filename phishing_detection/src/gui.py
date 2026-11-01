import streamlit as st
import pytesseract
from PIL import Image
from integrator import process_image_and_predict

st.title("👾 Phishing Detector")
st.markdown("Enter text or upload an image")

input_method = st.radio("Select input type:", ["Text", "Upload image (screenshot)"])

text = ""
image = None 

if input_method == "Text":
    text = st.text_area("📝 Type here :", height=200)

else:
    image_file = st.file_uploader("Upload image (jpg/png)", type=["png", "jpg", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

if st.button("🔍 Predict"):
    try:
        # Call process_image_and_predict and unpack the returned values
        extracted_text, prediction_label, explanation = process_image_and_predict(image, text)
        
        # Display the extracted text
        st.info("🧾 Result text from OCR or input:")
        st.write(extracted_text)
        
        # Display the prediction result
        if prediction_label == "phishing":
            st.error("⚠️ This is likely to be phishing.")
        else:
            st.success("✅ This is not phishing.")
        
        # Display the explanation
        st.info("🧾 Confidence:")
        st.write(explanation)
    except ValueError as e:
        st.warning(str(e))
