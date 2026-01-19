import streamlit as st
import requests

st.title("🎙️ AI Meeting Notes Generator")

audio_file = st.file_uploader(
    "Upload your meeting audio (.mp3 or .wav)",
    type=["mp3", "wav"]
)

if audio_file:
    st.audio(audio_file)

    if st.button("Generate Notes"):
        response = requests.post(
            "http://localhost:8000/process/",
            files={"file": audio_file}
        )

        output = response.json()

        st.subheader("📝 Summary")
        st.write(output["summary"])

        st.subheader("✅ Action Items")
        st.write(output["action_items"])

        with st.expander("📄 Full Transcript"):
            st.text_area(
                "Transcript",
                value=output["transcript"],
                height=300
            )