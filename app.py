import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os
import io

# Page Config
st.set_page_config(page_title="AI Speech Hub", page_icon="🎙️")
st.title("🎙️ AI Speech Hub")
st.markdown("Convert Speech to Text or Text to Speech using AI.")

# 1. Load Speech-to-Text Model (Whisper)
@st.cache_resource
def load_stt_model():
    # 'tiny' is best for web hosting to avoid memory errors
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

stt_pipe = load_stt_model()

# Select Mode
mode = st.radio("Choose a Tool:", ("Speech-to-Text (STT)", "Text-to-Speech (TTS)"))

# --- MODE 1: SPEECH TO TEXT ---
if mode == "Speech-to-Text (STT)":
    st.header("🎤 Voice to Text")
    st.info("Record or upload audio to transcribe.")
    
    audio_file = st.audio_input("Record your voice")
    
    if audio_file:
        with st.spinner("AI is transcribing..."):
            audio_bytes = audio_file.getvalue()
            result = stt_pipe(audio_bytes)
            text_output = result["text"]
            
            st.success("### Result:")
            st.write(text_output)
            
            if st.button("Clear Recording"):
                st.rerun()

# --- MODE 2: TEXT TO SPEECH ---
else:
    st.header("🔊 Text to Voice")
    user_text = st.text_area("Type what you want the AI to say:", placeholder="Hello world...")
    
    if st.button("Generate Audio"):
        if user_text.strip():
            with st.spinner("Converting..."):
                tts = gTTS(text=user_text, lang='en')
                temp_filename = "speech_output.mp3"
                tts.save(temp_filename)
                
                st.audio(temp_filename, format="audio/mp3")
                
                with open(temp_filename, "rb") as file:
                    st.download_button(
                        label="Download MP3",
                        data=file,
                        file_name="ai_speech.mp3",
                        mime="audio/mp3"
                    )
                os.remove(temp_filename)
        else:
            st.warning("Please enter some text first!")
