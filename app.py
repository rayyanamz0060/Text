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
    # Using the 'tiny' model for faster performance on Streamlit Free Tier
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

stt_pipe = load_stt_model()

# Select Mode
mode = st.radio("Choose a Tool:", ("Speech-to-Text (STT)", "Text-to-Speech (TTS)"))

# --- MODE 1: SPEECH TO TEXT ---
if mode == "Speech-to-Text (STT)":
    st.header("🎤 Voice to Text")
    st.info("Click the mic below and start speaking. The AI will transcribe it.")
    
    audio_file = st.audio_input("Record your voice")
    
    if audio_file:
        with st.spinner("AI is listening and transcribing..."):
            # Get the bytes from the uploaded audio
            audio_bytes = audio_file.getvalue()
            
            # Run the AI model
            result = stt_pipe(audio_bytes)
            text_output = result["text"]
            
            st.success("### Result:")
            st.write(text_output)
            
            # Button to copy/reset
            if st.button("Clear Recording"):
                st.rerun()

# --- MODE 2: TEXT TO SPEECH ---
else:
    st.header("🔊 Text to Voice")
    user_text = st.text_area("Type what you want the AI to say:", placeholder="Enter text here...")
    
    if st.button("Generate Audio"):
        if user_text.strip():
            with st.spinner("Converting text to speech..."):
                # Use gTTS to create audio in memory
                tts = gTTS(text=user_text, lang='en')
                
                # Save to a temporary file
                temp_filename = "speech_output.mp3"
                tts.save(temp_filename)
                
                # Display the player
                st.audio(temp_filename, format="audio/mp3")
                
                # Provide a download button
                with open(temp_filename, "rb") as file:
                    st.download_button(
                        label="Download MP3",
                        data=file,
                        file_name="ai_speech.mp3",
                        mime="audio/mp3"
                    )
                
                # Cleanup
                os.remove(temp_filename)
        else:
            st.warning("Please enter some text first!")
