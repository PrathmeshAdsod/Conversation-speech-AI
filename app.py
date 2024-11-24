import os
import whisper
from gtts import gTTS
from dotenv import load_dotenv
import openai
import streamlit as st
import tempfile

# Load environment variables
load_dotenv()

# Initialize Whisper Model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

whisper_model = load_whisper_model()

# Streamlit UI
st.title("Conversational AI with Speech-to-Speech Response")
st.write("Record your voice or upload an audio file to start the process.")

# Sidebar Interaction Mode
interaction_mode = st.sidebar.selectbox(
    "Choose Interaction Mode:", ["Record Voice", "Upload Audio"]
)

# Record Voice Functionality with st.audio_input
if interaction_mode == "Record Voice":
    st.write("Use the audio recorder below to record your voice:")
    
    # Record audio using st.audio_input
    audio_data = st.audio_input("Record your voice")
    
    if audio_data:
        st.info("Recording received. Processing...")
        
        # Save the audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data.getvalue())  # Use .getvalue() to extract raw bytes
            temp_audio_path = temp_audio.name

        # Play back the saved audio
        st.audio(temp_audio_path, format="audio/wav")
        st.success("Audio saved and ready for transcription!")


# Upload Audio Functionality
elif interaction_mode == "Upload Audio":
    uploaded_file = st.file_uploader("Upload your audio file (MP3/WAV)", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        st.info("File uploaded. Saving...")
        
        # Save the uploaded audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(uploaded_file.read())  # Write uploaded audio content
            temp_audio_path = temp_audio.name

        # Play back the uploaded audio
        st.audio(temp_audio_path, format="audio/mp3")
        st.success("Audio uploaded and ready for transcription!")

# Transcribe and Process Audio
if 'temp_audio_path' in locals() and temp_audio_path:
    st.write("Processing the audio file for transcription...")
    
    with st.spinner("Transcribing audio..."):
        result = whisper_model.transcribe(temp_audio_path)
        user_text = result["text"]
        st.write("Transcribed Text:", user_text)
        st.success("Transcription complete!")

    # Generate AI Response
    st.write("Generating a conversational response...")
    
    with st.spinner("Generating response..."):
        client = openai.OpenAI(
            api_key=os.environ.get("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )
        
        response = client.chat.completions.create(
            model='Meta-Llama-3.1-8B-Instruct',
            messages=[
                {"role": "system", "content": (
                    "You are a kind, empathetic, and intelligent assistant capable of meaningful conversations and emotional support. "
                    "Your primary goals are: "
                    "1. To engage in casual, friendly, and supportive conversations when the user seeks companionship or emotional relief. "
                    "2. To adapt your tone and responses to match the user's mood, providing warmth and encouragement if they seem distressed or seeking emotional support. "
                    "3. To answer questions accurately and provide explanations when asked, adjusting the depth and length of your answers based on the user's needs. "
                    "4. To maintain a positive and non-judgmental tone, offering helpful advice or lighthearted dialogue when appropriate. "
                    "5. To ensure the user feels heard, understood, and valued during every interaction. "
                    "If the user does not ask a question, keep the conversation engaging and meaningful by responding thoughtfully or with light humor where appropriate."
                )},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            top_p=0.1,
        )
        
        answer = response.choices[0].message.content
        st.write("Response:", answer)
        st.success("Response generated!")

    # Convert response text to speech using gTTS
    st.write("Converting the response to speech...")
    
    with st.spinner("Converting text to speech..."):
        tts = gTTS(text=answer, slow=False)
        response_audio_path = "final_response.mp3"
        tts.save(response_audio_path)
        st.success("Conversion complete!")

    # Play and download the response MP3
    st.audio(response_audio_path, format="audio/mp3")
    st.download_button(
        label="Download the Response",
        data=open(response_audio_path, "rb"),
        file_name="final_response.mp3",
        mime="audio/mpeg",
    )

    # Clean up temporary files
    os.remove(temp_audio_path)
    os.remove(response_audio_path)
