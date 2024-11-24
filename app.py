import os
import whisper
from gtts import gTTS
from dotenv import load_dotenv
import openai
import streamlit as st
import tempfile

# Load environment variables
load_dotenv()
custom_instructions = ""
# Initialize Whisper Model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

whisper_model = load_whisper_model()

# Streamlit UI
st.title("Conversational AI with Speech-to-Speech Response")
st.markdown(
    "This application lets you record or upload an audio file, transcribes it using Whisper, "
    "processes it with an AI model, and responds in both text and speech formats."
)

# Sidebar Interaction Mode
st.sidebar.header("Interaction Mode")
interaction_mode = st.sidebar.radio(
    "Choose Interaction Mode:",
    ["Record Voice", "Upload Audio"],
    index=0
)
"""
st.sidebar.markdown(
    "💡 *Record your voice or upload an audio file. "
    "The app transcribes the input, generates an AI response, and converts it to speech.*"
)
"""


# Record Voice Functionality
if interaction_mode == "Record Voice":
    st.subheader("Record Your Voice")
    audio_data = st.audio_input("Record your voice")

    if audio_data:
        st.info("Recording received. Processing...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data.getvalue())  # Use .getvalue() to extract raw bytes
            temp_audio_path = temp_audio.name

        st.audio(temp_audio_path, format="audio/wav")
        st.success("Audio saved and ready for transcription!")

# Upload Audio Functionality
elif interaction_mode == "Upload Audio":

     # Collect Additional Instructions
    st.write("### Add Custom Instructions for the AI Model:")
    custom_instructions = st.text_area(
        "Provide additional context or instructions (e.g., 'Perform sentiment analysis on this file')",
        placeholder="Enter your instructions here..."
    )

    st.subheader("📂 Upload Your Audio File")
    uploaded_file = st.file_uploader(
        "Upload your audio file (MP3/WAV)", type=["mp3", "wav"]
    )

    if uploaded_file is not None:
        st.info("File uploaded. Processing...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        st.audio(temp_audio_path, format="audio/mp3")
        st.success("Audio uploaded and ready for transcription!")

# Transcribe and Process Audio
if "temp_audio_path" in locals() and temp_audio_path:
    st.subheader("📄 Transcription and Response")
    with st.spinner("Transcribing audio..."):
        result = whisper_model.transcribe(temp_audio_path)
        user_text = result.get("text", "Transcription failed.")
        st.write("### Transcribed Text:")
        st.markdown(f"> {user_text}")
        st.success("Transcription complete!")

   
    combined_input = f"{custom_instructions}\n\n{user_text}" if custom_instructions else user_text

    # Generate AI Response
    with st.spinner("Generating response..."):
        try:
            client = openai.OpenAI(
                api_key=st.secrets["SAMBANOVA_API_KEY"],
                base_url="https://api.sambanova.ai/v1",
            )
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
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
                    {"role": "user", "content": combined_input},
                ],
                temperature=0.7,
                top_p=0.9,
            )
            answer = response.choices[0].message.content
            st.write("### AI Response:")
            st.markdown(f"> {answer}")
            st.success("Response generated!")
        except Exception as e:
            st.error(f"Failed to generate response: {e}")
            answer = None

    # Convert AI Response to Speech
    if answer:
        with st.spinner("Converting response to speech..."):
            tts = gTTS(text=answer, slow=False)
            response_audio_path = "final_response.mp3"
            tts.save(response_audio_path)
            st.audio(response_audio_path, format="audio/mp3")
            st.download_button(
                label="Download the Response",
                data=open(response_audio_path, "rb"),
                file_name="final_response.mp3",
                mime="audio/mpeg",
            )
            st.success("Response converted to speech!")

    # Clean up temporary files
    os.remove(temp_audio_path)
    if "response_audio_path" in locals():
        os.remove(response_audio_path)
