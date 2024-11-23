import os
import whisper
from gtts import gTTS
from dotenv import load_dotenv
import openai
import streamlit as st
import tempfile
from pydub import AudioSegment
import wave
import pyaudio

# Load environment variables
load_dotenv()

# Initialize Whisper Model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

whisper_model = load_whisper_model()

# Streamlit UI
st.title("Conversational AI with Speech-to-Speech Response")
st.write("Upload an audio file or record your voice to start the process.")

# Add a sidebar for interaction options
interaction_mode = st.sidebar.selectbox(
    "Choose Interaction Mode:", ["Record Voice", "Upload Audio"]
)

# Record Voice Functionality using pydub and pyaudio
def record_audio(filename, duration=5, sample_rate=44100):
    st.info(f"Recording for {duration} seconds...")
    p = pyaudio.PyAudio()
    
    # Open a stream for recording
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded frames as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    st.success("Recording complete!")

# Process Audio Input
if interaction_mode == "Record Voice":
    duration = st.slider("Select Recording Duration (seconds):", min_value=10, max_value=120, step=10)
    record_btn = st.button("Start Recording")
    
    if record_btn:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            record_audio(temp_audio.name, duration=duration)
            temp_audio_path = temp_audio.name
            st.audio(temp_audio_path, format="audio/wav")
elif interaction_mode == "Upload Audio":
    uploaded_file = st.file_uploader("Upload your audio file (MP3/WAV)", type=["mp3", "wav"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name
            st.audio(temp_audio_path, format="audio/mp3")

# Process and Transcribe Audio
if 'temp_audio_path' in locals() and temp_audio_path is not None:
    st.write("Processing the audio file...")
    
    # If the uploaded or recorded audio is in MP3 format, convert it to WAV for Whisper
    if temp_audio_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(temp_audio_path)
        temp_audio_path = temp_audio_path.replace(".mp3", ".wav")
        audio.export(temp_audio_path, format="wav")

    # Transcribe audio using Whisper
    result = whisper_model.transcribe(temp_audio_path)
    user_text = result["text"]
    st.write("Transcribed Text:", user_text)

    # Generate AI Response
    st.write("Generating a conversational response...")
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

    # Convert response text to speech using gTTS
    st.write("Converting the response to speech...")
    tts = gTTS(text=answer,  slow=False)
    response_audio_path = "final_response.mp3"
    tts.save(response_audio_path)

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
