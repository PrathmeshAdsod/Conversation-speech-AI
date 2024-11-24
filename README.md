# **Conversational AI with Multilingual Speech-to-Speech Response**

### Deployed App Link -- https://conversational-speech-ai.streamlit.app/

###  **Overview**
This project is a powerful **Conversational AI** application that combines **speech to text**, **AI-generated responses**, and **text-to-speech synthesis**. It supports **multilingual input and output**, making it a versatile tool for global communication and more.


### **Key Features**
1. **Speech-to-Text Transcription**:
   - Uses **OpenAI Whisper** for highly accurate transcription in multiple languages.
2. **Multilingual Support**:
   - Transcribe, process, and respond in **multiple languages** seamlessly.
3. **AI-Powered Responses**:
   - Powered by **Sambanova Meta-Llama API**, offering intelligent and empathetic responses tailored to user queries.
4. **Text-to-Speech Conversion**:
   - Converts AI responses into natural-sounding speech using **gTTS** (Google Text-to-Speech).
5. **Interactive Modes**:
   - Record audio directly or upload pre-recorded audio files for transcription and analysis.
6. **Real-World Applications**:
   - **Sentiment Analysis**: Analyze the mood and tone of the transcribed speech.
   - **Translation**: Transcribe speech in one language, translate it, and respond in another.
   - **Multimodal AI**: Support for text, audio, and potential future integrations like image-based processing.

---

### **Tech Stack**
**Speech Recognition**: OpenAI Whisper
**AI Processing**: Sambanova Meta-Llama API
**Text-to-Speech**: gTTS
**Frontend**: Streamlit
**Environment Management**: python dotenv, streamlit secrets


### Run Code to Local Machine
1. Fork the repository
2. Clone repository by giving command
   - git clone https://github.com/PrathmeshAdsod/Conversation-speech-AI.git 
3. Install Dependencies
   - pip install -r requirements.txt
4. Set up env file
   - create .env file in root directory
   - write  SAMBANOVA_API_KEY=your_sambanova_api_key
5. run command in termminal
   - streamlit run app.py

### If you require deployment of this app to streamlit cloud community
1. Create secrets.toml by creating a folder .streamlit
2. Give SAMBANOVA_API_KEY="your_sambanova_api_key"
3. Also give SAMBANOVA_API_KEY="your_sambanova_api_key" in secrets of advance setting
4. add this .streamlit/secrets.toml in .gitignore