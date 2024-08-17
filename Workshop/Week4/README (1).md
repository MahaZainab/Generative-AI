
# Voice-to-Voice Chatbot with Whisper, Llama 8B, and Gradio

This repository contains code for a voice-to-voice chatbot that uses OpenAI's Whisper for speech-to-text transcription, the Llama 8B model (via Groq API) for generating responses, and a text-to-speech engine to convert responses back to audio. The interface is created using Gradio, allowing real-time interactions.

## Features
- **Real-time Transcription:** Uses Whisper to transcribe audio input in real-time.
- **AI-Powered Responses:** Generates intelligent responses using the Llama 8B model via Groq API.
- **Text-to-Speech:** Converts the AI-generated responses back to speech for a seamless voice conversation.
- **Gradio Interface:** User-friendly interface for easy interaction.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/voice-to-voice-chatbot.git
   cd voice-to-voice-chatbot
   ```

2. **Install the required Python packages:**

   ```bash
   !pip install gradio groq openai-whisper pyttsx3
   ```

3. **Set up the Groq API key:**
   
   - Obtain your API key from Groq and set it as an environment variable:

     ```bash
     export GROQ_API_KEY="your_api_key_here"
     ```

## Usage

1. **Run the Colab Notebook:**

   Open the provided notebook in Google Colab and execute the cells to set up and run the chatbot.

2. **Interact with the Chatbot:**

   - Speak into your microphone.
   - The chatbot will transcribe your speech, generate a response, and reply back in audio.

## Code Overview

- **Whisper:** Used for speech-to-text transcription.
- **Groq API:** Connects to Llama 8B model for generating text responses.
- **pyttsx3:** Converts text responses to speech.
- **Gradio:** Provides an interactive web interface.

## Example

After running the code, you will see an interface that allows you to speak to the chatbot. The system will transcribe your input, generate a response, and then speak the response back to you.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
