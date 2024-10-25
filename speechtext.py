import streamlit as st
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

# Load the pretrained model and processor from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Set up Streamlit app interface
st.title("Speech-to-Text Converter")
st.write("Upload an audio file (WAV format) to transcribe to text.")

# Upload audio file
audio_file = st.file_uploader("Choose an audio file", type=["wav"])

if audio_file is not None:
    # Load the audio file
    data, samplerate = sf.read(audio_file)
    # Ensure the audio is in the correct format
    waveform = torch.tensor(data).float()
    
    # Resample to 16000Hz if necessary
    if samplerate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)(waveform)

    # Tokenize and transcribe
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Get predicted IDs and convert to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Display transcription
    st.write("### Transcription")
    st.write(transcription)
