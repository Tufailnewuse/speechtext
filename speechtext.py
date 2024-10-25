import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from scipy.io import wavfile
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
    # Read the audio file with scipy
    samplerate, data = wavfile.read(audio_file)

    # Ensure the audio data is in float format for model compatibility
    waveform = torch.tensor(data, dtype=torch.float32)

    # Resample to 16000Hz if the sample rate is different
    if samplerate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)
        waveform = resampler(waveform)

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
