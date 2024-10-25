import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from scipy.io import wavfile

# Load the pretrained model and processor from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Set up Streamlit app interface
st.title("Speech-to-Text Converter")
st.write("Upload an audio file (WAV format) to transcribe to text.")

# Upload audio file
audio_file = st.file_uploader("Choose an audio file", type=["wav"])

# Version 1: Original Code using `soundfile` (commented out due to compatibility issues)
def transcribe_with_soundfile(audio_file):
    import soundfile as sf
    data, samplerate = sf.read(audio_file)
    waveform = torch.tensor(data).float()
    if samplerate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)(waveform)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Version 2: Using `scipy.io.wavfile`
def transcribe_with_scipy(audio_file):
    samplerate, data = wavfile.read(audio_file)
    waveform = torch.tensor(data, dtype=torch.float32)
    if samplerate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)(waveform)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Version 3: Simplified, with option to switch between versions
if audio_file is not None:
    st.write("Choose a version to transcribe with:")
    version = st.selectbox("Select Version", ("Version 1 - Soundfile", "Version 2 - Scipy"))

    if version == "Version 1 - Soundfile":
        try:
            transcription = transcribe_with_soundfile(audio_file)
            st.write("### Transcription (Version 1 - Soundfile)")
            st.write(transcription)
        except ImportError:
            st.error("Soundfile library is not installed. Please use Version 2 or install Soundfile.")
    elif version == "Version 2 - Scipy":
        transcription = transcribe_with_scipy(audio_file)
        st.write("### Transcription (Version 2 - Scipy)")
        st.write(transcription)
