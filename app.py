# Streamlit Web Interface for Speech Personality Prediction
import streamlit as st
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import tempfile
import os
import json

# --- Page Config ---
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🎙️",
    layout="centered"
)

# --- Model Architecture (must match train.py) ---
class PersonalityModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PersonalityModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, combined_features):
        x = self.layer1(combined_features)
        x = self.activation(x)
        return torch.sigmoid(self.output_layer(x))

# --- Load Models (cached so they only load once) ---
@st.cache_resource
def load_models():
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        Wav2Vec2Processor, Wav2Vec2Model,
        BertTokenizer, BertModel
    )

    with st.spinner("Loading Whisper model..."):
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    with st.spinner("Loading Wav2Vec2 model..."):
        audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    with st.spinner("Loading BERT model..."):
        text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_model = BertModel.from_pretrained("bert-base-uncased")

    with st.spinner("Loading personality model weights..."):
        personality_model = PersonalityModel(1536, 5)
        if not os.path.exists("personality_model_final.pth"):
            st.error("❌ personality_model_final.pth not found. Please place it in the project folder.")
            st.stop()
        personality_model.load_state_dict(torch.load("personality_model_final.pth", map_location="cpu"))
        personality_model.eval()

    return processor, whisper_model, audio_processor, audio_model, text_tokenizer, text_model, personality_model

# --- Prediction Function ---
def predict(audio_path, models):
    processor, whisper_model, audio_processor, audio_model, text_tokenizer, text_model, personality_model = models

    # Load and resample audio
    speech, sample_rate = sf.read(audio_path)
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)  # Convert stereo to mono automatically
    if sample_rate != 16000:
        speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)

    device = "cpu"
    audio_model.to(device)
    text_model.to(device)
    personality_model.to(device)

    # Transcription
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # BERT text features
    text_inputs = text_tokenizer(transcription, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        text_features = text_model(**text_inputs).last_hidden_state.mean(dim=1)

    # Wav2Vec2 audio features
    audio_inputs = audio_processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_features = audio_model(**audio_inputs).last_hidden_state.mean(dim=1)

    # Personality prediction
    combined = torch.cat((text_features, audio_features), dim=1)
    with torch.no_grad():
        scores = personality_model(combined).squeeze().tolist()

    traits = {
        "Openness":          round(scores[0], 4),
        "Conscientiousness": round(scores[1], 4),
        "Extraversion":      round(scores[2], 4),
        "Agreeableness":     round(scores[3], 4),
        "Neuroticism":       round(scores[4], 4),
    }

    return transcription, traits

# --- UI ---
st.title("🎙️ Speech Personality Predictor")
st.markdown("Upload a audio file to transcribe it and predict **Big Five (OCEAN) personality traits**.")
st.divider()

# Load models
models = load_models()
st.success("✅ All models loaded and ready!")
st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    help="Stereo files are automatically converted to mono."
)

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍 Analyse Personality", type="primary"):
        # Save uploaded file to temp location
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Transcribing and analysing... this may take a minute."):
                transcription, traits = predict(tmp_path, models)

            # Results
            st.divider()
            st.subheader("📝 Transcription")
            st.info(transcription)

            st.subheader("🧠 Personality Traits (OCEAN)")

            trait_info = {
                "Openness":          ("🎨", "Creativity, curiosity, and openness to new experiences"),
                "Conscientiousness": ("📋", "Organisation, dependability, and self-discipline"),
                "Extraversion":      ("🗣️", "Sociability, assertiveness, and positive emotions"),
                "Agreeableness":     ("🤝", "Cooperativeness, warmth, and trust"),
                "Neuroticism":       ("🌊", "Emotional sensitivity and tendency to experience negative emotions"),
            }

            for trait, score in traits.items():
                emoji, description = trait_info[trait]
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{emoji} {trait}**")
                    st.caption(description)
                    st.progress(score)
                with col2:
                    st.metric(label="Score", label_visibility="collapsed", value=f"{score:.2%}")

            st.divider()

            # Download JSON
            result_json = json.dumps({"transcription": transcription, "traits": traits}, indent=2)
            st.download_button(
                label="⬇️ Download Results as JSON",
                data=result_json,
                file_name="personality_results.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
        finally:
            os.unlink(tmp_path)

st.divider()
st.caption("Built with Whisper · Wav2Vec2 · BERT · Streamlit")
