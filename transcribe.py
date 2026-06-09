# main transcription and personality prediction script
import os
import sys
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel

AUDIO_FILE = "audio.wav"
MODEL_FILE = "personality_model_final.pth"

# --- 1. Define the Personality Model Architecture ---

# This MUST match the class definition used in train.py
class PersonalityModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PersonalityModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, combined_features):
        x = self.layer1(combined_features)
        x = self.activation(x)
        output = self.output_layer(x)
        # Apply sigmoid to ensure output is between 0 and 1
        return torch.sigmoid(output)


# --- 2. Pre-flight Checks ---

def check_requirements():
    """Check all required files exist before loading heavy models."""
    errors = []

    if not os.path.exists(AUDIO_FILE):
        errors.append(
            f"[MISSING FILE] '{AUDIO_FILE}' not found.\n"
            "  → Place a mono WAV file named 'audio.wav' in the project folder.\n"
            "  → To convert a stereo file: ffmpeg -i your_file.wav -ac 1 audio.wav"
        )

    if not os.path.exists(MODEL_FILE):
        errors.append(
            f"[MISSING FILE] '{MODEL_FILE}' not found.\n"
            "  → Ask a teammate to share this file, or run train.py to generate it."
        )

    if errors:
        print("\n--- Pre-flight Check Failed ---")
        for err in errors:
            print(err)
        sys.exit(1)

    print("[OK] All required files found.")


def check_audio_format(filepath):
    """Validate audio file format and warn about common issues."""
    try:
        info = sf.info(filepath)
        print(f"[OK] Audio file: {info.duration:.1f}s, {info.samplerate}Hz, {info.channels} channel(s)")

        if info.channels > 1:
            print(
                "[WARNING] Audio file has multiple channels (stereo).\n"
                "  → The script will automatically mix to mono, but for best results convert first:\n"
                "  → ffmpeg -i audio.wav -ac 1 audio.wav"
            )

        if info.duration < 1.0:
            print("[WARNING] Audio is very short (under 1 second). Results may be unreliable.")

        if info.duration > 300:
            print("[WARNING] Audio is over 5 minutes. Processing may take a long time on CPU.")

        return True

    except Exception as e:
        print(f"[ERROR] Could not read audio file: {e}")
        print("  → Make sure the file is a valid WAV/audio file and not corrupted.")
        sys.exit(1)


# --- 3. Load ALL Models ---

def load_models():
    print("\nLoading all models (this may take a moment)...")

    try:
        print("  Loading Whisper...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    except Exception as e:
        print(f"[ERROR] Failed to load Whisper model: {e}")
        print("  → Check your internet connection or HuggingFace cache.")
        sys.exit(1)

    try:
        print("  Loading Wav2Vec2...")
        audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    except Exception as e:
        print(f"[ERROR] Failed to load Wav2Vec2 model: {e}")
        print("  → Check your internet connection or HuggingFace cache.")
        sys.exit(1)

    try:
        print("  Loading BERT...")
        text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_model = BertModel.from_pretrained("bert-base-uncased")
    except Exception as e:
        print(f"[ERROR] Failed to load BERT model: {e}")
        print("  → Check your internet connection or HuggingFace cache.")
        sys.exit(1)

    try:
        print("  Loading personality model weights...")
        INPUT_SIZE = 1536  # 768 from text + 768 from audio
        OUTPUT_SIZE = 5
        personality_model = PersonalityModel(INPUT_SIZE, OUTPUT_SIZE)
        personality_model.load_state_dict(
            torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
        )
        personality_model.eval()
    except RuntimeError as e:
        print(f"[ERROR] Model weights don't match the architecture: {e}")
        print("  → The .pth file may have been trained with a different model version.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load personality model: {e}")
        sys.exit(1)

    print("[OK] All models loaded.\n")
    return processor, model, audio_processor, audio_model, text_tokenizer, text_model, personality_model


# --- 4. Main Processing ---

def run_prediction(models):
    processor, whisper_model, audio_processor, audio_model, text_tokenizer, text_model, personality_model = models

    device = "cpu"
    audio_model.to(device)
    text_model.to(device)
    personality_model.to(device)

    # Load audio
    try:
        speech, sample_rate = sf.read(AUDIO_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to read audio file: {e}")
        sys.exit(1)

    # Convert stereo to mono if needed
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)
        print("[INFO] Converted stereo audio to mono.")

    # Resample if needed
    if sample_rate != 16000:
        print(f"[INFO] Resampling audio from {sample_rate}Hz to 16000Hz...")
        speech = librosa.resample(y=speech, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Part A: Transcription
    print("Transcribing audio...")
    try:
        input_features = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_features
        predicted_ids = whisper_model.generate(input_features)
        transcription_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        if not transcription_text.strip():
            print("[WARNING] Transcription is empty — the audio may be silent or too noisy.")
            print("  → Results may be unreliable.")
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        sys.exit(1)

    # Part B: BERT text features
    try:
        text_inputs = text_tokenizer(
            transcription_text, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            text_features = text_model(**text_inputs).last_hidden_state.mean(dim=1)
    except Exception as e:
        print(f"[ERROR] BERT feature extraction failed: {e}")
        sys.exit(1)

    # Part C: Wav2Vec2 audio features
    try:
        audio_inputs = audio_processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_features = audio_model(**audio_inputs).last_hidden_state.mean(dim=1)
    except Exception as e:
        print(f"[ERROR] Audio feature extraction failed: {e}")
        sys.exit(1)

    # Part D: Personality prediction
    try:
        combined_features = torch.cat((text_features, audio_features), dim=1)
        with torch.no_grad():
            predicted_scores = personality_model(combined_features)
        scores = predicted_scores.squeeze().tolist()
    except Exception as e:
        print(f"[ERROR] Personality prediction failed: {e}")
        sys.exit(1)

    # Output
    output_data = {
        "text": transcription_text,
        "traits": {
            "openness":          round(scores[0], 4),
            "conscientiousness": round(scores[1], 4),
            "extraversion":      round(scores[2], 4),
            "agreeableness":     round(scores[3], 4),
            "neuroticism":       round(scores[4], 4)
        }
    }

    print("\n--- Prediction Results ---")
    print(json.dumps(output_data, indent=2))


# --- Entry Point ---
if __name__ == "__main__":
    check_requirements()
    check_audio_format(AUDIO_FILE)
    models = load_models()
    run_prediction(models)
