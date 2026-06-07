# Running on Windows — Setup Guide

This guide documents how to set up and run the project on Windows. The original `requirements.txt` was generated on Linux with GPU support, so some extra steps are needed for Windows CPU-only inference.

---

## Prerequisites

Make sure the following are installed before starting:

- **Python 3.11** — [Download here](https://www.python.org/downloads/release/python-3117/)
  - During install, check **"Add python.exe to PATH"**
  - Python 3.13 is NOT compatible with this project
- **Git** — [Download here](https://git-scm.com/download/win)
- **ffmpeg** — [Download here](https://www.gyan.dev/ffmpeg/builds/)
  - Download `ffmpeg-release-essentials.zip`, unzip it
  - Add the `bin` folder inside to your system PATH

> **Tip:** To add ffmpeg to PATH in Git Bash permanently:
> ```bash
> echo 'export PATH=$PATH:"/c/path/to/ffmpeg/bin"' >> ~/.bashrc
> source ~/.bashrc
> ```

---

## Setup Steps

### 1. Clone the repository
```bash
git clone https://github.com/dangbros/3rd-year-project.git
cd 3rd-year-project
```

### 2. Create a virtual environment with Python 3.11
```bash
py -3.11 -m venv venv
```

### 3. Activate the virtual environment

**Git Bash:**
```bash
source venv/Scripts/activate
```

**Command Prompt:**
```
.\venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### 4. Remove Linux-only packages from requirements

The original `requirements.txt` includes Linux/GPU-only packages that don't exist on Windows. Remove them:

```bash
# Remove audioop-lts (not needed on Python 3.11, built-in)
sed -i '/audioop-lts/d' requirements.txt

# Create a Windows-compatible requirements file (removes nvidia/triton/torch lines)
grep -v -E "^(torch|nvidia|triton)" requirements.txt > requirements_win.txt
```

### 5. Install CPU-only PyTorch

The original requirements include CUDA GPU packages. Install the CPU-only version instead:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu \
  --trusted-host download.pytorch.org \
  --trusted-host download-r2.pytorch.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host pypi.org
```

### 6. Install remaining dependencies

```bash
pip install -r requirements_win.txt \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host pypi.python.org \
  --timeout 120
```

### 7. Install Whisper

```bash
pip install openai-whisper \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --trusted-host pypi.python.org
```

### 8. Get the pre-trained model

Make sure `personality_model_final.pth` is in the project root folder. Try:

```bash
git pull
```

If it's not in the repo (too large for GitHub), ask a teammate to share it.

---

## Running Inference

### 1. Prepare your audio file

- Place your audio file in the project folder
- Rename it to `audio.wav`
- It must be **mono (single channel)**. Convert stereo files using ffmpeg:

```bash
ffmpeg -i your_audio.wav -ac 1 audio.wav
```

### 2. Run the script

Make sure your virtual environment is active, then:

```bash
python transcribe.py
```

The script will print the transcription and predicted OCEAN personality traits as JSON.

---

## Common Issues on Windows

| Error | Cause | Fix |
|---|---|---|
| `audioop-lts` not found | Package only for Python 3.13 | Run `sed -i '/audioop-lts/d' requirements.txt` |
| `nvidia-cufile-cu12` not found | Linux-only CUDA package | Use `requirements_win.txt` instead |
| SSL certificate errors | Corporate/university network proxy | Add `--trusted-host` flags to pip commands |
| `ffmpeg: command not found` | ffmpeg not in PATH | Add ffmpeg `bin` folder to PATH and restart terminal |
| `bash: .\venv\Scripts\activate: command not found` | Wrong shell syntax | Use `source venv/Scripts/activate` in Git Bash |
| Torch/torchvision version mismatch | Mixed torch versions installed | Reinstall with `torch==2.5.1 torchvision==0.20.1` |

---

## Notes

- **CPU-only mode** is perfectly fine for inference. GPU is only needed for training.
- The first run will download ~1.1 GB of model weights (Whisper, wav2vec2, BERT). Subsequent runs load from cache instantly.
- The symlink warnings from HuggingFace are harmless and can be ignored.
- Tested on Windows 11 with Python 3.11.7 and Git Bash.
