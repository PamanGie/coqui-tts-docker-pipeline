# TTS Training with Coqui - Complete Setup Guide

A comprehensive guide for training custom Text-to-Speech models using Coqui TTS with Docker on Windows.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Setup](#docker-setup)
- [Training Environment Setup](#training-environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Training Process](#training-process)
- [Testing Trained Model](#testing-trained-model)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

## Prerequisites

- Windows 10/11 with WSL2 enabled
- NVIDIA GPU with CUDA support (tested with RTX 3060)
- Docker Desktop for Windows
- Minimum 16GB RAM (32GB recommended)
- At least 30GB free storage space
- NVIDIA GPU drivers installed

## Docker Setup

### 1. Install Docker Desktop

1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Run installer with default settings
3. Enable "Use WSL 2 instead of Hyper-V" during installation
4. Restart computer after installation
5. Launch Docker Desktop and wait for "Engine running" status

### 2. Move Docker to E Drive (Optional)

To save space on C drive:

1. Open Docker Desktop → Settings → Resources → Advanced
2. Change "Disk image location" to `E:\docker_new\DockerDesktopWSL`
3. Click "Apply & Restart"
4. Wait for Docker to restart (2-3 minutes)

### 3. Verify Docker Installation

```cmd
docker --version
docker run hello-world
```

## Training Environment Setup

### 1. Create Project Structure

Create the following folder structure:

```
E:\
├── Dockerfile
├── coqui-dataset\
│   ├── wavs\
│   ├── metadata.txt
│   ├── config.json
│   ├── output\
│   └── docker_commands.txt
└── docker_new\DockerDesktopWSL\
```

### 2. Create Dockerfile

Create `E:\Dockerfile` with the following content:

```dockerfile
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jakarta

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==1.13.1 \
    torchaudio==0.13.1 \
    TTS==0.15.6 \
    librosa \
    soundfile

WORKDIR /workspace
CMD ["bash"]
```

### 3. Build Docker Image

```cmd
cd E:\
docker build -t tts-training-fixed .
```

This process downloads dependencies (~3-5GB) and builds the training environment.

### 4. Verify Image Build

```cmd
docker images
```

Should show `tts-training-fixed` in the list.

## Dataset Preparation

### 1. Video to Audio Conversion

Use the provided `dataset.py` script to convert MP4 videos to Coqui-compatible format:

**Input:** MP4 video files in `C:\Users\[username]\input_videos\`
**Output:** Training-ready dataset in `E:\coqui-dataset\`

The script automatically:
- Extracts audio from videos (22050Hz, mono, 16-bit WAV)
- Generates transcripts using OpenAI Whisper
- Creates ljspeech-format metadata.txt (3 columns)
- Generates optimized config.json

### 2. Dataset Structure

After processing, your dataset should look like:

```
E:\coqui-dataset\
├── wavs\
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ... (all audio files)
├── metadata.txt          # Dataset metadata
├── config.json           # Training configuration
└── docker_commands.txt   # Ready-to-use commands
```

### 3. Metadata Format

The `metadata.txt` follows ljspeech format (3 columns):

```
filename|raw_text|normalized_text
audio_001|Hello how are you today|Hello how are you today
audio_002|This is a test sentence|This is a test sentence
```

### 4. Audio Requirements

- **Format:** WAV (PCM 16-bit)
- **Sample Rate:** 22050 Hz
- **Channels:** Mono
- **Duration:** 2-15 seconds per clip optimal
- **Quality:** Clean audio without background noise
- **Total Duration:** 10-30 minutes recommended for good results

## Training Configuration

### Config Parameters

The `config.json` contains training parameters:

```json
{
  "model": "tacotron2",
  "run_name": "custom_voice_coqui_training",
  "epochs": 1500,
  "batch_size": 6,
  "eval_batch_size": 3,
  "mixed_precision": false,
  "run_eval": true,
  "test_delay_epochs": 10,
  "print_eval": false,
  "print_step": 25,
  "save_step": 200,
  "plot_step": 50,
  "log_model_step": 400,
  "save_n_checkpoints": 3,
  "save_best_after": 400,
  "target_loss": "loss_1",
  "lr": 0.001,
  "wd": 0.000001,
  "warmup_steps": 4000,
  "seq_len_norm": false,
  "loss_masking": true,
  "datasets": [{
    "name": "custom_voice",
    "path": "/workspace/",
    "meta_file_train": "metadata.txt",
    "meta_file_val": "metadata.txt",
    "formatter": "ljspeech"
  }],
  "audio": {
    "sample_rate": 22050,
    "hop_length": 256,
    "win_length": 1024,
    "fft_size": 1024,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
    "num_mels": 80
  },
  "output_path": "/workspace/output/",
  "use_phonemes": false,
  "phoneme_language": "en-us",
  "compute_input_seq_cache": true,
  "precompute_num_workers": 2,
  "start_by_longest": true
}
```

### Parameter Optimization

For better performance:

1. **batch_size:** Reduce from 6 to 3-4 for RTX 3060
2. **epochs:** Adjust based on dataset size (1000-2000)
3. **save_step:** Lower values for more frequent saves
4. **lr:** Learning rate (0.001 is optimal for most cases)

## Training Process

### 1. Prepare Output Directory

```cmd
mkdir E:\coqui-dataset\output
```

### 2. Start Training

```cmd
docker run --gpus all --rm -it -v E:\coqui-dataset:/workspace tts-training-fixed python3 -m TTS.bin.train_tts --config_path config.json
```

### 3. Training Monitoring

During training, you'll see:
- Loss values every 25 steps
- Model checkpoints saved every 200 steps
- Validation results every 10 epochs
- Character vocabulary warnings (normal)

**Expected Training Time:** 4-8 hours depending on dataset size and GPU performance.

### 4. Training Output

Training generates these files in `E:\coqui-dataset\output\`:
- `best_model.pth.tar` - Best performing model
- `config.json` - Model configuration
- `events.out.tfevents.*` - Training logs for TensorBoard
- `checkpoint_*.pth.tar` - Periodic checkpoints

### 5. Stop/Resume Training

**To stop training:** Ctrl+C or close terminal
**To resume:** Re-run the same training command (starts from scratch)

Note: Coqui TTS doesn't support resume from checkpoint in this setup.

## Testing Trained Model

### 1. Quick Synthesis Test

```cmd
docker run --gpus all --rm -v E:\coqui-dataset:/workspace tts-training-fixed tts --model_path /workspace/output/best_model.pth.tar --config_path /workspace/output/config.json --text "Hello, this is my custom trained voice" --out_path /workspace/output/test_result.wav
```

### 2. Interactive TTS Session

```cmd
docker run --gpus all --rm -it -v E:\coqui-dataset:/workspace tts-training-fixed bash
```

Inside the container:
```bash
tts --model_path /workspace/output/best_model.pth.tar \
    --config_path /workspace/output/config.json \
    --text "Your custom text here" \
    --out_path custom_synthesis.wav
```

### 3. Batch Processing

Create multiple audio files:
```bash
for text in "Hello world" "How are you" "This is a test"; do
    tts --model_path /workspace/output/best_model.pth.tar \
        --config_path /workspace/output/config.json \
        --text "$text" \
        --out_path "${text// /_}.wav"
done
```

## Troubleshooting

### Common Issues

**"Character not found in vocabulary" warnings**
- This is normal during training
- Model is adapting to new characters/language
- Training continues normally

**System becomes laggy during training**
- Reduce `batch_size` in config.json (6 → 3 or 4)
- Close unnecessary applications
- Train during off-peak hours
- Monitor GPU temperature

**Training crashes/stops**
- Check available disk space (>10GB recommended)
- Verify GPU memory not exceeded
- Restart with same command (no data loss)

**Docker build failures**
- Ensure stable internet connection
- Retry build command
- Check Docker Desktop is running
- Clear Docker cache: `docker builder prune`

**Audio quality issues**
- Verify input audio is clean (no noise/distortion)
- Check transcript accuracy
- Ensure consistent audio levels

### Error Resolution

**ImportError or module conflicts:**
```cmd
docker rmi tts-training-fixed
docker build -t tts-training-fixed .
```

**Out of memory errors:**
- Reduce batch_size in config.json
- Use smaller audio clips
- Close other GPU applications

**Permission errors:**
- Run Command Prompt as Administrator
- Check folder permissions on E drive

## Performance Optimization

### Hardware Optimization

1. **GPU Monitoring:**
   - Use MSI Afterburner or GPU-Z to monitor temperature
   - Ensure adequate cooling (keep below 80°C)
   - Check power supply adequacy

2. **Storage Optimization:**
   - Use SSD for dataset storage
   - Keep at least 15GB free space during training
   - Regular cleanup of old checkpoints

3. **Memory Management:**
   - Close browser tabs and heavy applications
   - Set Windows power plan to "High Performance"
   - Disable Windows memory compression if needed

### Training Optimization

1. **Dataset Quality:**
   - Clean audio without background noise
   - Accurate transcriptions
   - Consistent audio levels across samples
   - Optimal total duration: 15-45 minutes

2. **Parameter Tuning:**
   - Start with provided config
   - Adjust batch_size based on GPU memory
   - Monitor loss curves for overfitting
   - Use early stopping if loss plateaus

3. **Workflow Optimization:**
   - Train overnight or during non-use hours
   - Save intermediate models frequently
   - Keep backup copies of best models

## File Structure Reference

Final project structure:

```
E:\
├── Dockerfile                    # Docker build configuration
├── coqui-dataset\
│   ├── wavs\                    # Audio files (22050Hz, mono)
│   │   ├── audio_001.wav
│   │   ├── audio_002.wav
│   │   └── ...
│   ├── metadata.txt             # Dataset metadata (ljspeech format)
│   ├── config.json              # Training configuration
│   ├── output\                  # Training results
│   │   ├── best_model.pth.tar   # Trained TTS model
│   │   ├── config.json          # Model config
│   │   ├── checkpoint_*.pth.tar # Training checkpoints
│   │   └── events.out.tfevents.* # Training logs
│   └── docker_commands.txt      # Command reference
└── docker_new\DockerDesktopWSL\ # Docker storage
```

## Technical Specifications

### Software Dependencies

**Docker Image includes:**
- Python 3.9
- PyTorch 1.13.1
- torchaudio 0.13.1
- Coqui TTS 0.15.6
- librosa (audio processing)
- soundfile (audio I/O)
- ffmpeg (multimedia framework)
- espeak-ng (phoneme generation)

**Host Requirements:**
- Docker Desktop 4.0+
- NVIDIA GPU Drivers 450.80.02+
- Windows 10 version 2004+ with WSL2

### Model Architecture

- **Base Model:** Tacotron2
- **Vocoder:** Griffin-Lim (default) or Neural Vocoder
- **Input:** Text sequences
- **Output:** Mel-spectrograms → Audio waveforms
- **Sample Rate:** 22050 Hz
- **Audio Format:** 16-bit PCM WAV

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch
3. Test thoroughly on different datasets
4. Submit pull request with detailed description

## License

This project uses:
- Coqui TTS (Mozilla Public License 2.0)
- OpenAI Whisper (MIT License)
- Docker (Apache License 2.0)

## Credits

- [Coqui TTS](https://github.com/coqui-ai/TTS) - TTS training framework
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech-to-text transcription
- [Docker](https://www.docker.com/) - Containerization platform
- [PyTorch](https://pytorch.org/) - Deep learning framework

## Support

For issues and questions:
1. Check troubleshooting section first
2. Search existing GitHub issues
3. Create new issue with detailed description
4. Include system specs and error logs

---

**Happy TTS Training!** 🎤→🤖→🔊


# TTS Training with Coqui - Petunjuk Lengkap Bahasa Indonesia

Complete guide untuk training Text-to-Speech model menggunakan Coqui TTS dengan Docker di Windows.

## Prerequisites

- Windows 10/11
- NVIDIA GPU dengan CUDA support (tested dengan RTX 3060)
- Docker Desktop
- RAM minimal 16GB (recommended 32GB)
- Storage kosong minimal 30GB

## Setup Docker

### 1. Install Docker Desktop

1. Download Docker Desktop dari https://www.docker.com/products/docker-desktop/
2. Install dengan default settings
3. Enable "Use WSL 2 instead of Hyper-V" saat install
4. Restart komputer setelah instalasi

### 2. Pindah Docker ke Drive E (Opsional)

Untuk menghemat space drive C:

1. Buka Docker Desktop → Settings → Resources → Advanced
2. Ganti "Disk image location" ke `E:\docker_new\DockerDesktopWSL`
3. Click "Apply & Restart"
4. Tunggu Docker restart

## Setup Training Environment

### 1. Struktur Folder

Buat struktur folder seperti ini:

```
E:\
├── Dockerfile
├── coqui-dataset\
│   ├── wavs\
│   ├── metadata.txt
│   ├── config.json
│   ├── output\
│   └── docker_commands.txt
└── docker_new\DockerDesktopWSL\
```

### 2. Buat Dockerfile

Buat file `E:\Dockerfile` dengan content:

```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==1.13.1 \
    torchaudio==0.13.1 \
    TTS==0.15.6 \
    librosa \
    soundfile

WORKDIR /workspace
CMD ["bash"]
```

### 3. Build Docker Image

```cmd
cd E:\
docker build -t tts-training-fixed .
```

Proses ini akan download dependencies (~3-5GB) dan build image training.

## Persiapan Dataset

### 1. Extract Video ke Audio + Transcript

Gunakan script `dataset.py` untuk convert video MP4 ke format Coqui:

- Input: Video MP4 files di `C:\Users\[username]\input_videos\`
- Output: Dataset siap training di `E:\coqui-dataset\`

Script akan:
- Extract audio dari video (format: 22050Hz, mono, 16-bit WAV)
- Generate transcript menggunakan Whisper
- Buat metadata.txt format ljspeech (3 kolom)
- Generate config.json optimal

### 2. Format Dataset

**Struktur folder hasil:**
```
E:\coqui-dataset\
├── wavs\
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── metadata.txt
├── config.json
└── docker_commands.txt
```

**Format metadata.txt (ljspeech):**
```
filename|raw_text|normalized_text
audio1|Halo apa kabar semuanya|Halo apa kabar semuanya
audio2|Selamat pagi Indonesia|Selamat pagi Indonesia
```

## Training Configuration

### Config Parameters

File `config.json` berisi parameter training:

```json
{
  "model": "tacotron2",
  "run_name": "custom_voice_coqui_training",
  "epochs": 1500,
  "batch_size": 6,
  "eval_batch_size": 3,
  "datasets": [{
    "name": "custom_voice",
    "path": "/workspace/",
    "meta_file_train": "metadata.txt",
    "meta_file_val": "metadata.txt",
    "formatter": "ljspeech"
  }],
  "audio": {
    "sample_rate": 22050,
    "hop_length": 256,
    "win_length": 1024,
    "fft_size": 1024,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
    "num_mels": 80
  },
  "output_path": "/workspace/output/"
}
```

### Optimasi Performance

Untuk mengurangi lag saat training:

1. **Kurangi batch_size** dari 6 ke 3 atau 4
2. **Tutup aplikasi lain** yang memory-intensive
3. **Monitor GPU temperature** untuk mencegah throttling

## Training Process

### 1. Persiapan

Buat folder output:
```cmd
mkdir E:\coqui-dataset\output
```

### 2. Start Training

```cmd
docker run --gpus all --rm -it \
  -v E:\coqui-dataset:/workspace \
  tts-training-fixed \
  python3 -m TTS.bin.train_tts --config_path config.json
```

### 3. Monitoring Training

Training akan menampilkan:
- Loss values setiap 25 steps
- Model checkpoints setiap 200 steps
- Validation results setiap 10 epochs

**Durasi training:** 4-8 jam tergantung dataset size dan GPU.

### 4. Training Output

File hasil training di `E:\coqui-dataset\output\`:
- `best_model.pth.tar` - Model terbaik
- `config.json` - Konfigurasi model
- `events.out.tfevents.*` - Training logs

## Testing Trained Model

### Synthesis Test

```cmd
docker run --gpus all --rm \
  -v E:\coqui-dataset:/workspace \
  tts-training-fixed \
  tts --model_path /workspace/output/best_model.pth.tar \
      --config_path /workspace/output/config.json \
      --text "Halo, ini hasil training TTS dengan suara saya sendiri" \
      --out_path /workspace/output/test_result.wav
```

### Interactive TTS

```cmd
docker run --gpus all --rm -it \
  -v E:\coqui-dataset:/workspace \
  tts-training-fixed \
  bash
```

Di dalam container:
```bash
tts --model_path /workspace/output/best_model.pth.tar \
    --config_path /workspace/output/config.json \
    --text "Text yang ingin disintetis" \
    --out_path custom_text.wav
```

## Troubleshooting

### Common Issues

**Error: "Character not found in vocabulary"**
- Normal warning, model sedang adapt dengan vocab baru
- Training tetap berjalan normal

**PC menjadi lag saat training**
- Kurangi batch_size di config.json
- Tutup aplikasi lain
- Training di waktu PC tidak digunakan

**Training terhenti/crash**
- Model dan dataset tetap aman
- Restart training dengan command yang sama
- Training mulai dari awal (tidak ada resume)

**Image build error**
- Pastikan koneksi internet stabil
- Retry build command
- Check Docker Desktop running

### Performance Tips

1. **Dataset Quality:**
   - Audio bersih tanpa noise
   - Transcript akurat
   - Durasi optimal 10-30 menit total

2. **Hardware Optimization:**
   - Monitor GPU temperature
   - Pastikan cooling adequate
   - Use SSD untuk dataset storage

3. **Training Parameters:**
   - Batch_size: 3-6 untuk RTX 3060
   - Epochs: 1000-2000 tergantung dataset
   - Learning rate: 0.001 (default)

## File Structure Summary

```
E:\
├── Dockerfile                 # Docker build file
├── coqui-dataset\
│   ├── wavs\                 # Audio files (22050Hz, mono)
│   │   ├── audio1.wav
│   │   └── ...
│   ├── metadata.txt          # Dataset metadata (ljspeech format)
│   ├── config.json           # Training configuration
│   ├── output\               # Training results
│   │   ├── best_model.pth.tar
│   │   ├── config.json
│   │   └── events.out.tfevents.*
│   └── docker_commands.txt   # Ready-to-use commands
└── docker_new\DockerDesktopWSL\  # Docker data storage
```

## Dependencies

**Docker Image includes:**
- Python 3.9
- PyTorch 1.13.1
- torchaudio 0.13.1
- TTS 0.15.6
- librosa
- soundfile
- ffmpeg
- espeak-ng

**Host Requirements:**
- Docker Desktop
- NVIDIA GPU Drivers
- CUDA compatible GPU

---

## Credits

- [Coqui TTS](https://github.com/coqui-ai/TTS) - TTS training framework
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech-to-text transcription
- [Docker](https://www.docker.com/) - Containerization platform
