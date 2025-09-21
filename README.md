# TTS Training with Coqui - Complete Setup Guide

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
