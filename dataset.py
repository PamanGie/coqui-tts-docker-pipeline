import os
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
from pathlib import Path
import numpy as np
from tqdm import tqdm
import whisper
import json

def extract_and_prepare_coqui_dataset(input_video_folder, output_dataset_folder):
    """
    Extract audio dari video MP4 dan prepare dataset untuk Coqui TTS dengan format ljspeech
    """
    
    print("Video to Coqui TTS Dataset Converter")
    print("=" * 50)
    
    # Setup paths
    input_path = Path(input_video_folder)
    output_path = Path(output_dataset_folder)
    wavs_folder = output_path / "wavs"
    
    # Create output folders
    output_path.mkdir(parents=True, exist_ok=True)
    wavs_folder.mkdir(exist_ok=True)
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    print(f"Scanning video folder: {input_video_folder}")
    
    for file in os.listdir(input_video_folder):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if not video_files:
        print("No video files found!")
        return None
    
    print(f"Found {len(video_files)} video files:")
    for video in video_files:
        print(f"   - {video}")
    print("-" * 50)
    
    # Load Whisper for transcription
    print("Loading Whisper model for transcription...")
    whisper_model = whisper.load_model("base")
    print("Whisper loaded!")
    print("-" * 50)
    
    metadata_entries = []
    processed_count = 0
    
    # Process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            input_video_path = input_path / video_file
            base_name = Path(video_file).stem
            
            # Clean filename for Coqui format
            clean_name = base_name.replace(' ', '_').replace('-', '_')
            audio_filename = f"{clean_name}.wav"
            output_audio_path = wavs_folder / audio_filename
            
            tqdm.write(f"Processing: {video_file}")
            
            # Extract audio from video
            tqdm.write("   Extracting audio from video...")
            video_clip = VideoFileClip(str(input_video_path))
            
            if video_clip.audio is None:
                tqdm.write("   No audio track found, skipping...")
                video_clip.close()
                continue
            
            # Extract to temporary file
            temp_audio = output_path / f"{clean_name}_temp.wav"
            video_clip.audio.write_audiofile(str(temp_audio), verbose=False, logger=None)
            video_clip.close()
            
            # Process audio for Coqui format
            tqdm.write("   Converting to Coqui format...")
            
            # Load and process audio
            audio, sr = librosa.load(str(temp_audio), sr=22050, mono=True)
            
            # Clean audio
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            audio_normalized = librosa.util.normalize(audio_trimmed)
            
            # Save in Coqui format (22050Hz, mono, 16-bit)
            sf.write(str(output_audio_path), audio_normalized, 22050, subtype='PCM_16')
            
            # Remove temp file
            os.remove(temp_audio)
            
            # Generate transcript with Whisper
            tqdm.write("   Generating transcript...")
            result = whisper_model.transcribe(str(output_audio_path), language="id")
            transcript = result["text"].strip()
            
            # Clean transcript for Coqui format
            clean_transcript = transcript.replace('|', ' ').replace('\n', ' ').replace('\r', ' ')
            clean_transcript = ' '.join(clean_transcript.split())  # Remove extra spaces
            
            # Calculate duration
            duration = len(audio_normalized) / 22050
            
            # Add to metadata in ljspeech format (filename|raw_text|normalized_text)
            audio_name_no_ext = audio_filename.replace('.wav', '')
            metadata_entries.append(f"{audio_name_no_ext}|{clean_transcript}|{clean_transcript}")
            
            processed_count += 1
            
            tqdm.write(f"   Success! Duration: {duration:.1f}s")
            tqdm.write(f"   Transcript: {clean_transcript[:80]}...")
            tqdm.write(f"   Saved: {audio_filename}")
            
        except Exception as e:
            tqdm.write(f"   Error processing {video_file}: {str(e)}")
    
    if processed_count == 0:
        print("No videos were processed successfully!")
        return None
    
    # Write metadata.txt for Coqui (ljspeech format)
    metadata_file = output_path / "metadata.txt"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(entry + '\n')
    
    print("-" * 50)
    print(f"Coqui dataset preparation completed!")
    print(f"Output folder: {output_dataset_folder}")
    print(f"Audio files: {processed_count} files in wavs/ folder")
    print(f"Metadata file: metadata.txt (ljspeech format)")
    
    # Calculate total duration
    total_duration = 0
    for audio_file in wavs_folder.glob("*.wav"):
        audio, sr = librosa.load(str(audio_file), sr=None)
        total_duration += len(audio) / sr
    
    print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    # Generate Coqui config
    generate_coqui_config(output_dataset_folder, total_duration, processed_count)
    
    # Generate Docker commands
    create_docker_commands(output_dataset_folder)
    
    return output_dataset_folder

def generate_coqui_config(dataset_path, total_duration, num_samples):
    """Generate config.json untuk Coqui TTS training dengan ljspeech formatter"""
    
    config = {
        "model": "tacotron2",
        "run_name": "custom_voice_coqui_training",
        "epochs": 1500,
        "batch_size": 6,
        "eval_batch_size": 3,
        "mixed_precision": False,
        "run_eval": True,
        "test_delay_epochs": 10,
        "print_eval": False,
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
        "seq_len_norm": False,
        "loss_masking": True,
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
        "use_phonemes": False,
        "phoneme_language": "en-us",
        "compute_input_seq_cache": True,
        "precompute_num_workers": 2,
        "start_by_longest": True
    }
    
    config_file = Path(dataset_path) / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Generated config.json with ljspeech formatter")
    print(f"   - Epochs: {config['epochs']}")
    print(f"   - Batch size: {config['batch_size']}")
    print(f"   - Format: ljspeech (3 columns)")

def create_docker_commands(dataset_path):
    """Generate Docker commands untuk training"""
    
    dataset_abs_path = os.path.abspath(dataset_path).replace('\\', '/')
    
    commands = f'''# Coqui TTS Docker Training Commands
# Dataset: {dataset_abs_path}

# 1. Create output folder:
mkdir {dataset_abs_path}/output

# 2. Start training:
docker run --gpus all --rm -it \\
  -v "{dataset_abs_path}:/workspace" \\
  tts-training \\
  python3 -m TTS.bin.train_tts --config_path config.json

# 3. Test trained model:
docker run --gpus all --rm \\
  -v "{dataset_abs_path}:/workspace" \\
  tts-training \\
  tts --model_path /workspace/output/best_model.pth.tar \\
      --config_path /workspace/output/config.json \\
      --text "Halo, ini hasil training TTS dengan suara saya sendiri" \\
      --out_path /workspace/output/test_result.wav
'''
    
    commands_file = Path(dataset_path) / "docker_commands.txt"
    with open(commands_file, 'w', encoding='utf-8') as f:
        f.write(commands)
    
    print(f"Docker commands saved to: docker_commands.txt")

# Main execution
if __name__ == "__main__":
    
    print("Video to Coqui TTS Dataset Converter")
    print("=" * 50)
    
    # Configuration
    INPUT_VIDEO_FOLDER = "C:/Users/phantom/tts_audio/input_videos"  # Folder berisi MP4 files
    OUTPUT_DATASET_FOLDER = "E:/coqui-dataset"           # Output Coqui dataset
    
    print(f"Input video folder: {INPUT_VIDEO_FOLDER}")
    print(f"Output dataset folder: {OUTPUT_DATASET_FOLDER}")
    print("-" * 50)
    
    # Check input folder exists
    if not os.path.exists(INPUT_VIDEO_FOLDER):
        print(f"Input folder tidak ditemukan: {INPUT_VIDEO_FOLDER}")
        print("Buat folder tersebut dan masukkan video MP4 Anda")
        exit(1)
    
    # Process videos to Coqui dataset
    try:
        result = extract_and_prepare_coqui_dataset(INPUT_VIDEO_FOLDER, OUTPUT_DATASET_FOLDER)
        
        if result:
            print("\nDataset conversion completed successfully!")
            print(f"\nFinal dataset structure:")
            print(f"{OUTPUT_DATASET_FOLDER}/")
            print(f"├── wavs/                    (audio files)")
            print(f"├── metadata.txt             (ljspeech format: file|text|text)")
            print(f"├── config.json              (training config)")
            print(f"└── docker_commands.txt      (ready commands)")
            
            print(f"\nNext steps:")
            print(f"1. Review files in {OUTPUT_DATASET_FOLDER}")
            print(f"2. Create output folder: mkdir {OUTPUT_DATASET_FOLDER}/output")
            print(f"3. Run training command dari docker_commands.txt")
            print(f"4. Training akan jalan 4-8 jam")
            print(f"5. Model hasil ada di {OUTPUT_DATASET_FOLDER}/output/")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Pastikan folder input berisi video MP4 dan dependencies terinstall")