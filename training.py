import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
from jiwer import wer
import time
from odf.opendocument import load
from odf.text import P

# Load your fine-tuned model
model_path = "./whisper-konkani-final"
print("Loading fine-tuned Whisper model...")

try:
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure the model path is correct and the model was saved properly")
    exit()

def read_odt(file_path):
    """
    Read ODT file content - same function from training
    """
    try:
        print(f"Reading ODT file: {file_path}")
        doc = load(file_path)
        paragraphs = []
        for p in doc.getElementsByType(P):
            text_parts = []
            for node in p.childNodes:
                if hasattr(node, "data"):
                    text_parts.append(node.data)
                elif hasattr(node, "childNodes"):
                    for subnode in node.childNodes:
                        if hasattr(subnode, "data"):
                            text_parts.append(subnode.data)
            if text_parts:
                paragraphs.append("".join(text_parts))
        text = " ".join(paragraphs).strip()
        print(f"Successfully read ODT, text length: {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def transcribe_audio_chunked(audio_path, model, processor, device="cpu", chunk_duration=30):
    """
    Transcribe long audio by chunking it into smaller pieces to avoid repetition
    """
    try:
        print(f"Processing: {audio_path}")
        audio_array, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        
        total_duration = len(audio_array) / sample_rate
        print(f"Audio duration: {total_duration:.2f} seconds")
        
        # If audio is short, transcribe directly
        if total_duration <= chunk_duration:
            return transcribe_single_chunk(audio_array, model, processor, device)
        
        # Chunk the audio
        chunk_size = int(chunk_duration * sample_rate)  # samples per chunk (ensure integer)
        transcriptions = []
        
        num_chunks = int(np.ceil(len(audio_array) / chunk_size))
        print(f"Splitting into {num_chunks} chunks of ~{chunk_duration} seconds each")
        
        for i in range(num_chunks):
            start_idx = int(i * chunk_size)
            end_idx = int(min((i + 1) * chunk_size, len(audio_array)))
            chunk = audio_array[start_idx:end_idx]
            
            print(f"Processing chunk {i+1}/{num_chunks} ({len(chunk)/sample_rate:.1f}s)")
            
            chunk_transcription = transcribe_single_chunk(chunk, model, processor, device)
            if chunk_transcription and chunk_transcription.strip():
                transcriptions.append(chunk_transcription.strip())
        
        # Combine all transcriptions
        full_transcription = " ".join(transcriptions)
        return full_transcription
        
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None

def transcribe_single_chunk(audio_chunk, model, processor, device="cpu"):
    """
    Transcribe a single audio chunk with improved generation parameters
    """
    try:
        # Process with Whisper processor
        inputs = processor(audio_chunk, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        # Generate transcription with better parameters to avoid repetition
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                repetition_penalty=1.2,   # Penalize repetition
                length_penalty=1.0,
                do_sample=False
            )
        
        # Decode the transcription
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
        
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

def compare_with_ground_truth(predicted_text, ground_truth_text):
    """
    Compare predicted transcription with ground truth and calculate WER
    """
    if not predicted_text or not ground_truth_text:
        return None
    
    # Calculate Word Error Rate
    error_rate = wer(ground_truth_text, predicted_text)
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Ground Truth Length: {len(ground_truth_text)} characters")
    print(f"Predicted Length: {len(predicted_text)} characters")
    print(f"\nGround Truth (first 200 chars): {ground_truth_text[:200]}...")
    print(f"Predicted (first 200 chars):    {predicted_text[:200]}...")
    print(f"\nWord Error Rate (WER): {error_rate:.4f} ({error_rate*100:.2f}%)")
    
    return error_rate

def test_single_file():
    """
    Test the model on a single audio file with chunking support
    """
    print("\n=== SINGLE FILE TEST ===")
    
    # You can specify a test audio file here
    test_audio_path = input("Enter path to test audio file (.wav): ").strip()
    
    if not os.path.exists(test_audio_path):
        print(f"File not found: {test_audio_path}")
        return
    
    # Ask for chunk duration for long files
    chunk_duration = input("Enter chunk duration in seconds for long files (default 30): ").strip()
    try:
        chunk_duration = float(chunk_duration) if chunk_duration else 30.0
    except:
        chunk_duration = 30.0
    
    print(f"Using chunk duration: {chunk_duration} seconds")
    
    # Transcribe with chunking
    print("\n--- Starting Transcription ---")
    start_time = time.time()
    transcription = transcribe_audio_chunked(test_audio_path, model, processor, device, chunk_duration)
    end_time = time.time()
    
    if transcription:
        print(f"\n=== TRANSCRIPTION RESULT ===")
        print(f"File: {os.path.basename(test_audio_path)}")
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Transcription length: {len(transcription)} characters")
        print(f"\nFull Transcription:\n{transcription}")
        
        # Optional: Compare with ground truth if available
        ground_truth_path = input("\nEnter path to ground truth transcript (.odt or .txt) (or press Enter to skip): ").strip()
        
        if ground_truth_path and os.path.exists(ground_truth_path):
            try:
                if ground_truth_path.endswith('.odt'):
                    ground_truth = read_odt(ground_truth_path)
                else:
                    with open(ground_truth_path, 'r', encoding='utf-8') as f:
                        ground_truth = f.read().strip()
                
                if ground_truth:
                    compare_with_ground_truth(transcription, ground_truth)
                else:
                    print("Could not read ground truth content")
                    
            except Exception as e:
                print(f"Error processing ground truth file: {e}")

def test_with_original_training_pair():
    """
    Test with one of the original training audio-transcript pairs
    """
    print("\n=== TEST WITH TRAINING DATA ===")
    
    # Paths to original data
    audio_path = "/content/drive/MyDrive/Anju Project (1)/Audio Prudent media (1)/August 2017 (1)/dataset/Audio"
    transcript_path = "/content/drive/MyDrive/Anju Project (1)/Audio Prudent media (1)/August 2017 (1)/dataset/Transcript"
    
    if not os.path.exists(audio_path) or not os.path.exists(transcript_path):
        print("Original training data directories not found")
        return
    
    # Get available files
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    
    if not audio_files:
        print("No audio files found in training directory")
        return
    
    print("Available audio files:")
    for i, file in enumerate(audio_files[:10]):  # Show first 10
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input(f"\nChoose a file (1-{min(10, len(audio_files))}): ")) - 1
        if choice < 0 or choice >= len(audio_files):
            print("Invalid choice")
            return
            
        selected_audio = audio_files[choice]
        audio_file_path = os.path.join(audio_path, selected_audio)
        
        # Try to find matching transcript using the same logic from training
        # Extract date from audio filename
        import re
        date_match = re.search(r'_(\d{6})\.wav', selected_audio)
        if date_match:
            date_str = date_match.group(1)
            day = int(date_str[:2])
            
            # Look for matching transcript
            transcript_file = None
            for f in os.listdir(transcript_path):
                if f.endswith('.odt') and f.startswith(f"{day} AUG"):
                    transcript_file = f
                    break
            
            if transcript_file:
                transcript_file_path = os.path.join(transcript_path, transcript_file)
                print(f"\nFound matching pair:")
                print(f"Audio: {selected_audio}")
                print(f"Transcript: {transcript_file}")
                
                # Transcribe
                print("\n--- Transcribing ---")
                transcription = transcribe_audio_chunked(audio_file_path, model, processor, device, 30)
                
                if transcription:
                    # Read ground truth
                    ground_truth = read_odt(transcript_file_path)
                    
                    if ground_truth:
                        print(f"\n=== RESULTS ===")
                        print(f"Predicted: {transcription[:300]}...")
                        print(f"Ground Truth: {ground_truth[:300]}...")
                        compare_with_ground_truth(transcription, ground_truth)
                    else:
                        print("Could not read ground truth transcript")
            else:
                print(f"No matching transcript found for {selected_audio}")
        else:
            print(f"Could not extract date from filename: {selected_audio}")
            
    except ValueError:
        print("Invalid input")
    except Exception as e:
        print(f"Error: {e}")

# Main menu
def main():
    print("="*50)
    print("WHISPER KONKANI MODEL TESTING (FIXED)")
    print("="*50)
    
    while True:
        print("\nChoose a testing option:")
        print("1. Test single audio file (with chunking)")
        print("2. Test with original training data pair")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            test_single_file()
        elif choice == "2":
            test_with_original_training_pair()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    main()
