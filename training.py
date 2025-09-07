# Install dependencies
!pip install git+https://github.com/openai/whisper.git
!pip install transformers datasets jiwer accelerate odfpy
!sudo apt update && sudo apt install -y ffmpeg

import os
import pandas as pd
from odf.opendocument import load
from odf.text import P
import librosa
import torchaudio
import numpy as np
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from jiwer import wer
import logging
import math
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to your audio and transcript folders
audio_path = "/content/drive/MyDrive/Anju Project (1)/Audio Prudent media (1)/August 2017 (1)"
transcript_path = "/content/drive/MyDrive/Anju Project (1)/Audio Prudent media (1)/August 2017 (1)"

# Custom function to match audio and transcript files
def create_file_mapping(audio_path, transcript_path):
    """
    Create a mapping between audio files and transcript files based on date patterns
    """
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    transcript_files = [f for f in os.listdir(transcript_path) if f.endswith('.odt')]
    
    print(f"DEBUG: Found {len(audio_files)} audio files")
    print(f"DEBUG: Found {len(transcript_files)} transcript files")
    
    # Function to extract date from audio filename
    def extract_date_from_audio(filename):
        # Pattern for "Konkani Prime News_070817.wav" -> extract "070817"
        match = re.search(r'_(\d{6})\.wav', filename)
        if match:
            date_str = match.group(1)  # "070817"
            # Convert to day format: "070817" -> "7" (remove leading zero from day)
            day = int(date_str[:2])  # "07" -> 7
            return str(day)
        return None
    
    # Function to extract date from transcript filename  
    def extract_date_from_transcript(filename):
        # Pattern for "7 AUG PRIME.odt" -> extract "7"
        match = re.search(r'^(\d+)\s+AUG', filename)
        if match:
            return match.group(1)  # "7"
        return None
    
    # Create mapping
    file_pairs = []
    for audio_file in audio_files:
        audio_date = extract_date_from_audio(audio_file)
        if not audio_date:
            print(f"WARNING: Could not extract date from audio file: {audio_file}")
            continue
            
        # Find matching transcript
        matching_transcript = None
        for transcript_file in transcript_files:
            transcript_date = extract_date_from_transcript(transcript_file)
            if transcript_date and transcript_date == audio_date:
                matching_transcript = transcript_file
                break
        
        if matching_transcript:
            file_pairs.append((audio_file, matching_transcript))
            print(f"MATCHED: {audio_file} <-> {matching_transcript}")
        else:
            print(f"NO MATCH: {audio_file} (date: {audio_date})")
    
    print(f"\nSUCCESSFULLY MATCHED: {len(file_pairs)} pairs")
    return file_pairs

# DEBUG: Check if paths exist
print(f"DEBUG: Audio path exists: {os.path.exists(audio_path)}")
print(f"DEBUG: Transcript path exists: {os.path.exists(transcript_path)}")

if os.path.exists(audio_path):
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    print(f"DEBUG: Found {len(audio_files)} .wav files in audio directory")
    print(f"DEBUG: First few audio files: {audio_files[:5]}")
else:
    print("DEBUG: Audio path does not exist!")
    # List what's actually in the parent directory
    parent_dir = os.path.dirname(audio_path)
    if os.path.exists(parent_dir):
        print(f"DEBUG: Contents of parent directory {parent_dir}:")
        print(os.listdir(parent_dir))

if os.path.exists(transcript_path):
    transcript_files = [f for f in os.listdir(transcript_path) if f.endswith('.odt')]
    print(f"DEBUG: Found {len(transcript_files)} .odt files in transcript directory")
    print(f"DEBUG: First few transcript files: {transcript_files[:5]}")
else:
    print("DEBUG: Transcript path does not exist!")
    # List what's actually in the parent directory
    parent_dir = os.path.dirname(transcript_path)
    if os.path.exists(parent_dir):
        print(f"DEBUG: Contents of parent directory {parent_dir}:")
        print(os.listdir(parent_dir))

# Function to read ODT transcript with error handling
def read_odt(file_path):
    try:
        print(f"DEBUG: Attempting to read ODT file: {file_path}")
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
        print(f"DEBUG: Successfully read ODT, text length: {len(text)} characters")
        if len(text) > 0:
            print(f"DEBUG: First 100 characters: {text[:100]}")
        return text
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        print(f"DEBUG: Full error details: {str(e)}")
        return None

# Function to load and preprocess audio using librosa
def load_and_preprocess_audio(audio_file, target_sr=16000):
    try:
        print(f"DEBUG: Attempting to load audio file: {audio_file}")
        logger.info(f"Loading audio file: {audio_file}")
        audio_array, sample_rate = librosa.load(audio_file, sr=target_sr, mono=True)
        audio_array = audio_array.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        print(f"DEBUG: Successfully loaded audio, shape: {audio_array.shape}, duration: {len(audio_array)/sample_rate:.2f}s")
        logger.info(f"Loaded and preprocessed {audio_file}: shape={audio_array.shape}, sr={sample_rate}")
        return audio_array, sample_rate

    except Exception as e:
        logger.error(f"Error loading audio {audio_file}: {e}")
        print(f"DEBUG: Audio loading error details: {str(e)}")
        return None, None

# Function to chunk text into smaller pieces
def chunk_text_by_tokens(text, processor, max_tokens=400):
    """
    Chunk text to stay under token limit, trying to break at sentence boundaries
    """
    # First check if text is already short enough
    tokens = processor.tokenizer(text).input_ids
    if len(tokens) <= max_tokens:
        return [text]
    
    # Split by sentences first
    sentences = text.replace('।', '।\n').replace('.', '.\n').replace('!', '!\n').replace('?', '?\n').split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(processor.tokenizer(sentence).input_ids)
        
        # If adding this sentence would exceed limit, save current chunk and start new one
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

# Function to chunk audio to match text chunks
def chunk_audio_proportionally(audio_array, text_chunks, total_text_length):
    """
    Split audio proportionally based on text chunk lengths
    """
    audio_chunks = []
    audio_length = len(audio_array)
    current_pos = 0
    
    for i, chunk_text in enumerate(text_chunks):
        chunk_length = len(chunk_text)
        chunk_ratio = chunk_length / total_text_length
        
        if i == len(text_chunks) - 1:  # Last chunk gets remaining audio
            chunk_audio = audio_array[current_pos:]
        else:
            chunk_samples = int(audio_length * chunk_ratio)
            chunk_audio = audio_array[current_pos:current_pos + chunk_samples]
            current_pos += chunk_samples
        
        audio_chunks.append(chunk_audio)
    
    return audio_chunks

# Only proceed if both directories exist
if not os.path.exists(audio_path) or not os.path.exists(transcript_path):
    print("\n=== DIRECTORY PATH ISSUE ===")
    print("One or both directories don't exist. Please check your paths:")
    print(f"Audio path: {audio_path}")
    print(f"Transcript path: {transcript_path}")
    print("\nTry checking your Google Drive mount or directory structure.")
    exit()

# Create file mapping using custom matching logic
print("\n=== CREATING FILE MAPPINGS ===")
file_pairs = create_file_mapping(audio_path, transcript_path)

if not file_pairs:
    print("\n=== NO FILE MATCHES FOUND ===")
    print("Please check if your files follow the expected naming patterns:")
    print("Audio: 'Konkani Prime News_DDMMYY.wav' (e.g., 'Konkani Prime News_070817.wav')")
    print("Transcript: 'DD AUG PRIME.odt' (e.g., '7 AUG PRIME.odt')")
    exit()

# Load Whisper processor first for tokenization
print("DEBUG: Loading Whisper processor...")
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
print("DEBUG: Whisper processor loaded successfully")

# Collect and chunk data with detailed debugging
print("\n=== PROCESSING MATCHED FILES ===")
data = []
processed_count = 0
skipped_count = 0

for audio_file, transcript_file in file_pairs:
    print(f"\n--- Processing pair: {audio_file} <-> {transcript_file} ---")
    
    full_audio_path = os.path.join(audio_path, audio_file)
    full_transcript_path = os.path.join(transcript_path, transcript_file)

    # Read transcript
    text = read_odt(full_transcript_path)
    if not text or len(text.strip()) == 0:
        print(f"DEBUG: Empty or invalid transcript for {transcript_file}")
        logger.warning(f"Empty transcript for {transcript_file}")
        skipped_count += 1
        continue

    # Load audio
    audio_array, sr = load_and_preprocess_audio(full_audio_path)
    if audio_array is None:
        print(f"DEBUG: Failed to load audio: {audio_file}")
        logger.warning(f"Failed to load audio: {audio_file}")
        skipped_count += 1
        continue

    # Check if text needs chunking
    tokens = processor.tokenizer(text).input_ids
    print(f"DEBUG: File {audio_file}: text length = {len(text)} chars, tokens = {len(tokens)}")
    logger.info(f"File {audio_file}: text length = {len(text)} chars, tokens = {len(tokens)}")
    
    if len(tokens) > 400:  # Need to chunk
        print(f"DEBUG: Chunking {audio_file} (tokens: {len(tokens)})")
        logger.info(f"Chunking {audio_file} (tokens: {len(tokens)})")
        text_chunks = chunk_text_by_tokens(text, processor, max_tokens=400)
        audio_chunks = chunk_audio_proportionally(audio_array, text_chunks, len(text))
        
        # Add each chunk as a separate training sample
        for i, (text_chunk, audio_chunk) in enumerate(zip(text_chunks, audio_chunks)):
            # Verify chunk token count
            chunk_tokens = len(processor.tokenizer(text_chunk).input_ids)
            print(f"DEBUG: Chunk {i+1}: {chunk_tokens} tokens, {len(audio_chunk)/sr:.2f}s audio")
            logger.info(f"  Chunk {i+1}: {chunk_tokens} tokens, {len(audio_chunk)/sr:.2f}s audio")
            
            data.append({
                "audio_array": audio_chunk,
                "audio_path": f"{audio_file}_chunk{i+1}",
                "text": text_chunk,
                "sampling_rate": sr
            })
    else:
        # Text is already short enough
        print(f"DEBUG: Adding full file as single sample")
        data.append({
            "audio_array": audio_array,
            "audio_path": audio_file,
            "text": text,
            "sampling_rate": sr
        })
    
    processed_count += 1
    logger.info(f"Successfully processed {audio_file} <-> {transcript_file}")

print(f"\n=== PROCESSING SUMMARY ===")
print(f"Total matched pairs: {len(file_pairs)}")
print(f"Successfully processed: {processed_count}")
print(f"Skipped due to issues: {skipped_count}")
print(f"Total training samples created: {len(data)}")

logger.info(f"Total samples collected: {len(data)}")

if len(data) == 0:
    print("\n=== NO VALID SAMPLES FOUND ===")
    print("Possible issues:")
    print("1. Matched files exist but transcripts are empty")
    print("2. Audio files are corrupted")
    print("3. File matching logic needs adjustment")
    raise ValueError("No valid samples found!")

print(f"\n=== SUCCESS ===")
print(f"Found {len(data)} valid training samples. Proceeding with training setup...")

# Continue with the rest of the training code...
# Create DataFrame and Dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Split dataset if we have multiple samples
if len(dataset) > 1:
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
else:
    train_dataset = dataset
    eval_dataset = None
    logger.info(f"Single sample training dataset: {len(train_dataset)}")

# Load model
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Simple preprocessing function with token length validation
def preprocess_function(example):
    audio_array = example["audio_array"]
    sr = example["sampling_rate"]
    text = example["text"]

    # Ensure audio_array is a NumPy array
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)

    # Process audio with Whisper feature extractor
    inputs = processor(audio_array, sampling_rate=int(sr), return_tensors="pt")

    # Process text with tokenizer and validate length
    labels = processor.tokenizer(text).input_ids
    
    # Final safety check - truncate if still too long
    max_length = 448
    if len(labels) > max_length:
        logger.warning(f"Truncating labels from {len(labels)} to {max_length} tokens")
        labels = labels[:max_length]

    return {
        "input_features": inputs.input_features[0].numpy(),
        "labels": labels
    }

# Apply preprocessing
logger.info("Applying preprocessing...")
train_dataset = train_dataset.map(
    preprocess_function,
    remove_columns=["audio_array", "audio_path", "text", "sampling_rate"],
    desc="Preprocessing train dataset"
)

if eval_dataset is not None:
    eval_dataset = eval_dataset.map(
        preprocess_function,
        remove_columns=["audio_array", "audio_path", "text", "sampling_rate"],
        desc="Preprocessing eval dataset"
    )

logger.info(f"Preprocessing complete. Train samples: {len(train_dataset)}")

# Data collator
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Metric computation
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 in the labels as we can't decode them
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    try:
        wer_score = wer(label_str, pred_str)
    except:
        wer_score = 1.0

    return {"wer": wer_score}

# Training arguments - adjusted for chunked data
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-konkani",
    per_device_train_batch_size=4,  # Can increase since chunks are smaller
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=100,
    num_train_epochs=3,  # Use epochs now that we have more samples
    logging_steps=25,
    save_steps=100,
    eval_steps=100 if eval_dataset else None,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    eval_strategy="steps" if eval_dataset else "no",
    save_strategy="steps",
    load_best_model_at_end=eval_dataset is not None,
    metric_for_best_model="wer" if eval_dataset else None,
    greater_is_better=False,
    remove_unused_columns=False,
    report_to="none",  # Disable wandb logging
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    compute_metrics=compute_metrics if eval_dataset else None,
)

# Train
logger.info("Starting training...")
try:
    trainer.train()
    logger.info("Training completed!")

    # Save model
    trainer.save_model("./whisper-konkani-final")
    processor.save_pretrained("./whisper-konkani-final")
    logger.info("Model saved!")

except Exception as e:
    logger.error(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
    raise

print("Training complete! Model saved to ./whisper-konkani-final")
