# Based on: [https://huggingface.co/spaces/Matthijs/whisper_word_timestamps/tree/main]

import os
import sys
import argparse
import librosa
import torch
import pandas as pd
from pathlib import Path
from pydub import AudioSegment

from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline
)

# We need to set alignment_heads on the model's generation_config (at least
# until the models have been updated on the hub).
# If you're going to use a different version of whisper, see the following
# for which values to use for alignment_heads:
# https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a
alignment_heads_dict = {
    'tiny': [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]],
    'base': [[3, 1], [4, 2], [4, 3], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]],
    'small': [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]],
    'medium': [[13, 15], [15, 4], [15, 15], [16, 1], [20, 0], [23, 4]],
    'large' : [[10, 12], [13, 17], [16, 11], [16, 12], [16, 13], [17, 15], [17, 16], [18, 4], 
               [18, 11], [18, 19], [19, 11], [21, 2], [21, 3], [22, 3], [22, 9], [22, 12], 
               [23, 5], [23, 7], [23, 13], [25, 5], [26, 1], [26, 12], [27, 15]]
}
checkpoints = {
    'tiny': 'openai/whisper-tiny', 
    'base': 'openai/whisper-base',
    'small': 'openai/whisper-small',    # seems to be the best trade-off between speed and accuracy
    'medium': 'openai/whisper-medium',
    'large': 'openai/whisper-large'
}
cuda_up = torch.cuda.is_available()


def get_pipeline(checkpoint, alignment_heads):
    """Get the pipeline."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("Using GPU")
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint).to("cuda").half()
        processor = WhisperProcessor.from_pretrained(checkpoint)
        pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            batch_size=8,
            torch_dtype=torch.float16,
            device="cuda:0"
        )
        print("Memory allocated (CUDA): {} GB".format(torch.cuda.memory_allocated() / 1e9))
    else:
        print("Using CPU")
        pipe = pipeline(model=checkpoint)
        pipe.model.config.alignment_heads = alignment_heads
        print("Memory allocated (CPU): {} GB".format(sys.getsizeof(pipe) / 1e9))

    pipe.model.generation_config.alignment_heads = alignment_heads
    return pipe

def transcribe(audio_path, pipe, language, max_duration, rows_out):
    """Transcribe audio with Whisper and get word-level timestamps.
    
    Returns a DataFrame with the following columns:
    - word: the word
    - start: the start time of the word in seconds
    - end: the end time of the word in seconds"""
    if not Path(audio_path).is_file():
        raise ValueError(f"No such file: '{audio_path}'")

    audio_data, sr = librosa.load(audio_path, mono=True)
    duration = librosa.get_duration(y=audio_data, sr=sr)
    duration = min(max_duration, duration)
    audio_data = audio_data[:int(duration * sr)]

    if language is not None:
        pipe.model.config.forced_decoder_ids = (
            pipe.tokenizer.get_decoder_prompt_ids(
                language=language,
                task="transcribe"
            )
        )

    # Run Whisper to get word-level timestamps.
    audio_inputs = librosa.resample(audio_data, orig_sr=sr, target_sr=pipe.feature_extractor.sampling_rate)
    output = pipe(audio_inputs, chunk_length_s=30, stride_length_s=[4, 2], return_timestamps="word")
    chunks = output["chunks"]

    # Create a DataFrame for the output
    df = pd.DataFrame(chunks)
    df = df.rename(columns={'text': 'word'})

    # split timestamp into two columns
    df[['start', 'end']] = pd.DataFrame(df['timestamp'].tolist(), index=df.index)
    df.drop(columns=['timestamp'], inplace=True)

    csv_output_path = f"{Path(audio_path).stem[:6]}_transcription.csv"
    df.to_csv(csv_output_path, index=False)
    print(df.head(rows_out))
    return csv_output_path

def extract_word_audio(word, csv_path, audio_path):
    df = pd.read_csv(csv_path)
    # split words in 'word' column by non-alphabetic characters and check if any part matches the word
    rows = df[df['word'].str.lower().str.split(r'\W+').apply(lambda x: word.lower() in x)]
    print(f"Found {len(rows)} instances of '{word}' in '{csv_path}'.")
    audio = AudioSegment.from_file(audio_path)

    for idx, row in rows.iterrows():
        # convert float timestamp to the correct format
        start_time = int(row['start'] * 1000)  # pydub works in milliseconds
        end_time = int(row['end'] * 1000)
        word_audio = audio[start_time:end_time]
        # create filename with correctly formatted timestamp
        filename = f"validation/val_{Path(audio_path).stem[:6]}_{word}_{str(row['start']).replace('.', '')}_{str(row['end']).replace('.', '')}.mp3"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        word_audio.export(filename, format="mp3")


def get_parser():
    parser = argparse.ArgumentParser(description='Transcribe audio with Whisper and get word-level timestamps.')
    parser.add_argument('--audio', 
                        type=str, 
                        help='Path to the audio file.')
    parser.add_argument('--language', 
                        type=str, 
                        help='Language of the audio, e.g. english, german, french, etc.',
                        default='english')
    parser.add_argument('--info',
                        action='store_true',
                        help='Print information about the model.')
    parser.add_argument('--length',
                        type=int,
                        help='Used to limit the duration of the audio file in seconds. Default is 600 seconds (10 minutes).')
    parser.add_argument('--model',
                        type=str,
                        help='OpenAI Whisper model to use. e.g. small, medium, large.')
    parser.add_argument('--word', 
                        type=str, 
                        help='Word to extract from the audio. If provided, an mp3 file will be generated for each occurrence of the word in the audio.')
    return parser

def main():
    # ======== CONFIG ======== 

    # >>> Settings are overwritten by command line arguments <<<

    # The checkpoint to use. You can use any of the listed checkpoints above,
    # or any other model that has alignment heads available (see note on alignment heads above).
    # (the smaller the model, the faster it will run, but the less accurate it will be)
    model = 'small'
    max_duration = 600    # seconds // used to limit the duration of the audio file
    rows_out = 30         # number of rows // to print in the output DataFrame (for quick inspection)
    language = 'english'  # language of the audio file

    # ======== CONFIG ========

    checkpoint = checkpoints[model]
    alignment_heads = alignment_heads_dict[model]

    parser = get_parser()
    args = parser.parse_args()

    if args.length is not None:
        max_duration = args.length
    if args.model:
        checkpoint = checkpoints[args.model]
        alignment_heads = alignment_heads_dict[args.model]
    if args.info:
        print("CUDA available: {}".format(cuda_up))
        print("Model: {}".format(checkpoint))
        print(f"Alignment heads: {alignment_heads}")
    if args.language:
        language = args.language        
    if args.audio:
        pipe = get_pipeline(checkpoint, alignment_heads)
        csv_output_path = transcribe(args.audio, pipe, language, max_duration, rows_out)
        if args.word:
            extract_word_audio(args.word, csv_output_path, args.audio)
    elif args.word:
        # look for csv file in current directory
        csv_path = [file for file in os.listdir() if file.endswith('.csv')][0]
        # look for audio file in current directory
        audio_path = [file for file in os.listdir() if file.endswith('.mp3')][0]
        if csv_path and audio_path:
            print(f"Using '{csv_path}' and '{audio_path}'")
            extract_word_audio(args.word, csv_path, audio_path)
        else:
            print("Please provide an audio file. Use --audio <path/to/mp3>")
    
    # if settings provided but no audio file
    elif args.length or args.model or args.language:
        print("Please provide an audio file. Use --audio <path/to/mp3>")


if __name__ == "__main__":
    main()