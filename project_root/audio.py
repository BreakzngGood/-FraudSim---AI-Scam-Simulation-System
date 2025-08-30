import os
import numpy as np
import torch
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.dtype,
    np.dtypes.Float64DType,
    np.float64,
    np.float32,
    np.int32,
    np.int64,
])

import librosa
import json
import random
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["SUNO_USE_SMALL_MODELS"] = "1"

from scipy.signal import butter, lfilter
import nltk  # we'll use this to split into sentences

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from bark.generation import (
    generate_text_semantic,
    preload_models,
    ALLOWED_PROMPTS
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
import re
import soundfile as sf

# Initialize Model
preload_models()

sid = SentimentIntensityAnalyzer()
GEN_TEMP = 0.6
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

lang_label_dict = {
    "English": "I am a fraudster...",
    "Chinese": "我是一名欺诈者...",
    "French": "Je suis un fraudeur...",
    "German": "Ich bin ein Betrüger...",
    "Hindi": r"मैं एक धोखेबाज़ हूँ...",
    "Italian": "Sono un truffatore...",
    "Japanese": "私は詐欺師です...",
    "Korean": "저는 사기꾼입니다...",
    "Polish": "Jestem oszustem...",
    "Portuguese": "Eu sou um fraudador...",
    "Russian": "Я мошенник...",
    "Spanish": "Soy un estafador...",
    "Turkish": r"Ben bir dolandırıcıyım..."
}

speaker_dict= {
    "female" : "v2/en_speaker_9",
    "male": "v2/en_speaker_6"
}

# Main function
def process_audio_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filename = os.path.splitext(os.path.basename(json_path))[0]
    text_prompt = data.get("generated", {}).get("audio_script", "")

    emotion_text = prompt_sentiment(text_prompt)
    new_prompt, speaker_selection = process_tagged_prompt(text_prompt, emotion_text)
    new_prompt = process_prompt_format(new_prompt)
    sentences = split_sentence(new_prompt)

    # if speaker_selection not in ALLOWED_PROMPTS:
    #     print(f"⚠️ Speaker {speaker_selection} is not in ALLOWED_PROMPTS, defaults to 'v2/en_speaker_6'")
    #     speaker_selection = "v2/en_speaker_6"

    pieces = []
    for sentence in sentences:
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=speaker_selection,
            temp=GEN_TEMP,
            min_eos_p=0.05,
        )
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=speaker_selection)
        pieces += [audio_array, silence.copy()]

    final_audio = np.concatenate(pieces[2:])

    os.makedirs("outputs/audio", exist_ok=True)
    input_file = os.path.join("outputs/audio", f"{filename}.wav")
    filtered_file = os.path.join("outputs/audio", f"{filename}_telephone_filtered.wav")
    fast_file = os.path.join("outputs/audio", f"{filename}_fast.wav")

    sf.write(input_file, final_audio, SAMPLE_RATE, subtype='PCM_16')
    apply_telephone_filter(input_file, filtered_file)
    change_audio_speed(filtered_file, fast_file)
    print(f"✅ Audio generation completed: {fast_file}")

def prompt_sentiment(prompt):
    cleaned_text = prompt.split(":", 1)[1].strip()
    score = sid.polarity_scores(cleaned_text)
    if score["compound"] >= 0.05:
        return "[laughs]"
    elif score["compound"] <= -0.05:
        return '[sighs]'
    else:
        return None

def process_tagged_prompt(prompt, emotion_text):
    prompt_tags = re.sub(r"\((.*?)\)", r"[\1]", prompt)
    tags = re.findall(r"\[.*?\]", prompt_tags)
    if len(tags) < 3 or ":" not in prompt_tags:
        return prompt_tags, speaker_dict["male"]

    gender_tag = tags[2]
    country_tag = tags[0]
    before_colon, after_colon = prompt_tags.split(":", 1)

    gender_tag_content = re.findall(r"\[(.*?)\]", gender_tag)[0]
    country_tag_content = re.findall(r"\[(.*?)\]", country_tag)[0].capitalize()

    if gender_tag_content in speaker_dict:
        speaker_selection = speaker_dict[gender_tag_content]
        gender_tag_content = '[woman]' if gender_tag_content == 'female' else '[man]'
    else:
        speaker_selection = speaker_dict["male"]
        gender_tag_content = '[man]'

    if country_tag_content in lang_label_dict and country_tag_content != "English":
        new_content = lang_label_dict[country_tag_content]
        new_prompt = f"{gender_tag_content}{emotion_text or ''}{new_content}{after_colon.strip()}"
    else:
        new_prompt = f"{gender_tag_content}{emotion_text or ''} {after_colon.strip()}"

    return new_prompt, speaker_selection

def process_prompt_format(prompt):
    return prompt.replace(r"’", r"'").replace(r"...", ".").replace("\n", " ").strip()

def split_sentence(text):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []
    first_sentence = sentences[0]
    tags = re.findall(r"\[.*?\]", first_sentence)
    tag_text = "".join(tags)
    content = re.sub(r"\[.*?\]", "", first_sentence)
    match = re.search(r"[a-zA-Z]", content)
    if match:
        idx = match.start()
        foreign_part = tag_text + content[:idx].strip()
        english_part = content[idx:].strip()
        return [foreign_part, english_part] + sentences[1:]
    else:
        return sentences

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def apply_telephone_filter(input_wav_path, output_wav_path, lowcut=300.0, highcut=3400.0, order=6):
    audio_data, sample_rate = sf.read(input_wav_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    filtered_audio = bandpass_filter(audio_data, lowcut, highcut, sample_rate, order)
    sf.write(output_wav_path, filtered_audio, sample_rate)

def change_audio_speed(input_wav_path, output_wav_path, speed_rate=1.3):
    audio_data, sr = librosa.load(input_wav_path, sr=None)
    audio_stretched = librosa.effects.time_stretch(y=audio_data, rate=speed_rate)
    sf.write(output_wav_path, audio_stretched, sr)
