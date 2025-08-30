import os
import shutil
import json
from audio import process_audio_from_json
from poster import generate_poster
from banner import generate_banner
import torch


def route_by_suffix(json_path):
    filename = os.path.basename(json_path)
    suffix = filename.split('-')[-1].split('.')[0].upper()

    if suffix == 'A':
        print(f"🔊 Processing Audio: {filename}")
        process_audio_from_json(json_path)
    elif suffix == 'P':
        print(f"🖼️ Processing poster: {filename}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        title = data["generated"]["poster_title"]
        body = data["generated"]["poster_body"]
        output_path = f"outputs/poster/{filename.replace('.json', '.png')}"
        generate_poster(title, body, output_path)
    elif suffix == 'B':
        print(f"🚨 Processing banner: {filename}")
        generate_banner(json_path)
    else:
        print(f"❓ Unknown file type: {filename}")

if __name__ == "__main__":
    input_folder = "text_prompt"
    archive_folder = "selected_text_prompt"
    os.makedirs(archive_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not files:
        print("❌ JSON file not found, please put the file in text_prompt/ folder.")
    else:
        for file in files:
            full_path = os.path.join(input_folder, file)
            print(f"Processing File: {file}")
            route_by_suffix(full_path)
            # Move to Archive Folder
            archive_path = os.path.join(archive_folder, file)
            shutil.move(full_path, archive_path)
            print(f"Moved to {archive_folder}/\n")
