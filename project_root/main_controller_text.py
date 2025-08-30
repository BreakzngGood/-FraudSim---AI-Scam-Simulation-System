import os
import shutil
import json
import traceback
from pathlib import Path
from audio import process_audio_from_json
from poster import generate_poster
from banner import generate_banner
from deepseek_final import generate_one_case

def route_by_suffix(json_path):

    try:
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
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            generate_poster(title, body, output_path)

        elif suffix == 'B':
            print(f"🚨 Processing banner: {filename}")
            sticker_path = "stickers.png"
            if not os.path.exists(sticker_path):
                sticker_path = None
            generate_banner(json_path)

        else:
            print(f"❓ Unknown file type: {filename}")
        return True

    except Exception as e:
        print(f"❌ Error processing {json_path} : {str(e)}")
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("Fraud simulation generation")
    print("=" * 50)

    try:
        # Step 1: Generate JSON example
        print("\n🔄 Generating new case...")
        new_json_path = generate_one_case()
        new_json_path = Path(new_json_path).resolve()
        print(f"📁 Generate file path: {new_json_path}")

        if not new_json_path.exists():
            print(f"❌ File not found: {new_json_path}")
            print(f"📍 Current working directory: {os.getcwd()}")
            print(f"📁 text_prompt contents: {os.listdir('text_prompt')}")
            raise FileNotFoundError("No valid JSON file was generated")

        print(f"✅ Case generated successfully: {new_json_path.name}")

        # Step 2: Processing generated content
        print("\n🔄 Processing generated content...")
        if not route_by_suffix(str(new_json_path)):
            raise RuntimeError("Content processing failed")

        # Step 3: File in selected_text_prompt/
        archive_folder = Path("selected_text_prompt")
        archive_folder.mkdir(exist_ok=True)
        dest_path = archive_folder / new_json_path.name

        if dest_path.exists():
            dest_path.unlink()
        shutil.move(str(new_json_path), str(dest_path))

        print(f"\n✅ Processing completed, file archived to: {dest_path}")

    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("\n" + "=" * 50)
        print("🎉 All processing completed！")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("⚠️ An error occurred during processing, please check the log")
        print("=" * 50)
