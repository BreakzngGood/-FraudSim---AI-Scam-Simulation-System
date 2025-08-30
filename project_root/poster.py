import os
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline

IMAGE_SIZE = (1080, 1920)
TITLE_FONT_PATH = "arialbd.ttf"
BODY_FONT_PATH = "arial.ttf"
AI_MODEL_ID = "runwayml/stable-diffusion-v1-5"
BG_PROMPT = "A clean UK immigration office waiting area, neutral color palette, passport posters, plastic chairs, LED counter display, no people, no text, realistic style"

BACKGROUND_COLORS = {
    "white": (255, 255, 255),
    "deep_red": (80, 30, 30),
    "deep_blue": (30, 30, 80)
}

BANNER_COLORS = {
    "white": (200, 0, 0),       
    "deep_red": (0, 0, 0),  
    "deep_blue": (200, 0, 0)    
}

TEXT_COLORS = {
    "white": (0, 0, 0),
    "deep_red": (255, 255, 255),
    "deep_blue": (255, 255, 255)
}


def generate_ai_background(prompt: str, output_path: str, steps: int = 15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        AI_MODEL_ID, torch_dtype=dtype
    ).to(device)
    image = pipe(prompt, num_inference_steps=steps).images[0]
    image.save(output_path)
    return image

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        width = draw.textbbox((0, 0), test_line, font=font)[2]
        if width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

def add_text(image, title, body, font_color, banner_color, show_banner, show_border):
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(TITLE_FONT_PATH, size=100)
    body_font = ImageFont.truetype(BODY_FONT_PATH, size=50)

    W, H = image.size
    max_width = W - 100
    y_text = 100

    title_lines = wrap_text(draw, title, title_font, max_width)
    title_height = sum([
        draw.textbbox((0, 0), line, font=title_font)[3] - draw.textbbox((0, 0), line, font=title_font)[1] + 30
        for line in title_lines
    ]) - 30

    if show_banner:
        banner_height = title_height + 60
        banner_top = y_text - 30
        draw.rectangle([(0, banner_top), (W, banner_top + banner_height)], fill=banner_color)

    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_width = bbox[2] - bbox[0]
        draw.text(
            ((W - line_width) / 2, y_text),
            line,
            font=title_font,
            fill=font_color,
            stroke_width=3,
            stroke_fill=(0, 0, 0)
        )
        y_text += bbox[3] - bbox[1] + 30
    y_text += 40

    body_lines = wrap_text(draw, body, body_font, max_width)

    if show_border:
        top = y_text - 20
        bottom = top + len(body_lines) * (body_font.size + 20) + 40
        padding = 60
        outer_box = [(padding - 5, top - 5), (W - padding + 5, bottom + 5)]
        inner_box = [(padding, top), (W - padding, bottom)]
        draw.rectangle(outer_box, outline=banner_color, width=6)
        draw.rectangle(inner_box, outline=banner_color, width=2)

    for line in body_lines:
        bbox = draw.textbbox((0, 0), line, font=body_font)
        line_width = bbox[2] - bbox[0]
        draw.text(
            ((W - line_width) / 2, y_text),
            line,
            font=body_font,
            fill=font_color
        )
        y_text += bbox[3] - bbox[1] + 20

    return image


def add_sticker(image, sticker_folder="stickers", scale_factor=1.2):
    if not os.path.isdir(sticker_folder):
        return image

    sticker_files = [f for f in os.listdir(sticker_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not sticker_files:
        return image

    sticker_path = os.path.join(sticker_folder, random.choice(sticker_files))
    sticker = Image.open(sticker_path).convert("RGBA")

    new_size = (int(sticker.width * scale_factor), int(sticker.height * scale_factor))
    sticker = sticker.resize(new_size, Image.LANCZOS)

    padding = 50
    image.paste(sticker, (padding, image.height - new_size[1] - padding), sticker)
    return image


def generate_poster(title: str, body: str, output_path: str = "outputs/poster/final_poster.png"):
    title = title.replace("*", "").strip()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bg_name = random.choice(list(BACKGROUND_COLORS.keys()))
    bg_color = BACKGROUND_COLORS[bg_name]
    text_color = TEXT_COLORS[bg_name]
    banner_color = BANNER_COLORS[bg_name]

    show_banner = random.choice([True, False])
    show_border = random.choice([True, False])

    ai_bg_path = os.path.join(os.path.dirname(output_path), "ai_background.jpg")
    ai_bg = generate_ai_background(BG_PROMPT, ai_bg_path)

    base = Image.new("RGB", IMAGE_SIZE, bg_color)
    ai_bg = ai_bg.resize(IMAGE_SIZE).convert("RGBA")
    ai_bg.putalpha(int(255 * 0.40))
    base = base.convert("RGBA")
    base = Image.alpha_composite(base, ai_bg)

    poster = add_text(base.convert("RGB"), title, body, text_color, banner_color, show_banner, show_border)
    poster = add_sticker(poster)

    poster.save(output_path)
    print(f"✅ : {output_path}")



# if __name__ == "__main__":
#     TITLE = "Visa Status Alert: Please Act Now!"
#     BODY = (
#         "Our system has flagged inconsistencies in your visa records.\n"
#         "Please verify your identity within 24 hours to avoid:\n"
#         "- Visa cancellation\n"
#         "- Report to university/employer\n"
#         "- Forced deportation\n"
#         "This message is part of an official review. Do not ignore."
#     )
#     generate_poster(TITLE, BODY, "outputs/poster/final_poster.png")
