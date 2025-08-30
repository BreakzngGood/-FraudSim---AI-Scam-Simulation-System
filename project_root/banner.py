import os
import json
import re
import textwrap
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
from pathlib import Path

def get_text_color(image_region):
    avg_color = image_region.resize((1, 1)).getpixel((0, 0))
    r, g, b = avg_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 160 else "white"

def draw_text_with_outline(draw, pos, text, font, fill, outline_color="black", outline_width=2, align="left"):
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color, align=align)
    draw.text((x, y), text, font=font, fill=fill, align=align)

def draw_highlighted_text(draw, x, y, text, font, normal_color, highlight_color, highlight_words, outline_color="black"):
    words = text.split()
    for word in words:
        clean_word = re.sub(r"[^\w£$]", "", word.lower())
        is_highlight = any(hw in clean_word for hw in highlight_words)
        color = highlight_color if is_highlight else normal_color
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), word + " ", font=font, fill=outline_color)
        draw.text((x, y), word + " ", font=font, fill=color)
        x += draw.textlength(word + " ", font=font)

def draw_multiline_highlighted_text(draw, text, font, x_start, y_start, max_width, line_height, normal_color, highlight_color, highlight_words, outline_color="black"):
    words = text.split()
    current_line = ""
    y = y_start
    for word in words:
        test_line = current_line + word + " "
        if draw.textlength(test_line, font=font) > max_width:
            draw_highlighted_text(draw, x_start, y, current_line.strip(), font, normal_color, highlight_color, highlight_words, outline_color)
            y += line_height
            current_line = word + " "
        else:
            current_line = test_line
    if current_line:
        draw_highlighted_text(draw, x_start, y, current_line.strip(), font, normal_color, highlight_color, highlight_words, outline_color)

def adjust_image_opacity(image: Image.Image, alpha: float) -> Image.Image:
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    r, g, b, a = image.split()
    a = a.point(lambda p: int(p * alpha))
    return Image.merge('RGBA', (r, g, b, a))


def generate_banner(json_path):
    project_root = os.path.dirname(os.path.abspath(__file__))
    sticker_path = os.path.join(project_root, "sticker.png")

    output_width, output_height = 896, 512
    num_steps = 25
    model_id = "runwayml/stable-diffusion-v1-5"
    overlay_alpha = 50

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    title = data["generated"]["poster_title"]
    body = data["generated"]["poster_body"]
    sd_background_prompt = data["generated"]["sd_background_prompt"]

    
    title = title.replace("*", "").strip()

    highlight_words = [
        "immediately", "urgent", "now", "asap", "quickly", "today", "deadline", "expire", "last chance",
        "payment", "pay", "fee", "charge", "money", "bank", "account", "transfer", "credit", "debit", "balance",
        "click", "link", "call", "text", "visit", "open", "reply", "login", "verify", "update", "confirm",
        "suspended", "blocked", "deactivated", "arrested", "lawsuit", "court", "fine", "warning", "penalty", "deported", "reported",
        "government", "irs", "customs", "officer", "security", "agent", "support", "admin",
        "win", "winner", "gift", "reward", "congratulations", "selected", "prize", "lottery", "lucky"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    image = pipe(prompt=sd_background_prompt, height=output_height, width=output_width, num_inference_steps=num_steps).images[0].convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, overlay_alpha))
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.truetype("arialbd.ttf", 40)
    body_font = ImageFont.truetype("arial.ttf", 22)
    title_lines = textwrap.wrap(title, width=28)
    title_height = len(title_lines) * 44 + 30
    title_color = "white"
    outline_color = "black"

    draw.rectangle((0, 0, output_width, title_height), fill=(0, 0, 0, 128))
    y = 10
    for line in title_lines:
        text_width = draw.textlength(line, font=title_font)
        x = (output_width - text_width) // 2
        draw_text_with_outline(draw, (x, y), line, title_font, fill=title_color, outline_color=outline_color)
        y += 44

    draw.rectangle((0, title_height - 8, output_width, title_height), fill=(255, 0, 0, 220))
    margin_x = 30
    max_body_width = output_width - 2 * margin_x
    line_height = 28

    sample_words = body.split()
    line_count = 1
    current_line = ""
    for word in sample_words:
        test_line = current_line + word + " "
        if draw.textlength(test_line, font=body_font) > max_body_width:
            line_count += 1
            current_line = word + " "
        else:
            current_line = test_line

    body_height = line_count * line_height + 20
    body_y_start = output_height - body_height - 30
    draw.rectangle((margin_x - 10, body_y_start - 10, output_width - margin_x + 10, body_y_start + body_height), fill=(0, 0, 0, 128))

    draw_multiline_highlighted_text(
        draw=draw,
        text=body,
        font=body_font,
        x_start=margin_x,
        y_start=body_y_start,
        max_width=max_body_width,
        line_height=line_height,
        normal_color="white",
        highlight_color="yellow",
        highlight_words=highlight_words,
        outline_color="black"
    )

    try:
        sticker = Image.open(sticker_path).convert("RGBA")
        sticker_width = int(output_width * 0.4)
        sticker = sticker.resize((sticker_width, int(sticker.size[1] * sticker_width / sticker.size[0])), resample=Image.Resampling.LANCZOS)
        sticker = adjust_image_opacity(sticker, alpha=0.7)
        x_center = (output_width - sticker.size[0]) // 2
        y_center = (output_height - sticker.size[1]) // 2
        image.paste(sticker, (x_center, y_center), sticker)
    except Exception as e:
        print("Sticker failed to load:", e)

    os.makedirs("outputs/banner", exist_ok=True)
    json_stem = Path(json_path).stem  # e.g., "1-3-3-a"

    output_name = os.path.join("outputs/banner", f"{json_stem}.png")
    image.convert("RGB").save(output_name)
    print("Banner save as:", output_name)


'''
For separate test：

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    json_path = current_file.parent.parent / "all_cases" / "4-8-38-b.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    generate_banner(json_path)

'''