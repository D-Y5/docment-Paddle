from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

image = Image.new("RGB", (800, 600), color="white")
draw = ImageDraw.Draw(image)
try:
    font_large = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
    )
    font_medium = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36
    )
    font_small = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
    )
except:
    font_large = ImageFont.load_default()
    font_medium = ImageFont.load_default()
    font_small = ImageFont.load_default()
draw.text((200, 200), "Test Document", fill="black", font=font_large)
draw.text((280, 280), "Sample Text", fill="black", font=font_medium)
draw.text((100, 380), "Line 1", fill="black", font=font_small)
draw.text((100, 420), "Line 2", fill="black", font=font_small)
draw.rectangle([50, 50, 750, 550], outline="black", width=5)
output_path = Path("test_document_original.jpg")
image.save(output_path)
print(f"Original test document saved to: {output_path}")
