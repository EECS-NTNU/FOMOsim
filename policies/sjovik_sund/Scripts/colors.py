from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


swatches = [
    ("Pale Fern", "#D9E5D6"),
    ("Sage Green", "#84A579"),
    ("Dark Spruce", "#344E41"),
    ("Warm Red", "#D1495B"),
    ("Ochre Gold", "#EDAE49"),
    ("Blue Accent", "#526ECA"),  
    ("Mist Grey", "#E6E6E6"),
]

width = 800
height = 100 * len(swatches)
img = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 32)
except:
    font = ImageFont.load_default()

for i, (name, hexcode) in enumerate(swatches):
    y = i * 100
    draw.rectangle([0, y, 200, y+100], fill=hexcode)
    draw.text((220, y+30), f"{name} – {hexcode}", fill="black", font=font)

# Save in the same folder as this script
script_dir = Path(__file__).parent
filepath = script_dir / "forest_palette3.png"
img.save(filepath)

filepath