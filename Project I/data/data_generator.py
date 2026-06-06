import os
import csv
from PIL import Image, ImageDraw
import random
import math
import numpy as np

DATASET_DIR = "dataset"
CSV_FILE = "data\labels.csv"

SIZE = 64
NUM_IMAGES = 5000
SHAPES = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]

os.makedirs(DATASET_DIR, exist_ok=True)


def draw_polygon(draw, cx, cy, radius, sides, color):
    points = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, outline=color, width=1)

def draw_shape(draw, shape):
    color = 255

    x1 = random.randint(5, 25)
    y1 = random.randint(5, 25)
    x2 = random.randint(38, 59)
    y2 = random.randint(38, 59)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = min(x2 - x1, y2 - y1) // 2

    if shape == "circle":
        side = min(x2 - x1, y2 - y1)
        draw.ellipse([cx - side//2, cy - side//2,
                      cx + side//2, cy + side//2], outline=color, width=1)

    elif shape == "ellipse":
        # Đảm bảo ellipse không tròn: 1 chiều phải dài hơn ít nhất 1.4x
        w = x2 - x1
        h = y2 - y1
        if w / h > 0.8 and w / h < 1.25:
            # Kéo dài theo chiều ngang hoặc dọc
            if random.random() > 0.5:
                x2 = x1 + int(h * 1.5)
            else:
                y2 = y1 + int(w * 1.5)
        draw.ellipse([x1, y1, x2, y2], outline=color, width=1)

    elif shape == "square":
        side = min(x2 - x1, y2 - y1)
        draw.rectangle([cx - side//2, cy - side//2,
                        cx + side//2, cy + side//2], outline=color, width=1)

    elif shape == "rectangle":
        # Đảm bảo rectangle không vuông: ratio phải > 1.3
        w = x2 - x1
        h = y2 - y1
        if 0.8 < w / h < 1.3:
            if random.random() > 0.5:
                x2 = x1 + int(h * 1.5)
            else:
                y2 = y1 + int(w * 1.5)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

    elif shape == "triangle":
        points = [(cx, y1), (x1, y2), (x2, y2)]
        draw.polygon(points, outline=color, width=1)

    elif shape == "hexagon":
        draw_polygon(draw, cx, cy, radius, 6, color)

    elif shape == "octagon":
        draw_polygon(draw, cx, cy, radius, 8, color)

with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    for i in range(NUM_IMAGES):
        img = Image.new("L", (SIZE, SIZE), 0)
        draw = ImageDraw.Draw(img)

        shape = random.choice(SHAPES)
        draw_shape(draw, shape)

        filename = f"img_{i}.png"
        filepath = os.path.join(DATASET_DIR, filename)

        img.save(filepath)
        writer.writerow([filename, shape])

print("Done!")

# if __name__ == "__main__":
#     img = Image.new("L", (SIZE, SIZE), 0)
#     draw = ImageDraw.Draw(img)

#     shape = random.choice(SHAPES)
#     draw_shape(draw, shape)

#     filename = f"img_{0}.png"

#     img.save(filename)
#     print("Done!")