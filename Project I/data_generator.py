import os
import csv
from PIL import Image, ImageDraw
import random
import math

DATASET_DIR = "dataset"
CSV_FILE = "labels.csv"

SIZE = 64
NUM_IMAGES = 1000
SHAPES = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]

os.makedirs(DATASET_DIR, exist_ok=True)

def random_color():
    return tuple(random.randint(36, 236) for _ in range(3))

def draw_polygon(draw, cx, cy, radius, sides, color):
    points = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=color)

def draw_shape(draw, shape):
    color = random_color()

    x1 = random.randint(5, 30)
    y1 = random.randint(5, 30)
    x2 = random.randint(34, 59)
    y2 = random.randint(34, 59)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = min(x2 - x1, y2 - y1) // 2

    if shape == "circle":
        draw.ellipse([x1, y1, x2, y2], fill=color)

    elif shape == "ellipse":
        draw.ellipse([x1, y1, x2, y2], fill=color)

    elif shape == "square":
        side = min(x2 - x1, y2 - y1)
        draw.rectangle([x1, y1, x1 + side, y1 + side], fill=color)

    elif shape == "rectangle":
        draw.rectangle([x1, y1, x2, y2], fill=color)

    elif shape == "triangle":
        points = [(cx, y1), (x1, y2), (x2, y2)]
        draw.polygon(points, fill=color)

    elif shape == "hexagon":
        draw_polygon(draw, cx, cy, radius, 6, color)

    elif shape == "octagon":
        draw_polygon(draw, cx, cy, radius, 8, color)

with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    for i in range(NUM_IMAGES):
        img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        shape = random.choice(SHAPES)
        draw_shape(draw, shape)

        filename = f"img_{i}.png"
        filepath = os.path.join(DATASET_DIR, filename)

        img.save(filepath)
        writer.writerow([filename, shape])

print("Done!")