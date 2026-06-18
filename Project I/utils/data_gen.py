import os
import csv
from PIL import Image, ImageDraw, ImageFilter
import random
import math
import numpy as np
from scipy.ndimage import convolve

DATASET_DIR = "dataset"
CSV_FILE    = "labels.csv"

SIZE       = 64
NUM_IMAGES = 10000
SHAPES     = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]

os.makedirs(DATASET_DIR, exist_ok=True)

# ==========================================================
# SHAPES
# ==========================================================

def draw_polygon(draw, cx, cy, radius, sides, color):
    angle_offset = -math.pi / 2   # start from top
    points = []
    for i in range(sides):
        angle = angle_offset + 2 * math.pi * i / sides
        points.append((cx + radius * math.cos(angle),
                        cy + radius * math.sin(angle)))
    draw.polygon(points, outline=color, width=1)


def draw_shape(draw, shape):
    color = 255
    cx    = random.randint(20, 44)
    cy    = random.randint(20, 44)

    if shape == "circle":
        # always perfect square bounding box
        r = random.randint(10, 20)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=1)

    elif shape == "ellipse":
        # enforce noticeable aspect ratio — never close to 1:1
        r_major = random.randint(18, 26)
        r_minor = random.randint(7,  12)   # always much smaller than major
        if random.random() < 0.5:
            draw.ellipse([cx-r_major, cy-r_minor, cx+r_major, cy+r_minor], outline=color, width=1)
        else:
            draw.ellipse([cx-r_minor, cy-r_major, cx+r_minor, cy+r_major], outline=color, width=1)

    elif shape == "square":
        # always equal sides
        s = random.randint(10, 22)
        draw.rectangle([cx-s, cy-s, cx+s, cy+s], outline=color, width=1)

    elif shape == "rectangle":
        # enforce noticeable aspect ratio — never close to 1:1
        w = random.randint(20, 28)
        h = random.randint(7,  13)   # always much smaller than w
        if random.random() < 0.5:
            draw.rectangle([cx-w, cy-h, cx+w, cy+h], outline=color, width=1)
        else:
            draw.rectangle([cx-h, cy-w, cx+h, cy+w], outline=color, width=1)

    elif shape == "triangle":
        offset = random.randint(5, 10)
        r      = random.randint(15, 25)
        points = [
            (cx,          cy - r),
            (cx - r,      cy + r - offset),
            (cx + r,      cy + r - offset),
        ]
        draw.polygon(points, outline=color, width=1)

    elif shape == "hexagon":
        draw_polygon(draw, cx, cy, random.randint(14, 24), 6, color)

    elif shape == "octagon":
        draw_polygon(draw, cx, cy, random.randint(14, 22), 8, color)


# ==========================================================
# GENERATE
# ==========================================================

# with open(CSV_FILE, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["filename", "label"])

#     for i in range(NUM_IMAGES):
#         img   = Image.new("L", (SIZE, SIZE), 0)
#         draw  = ImageDraw.Draw(img)

#         shape = random.choice(SHAPES)
#         draw_shape(draw, shape)

#         filename = f"img_{i}.png"
#         img.save(os.path.join(DATASET_DIR, filename))
#         writer.writerow([filename, shape])

# print(f"Done! {NUM_IMAGES} images saved to {DATASET_DIR}/")
if __name__ == "__main__":
    img   = Image.new("L", (SIZE, SIZE), 0)
    draw  = ImageDraw.Draw(img)

    shape = "octagon"
    draw_shape(draw, shape)

    # add noise to ~70% of images
    if random.random() < 0.7:
        arr = np.array(img)
        img = Image.fromarray(arr)

    filename = f"img_{0}.png"
    img.save(filename)
