import os
import random
from PIL import Image, ImageDraw
import math

# Tạo folder nếu chưa có
os.makedirs("dataset", exist_ok=True)

color = tuple(random.randint(0, 255) for i in range(3))
img = Image.new("RGB", (64, 64), (54, 24, 25))
x1 = random.randint(5, 30)
y1 = random.randint(5, 30)
x2 = random.randint(34, 59)
y2 = random.randint(34, 59)
side = min(x2-x1, y2-y1)
d = ImageDraw.Draw(img).rectangle([x1, y1, x1+side, y1+side])


# ❗ phải có tên file + extension
img.save("dataset/img_0.png")