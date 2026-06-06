import numpy as np
from PIL import Image

from model import CNN

SHAPES = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]

# =========================
# Load model
# =========================
model = CNN()
model.load("weights.npy")

# =========================
# Load image
# =========================
image = Image.open("img_0.png").convert("L")

# resize đúng size model
image = image.resize((64, 64))

# normalize về [0,1]
image = np.array(image, dtype=np.float32) / 255.0

# shape:
# (64,64) -> (1,64,64)
image = image.reshape(1, 64, 64)

# =========================
# Predict
# =========================
pred, probs = model.predict(image)

print("Predict:", SHAPES[pred])

print("\nProbabilities:")
for i, p in enumerate(probs):
    print(f"{SHAPES[i]}: {p:.4f}")