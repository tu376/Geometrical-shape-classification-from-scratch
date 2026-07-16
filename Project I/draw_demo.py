import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from model import CNN, SHAPES

WEIGHTS_FILE = "weights.npy"
CANVAS_SIZE = 256
IMG_SIZE = 64
BRUSH_SIZE = 2


class DrawDemoApp:
    def __init__(self):
        if not os.path.exists(WEIGHTS_FILE):
            raise FileNotFoundError(
                f"Could not find '{WEIGHTS_FILE}'. Please place weights.npy in the project folder or train the model first."
            )

        self.model = CNN()
        self.model.load(WEIGHTS_FILE)
        self.model.training = False

        self.root = tk.Tk()
        self.root.title("Draw 64x64 Shape and Predict")
        self.root.resizable(False, False)

        self.image = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="cross",
        )
        self.canvas.pack(padx=10, pady=10)

        self.photo_image = None
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=None)

        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<Button-1>", self.on_paint)

        self.info_label = tk.Label(
            self.root,
            text="Draw a shape in the box, then click Predict.",
            font=("Arial", 12),
            fg="white",
            bg="black",
            anchor="w",
        )
        self.info_label.pack(fill="x", padx=10)

        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(fill="x", padx=10, pady=8)

        predict_button = tk.Button(buttons_frame, text="Predict", command=self.predict)
        clear_button = tk.Button(buttons_frame, text="Clear", command=self.clear)
        save_button = tk.Button(buttons_frame, text="Save", command=self.save_image)

        predict_button.pack(side="left", expand=True, fill="x", padx=4)
        clear_button.pack(side="left", expand=True, fill="x", padx=4)
        save_button.pack(side="left", expand=True, fill="x", padx=4)

        self.clear()

    def on_paint(self, event):
        x = int(event.x * IMG_SIZE / CANVAS_SIZE)
        y = int(event.y * IMG_SIZE / CANVAS_SIZE)
        radius = BRUSH_SIZE // 2
        bbox = [x - radius, y - radius, x + radius, y + radius]
        self.draw.ellipse(bbox, fill=255)
        self.update_canvas()

    def update_canvas(self):
        display_image = self.image.resize((CANVAS_SIZE, CANVAS_SIZE), resample=Image.NEAREST)
        self.photo_image = ImageTk.PhotoImage(display_image)
        self.canvas.itemconfig(self.canvas_image_id, image=self.photo_image)

    def clear(self):
        self.draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=0)
        self.update_canvas()
        self.info_label.config(text="Draw a shape in the box, then click Predict.")

    def predict(self):
        img_array = np.array(self.image, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, 1, IMG_SIZE, IMG_SIZE)
        preds, probs = self.model.predict(img_array)
        pred_idx = int(preds[0])
        pred_label = SHAPES[pred_idx]
        confidence = float(probs[0, pred_idx]) * 100.0

        prob_text = ", ".join(
            f"{SHAPES[i]}: {probs[0, i] * 100:.1f}%" for i in range(len(SHAPES))
        )

        self.info_label.config(
            text=f"Prediction: {pred_label} ({confidence:.1f}%) | {prob_text}"
        )

    def save_image(self):
        output_path = "drawn_shape.png"
        self.image.save(output_path)
        messagebox.showinfo("Saved", f"Saved drawn image as {output_path}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DrawDemoApp()
    app.run()
