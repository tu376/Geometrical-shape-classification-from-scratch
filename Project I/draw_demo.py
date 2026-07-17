import math
import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from model import CNN, SHAPES

WEIGHTS_FILE = "weights.npy"
CANVAS_SIZE = 256
IMG_SIZE = 64
DEFAULT_BRUSH_SIZE = 1


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
        self.root.title("Paint-like Shape Drawer")
        self.root.resizable(False, False)

        self.image = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.shape_start = None
        self.current_tool = "rectangle"
        self.current_color = 255
        self.current_brush_size = DEFAULT_BRUSH_SIZE

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

        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.info_frame = tk.Frame(self.root, bg="#1e1e1e", bd=1, relief="solid", padx=10, pady=8)
        self.info_frame.pack(fill="x", padx=10)

        self.info_label = tk.Label(
            self.info_frame,
            text="Vẽ nét mảnh, rõ ràng và giống dữ liệu đầu vào. Sau đó bấm Predict.",
            font=("Arial", 11, "bold"),
            fg="#f5f5f5",
            bg="#1e1e1e",
            anchor="w",
            justify="left",
            wraplength=500,
        )
        self.info_label.pack(fill="x")

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill="x", padx=10, pady=8)

        tools_frame = tk.Frame(controls_frame)
        tools_frame.pack(fill="x", pady=(0, 4))

        self.rectangle_button = tk.Button(tools_frame, text="Rectangle", command=lambda: self.set_tool("rectangle"))
        self.ellipse_button = tk.Button(tools_frame, text="Ellipse", command=lambda: self.set_tool("ellipse"))
        self.triangle_button = tk.Button(tools_frame, text="Triangle", command=lambda: self.set_tool("triangle"))
        self.hexagon_button = tk.Button(tools_frame, text="Hexagon", command=lambda: self.set_tool("hexagon"))
        self.octagon_button = tk.Button(tools_frame, text="Octagon", command=lambda: self.set_tool("octagon"))
        self.pen_button = tk.Button(tools_frame, text="Pen", command=lambda: self.set_tool("pen"))

        self.rectangle_button.pack(side="left", padx=2, pady=2)
        self.ellipse_button.pack(side="left", padx=2, pady=2)
        self.triangle_button.pack(side="left", padx=2, pady=2)
        self.hexagon_button.pack(side="left", padx=2, pady=2)
        self.octagon_button.pack(side="left", padx=2, pady=2)
        self.pen_button.pack(side="left", padx=2, pady=2)

        actions_frame = tk.Frame(controls_frame)
        actions_frame.pack(fill="x")

        predict_button = tk.Button(actions_frame, text="Predict", command=self.predict)
        clear_button = tk.Button(actions_frame, text="Clear", command=self.clear)
        save_button = tk.Button(actions_frame, text="Save", command=self.save_image)

        predict_button.pack(side="left", padx=2, pady=2)
        clear_button.pack(side="left", padx=2, pady=2)
        save_button.pack(side="left", padx=2, pady=2)

        self.set_tool("rectangle")
        self.clear()

    def _to_image_coords(self, x, y):
        return int(x * IMG_SIZE / CANVAS_SIZE), int(y * IMG_SIZE / CANVAS_SIZE)

    def set_tool(self, tool):
        self.current_tool = tool
        if tool == "rectangle":
            self.rectangle_button.config(relief="sunken")
            self.ellipse_button.config(relief="raised")
            self.triangle_button.config(relief="raised")
            self.hexagon_button.config(relief="raised")
            self.octagon_button.config(relief="raised")
        elif tool == "ellipse":
            self.rectangle_button.config(relief="raised")
            self.ellipse_button.config(relief="sunken")
            self.triangle_button.config(relief="raised")
            self.hexagon_button.config(relief="raised")
            self.octagon_button.config(relief="raised")
        elif tool == "triangle":
            self.rectangle_button.config(relief="raised")
            self.ellipse_button.config(relief="raised")
            self.triangle_button.config(relief="sunken")
            self.hexagon_button.config(relief="raised")
            self.octagon_button.config(relief="raised")
        elif tool == "hexagon":
            self.rectangle_button.config(relief="raised")
            self.ellipse_button.config(relief="raised")
            self.triangle_button.config(relief="raised")
            self.hexagon_button.config(relief="sunken")
            self.octagon_button.config(relief="raised")
        elif tool == "octagon":
            self.rectangle_button.config(relief="raised")
            self.ellipse_button.config(relief="raised")
            self.triangle_button.config(relief="raised")
            self.hexagon_button.config(relief="raised")
            self.octagon_button.config(relief="sunken")
            self.pen_button.config(relief="raised")
        elif tool == "pen":
            self.rectangle_button.config(relief="raised")
            self.ellipse_button.config(relief="raised")
            self.triangle_button.config(relief="raised")
            self.hexagon_button.config(relief="raised")
            self.octagon_button.config(relief="raised")
            self.pen_button.config(relief="sunken")

    def _draw_shape_preview(self, image, start, end, mode):
        draw = ImageDraw.Draw(image)
        x0, y0 = start
        x1, y1 = end

        if mode == "rectangle":
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            x2 = x0 + width if x1 >= x0 else x0 - width
            y2 = y0 + height if y1 >= y0 else y0 - height
            bbox = [min(x0, x2), min(y0, y2), max(x0, x2), max(y0, y2)]
            draw.rectangle(bbox, outline=255, width=1)
        elif mode == "ellipse":
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            x2 = x0 + width if x1 >= x0 else x0 - width
            y2 = y0 + height if y1 >= y0 else y0 - height
            bbox = [min(x0, x2), min(y0, y2), max(x0, x2), max(y0, y2)]
            draw.ellipse(bbox, outline=255, width=1)
        elif mode == "triangle":
            width = max(4, abs(x1 - x0))
            height = max(4, abs(y1 - y0))

            if x1 >= x0:
                base_left = (x0, y0)
                base_right = (x0 + width, y0)
            else:
                base_left = (x0, y0)
                base_right = (x0 - width, y0)

            mid_base_x = (base_left[0] + base_right[0]) / 2
            tip_y = y0 - height
            tip_x = mid_base_x

            points = [
                base_left,
                base_right,
                (tip_x, tip_y),
            ]
            draw.polygon(points, outline=255, width=1)
        elif mode == "hexagon":
            radius = max(1, max(abs(x1 - x0), abs(y1 - y0)) // 2)
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            points = []
            for i in range(6):
                angle = -3.14159 / 2 + i * 2 * 3.14159 / 6
                points.append((cx + radius * 1.0 * 0.9 * __import__("math").cos(angle), cy + radius * 1.0 * 0.9 * __import__("math").sin(angle)))
            draw.polygon(points, outline=255, width=1)
        elif mode == "octagon":
            radius = max(1, max(abs(x1 - x0), abs(y1 - y0)) // 2)
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            points = []
            for i in range(8):
                angle = -3.14159 / 2 + i * 2 * 3.14159 / 8
                points.append((cx + radius * 1.0 * 0.9 * __import__("math").cos(angle), cy + radius * 1.0 * 0.9 * __import__("math").sin(angle)))
            draw.polygon(points, outline=255, width=1)
        elif mode == "pen":
            draw.line([start, end], fill=self.current_color, width=self.current_brush_size)

    def on_button_press(self, event):
        self.last_point = self._to_image_coords(event.x, event.y)
        self.shape_start = self.last_point

        if self.current_tool == "pen" and self.last_point is not None:
            self.draw.point(self.last_point, fill=self.current_color)
            self.update_canvas()

    def on_paint(self, event):
        x, y = self._to_image_coords(event.x, event.y)

        if self.current_tool == "pen" and self.last_point is not None:
            self.draw.line([self.last_point, (x, y)], fill=self.current_color, width=self.current_brush_size)
            self.last_point = (x, y)
            self.update_canvas()
            return

        if self.current_tool not in {"rectangle", "ellipse", "triangle", "hexagon", "octagon"}:
            return

        preview_image = self.image.copy()
        self._draw_shape_preview(preview_image, self.shape_start, (x, y), self.current_tool)
        self._show_image(preview_image)

    def on_button_release(self, event):
        if self.current_tool == "pen":
            self.last_point = None
            self.shape_start = None
            return

        if self.current_tool in {"rectangle", "ellipse", "triangle", "hexagon", "octagon"} and self.shape_start is not None:
            x, y = self._to_image_coords(event.x, event.y)
            self._draw_shape_preview(self.image, self.shape_start, (x, y), self.current_tool)
            self.update_canvas()
        self.last_point = None
        self.shape_start = None

    def _show_image(self, image):
        display_image = image.resize((CANVAS_SIZE, CANVAS_SIZE), resample=Image.NEAREST)
        self.photo_image = ImageTk.PhotoImage(display_image)
        self.canvas.itemconfig(self.canvas_image_id, image=self.photo_image)

    def update_canvas(self):
        self._show_image(self.image)

    def clear(self):
        self.image = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None
        self.shape_start = None
        self.update_canvas()
        self.info_label.config(text="Draw a clear, thin shape that resembles the input data. Then, click Predict.")

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
            text=f"Prediction: {pred_label}\nConfidence: {confidence:.1f}%\nProbabilities: {prob_text}",
            font=("Arial", 11, "bold"),
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
