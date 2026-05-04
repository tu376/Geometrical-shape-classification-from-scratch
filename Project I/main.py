import numpy as np
import csv
import os
from PIL import Image
from models.model import CNN, SHAPES

# ========================
# CẤU HÌNH
# ========================
DATASET_DIR = "dataset"
CSV_FILE = "data/labels.csv"
WEIGHTS_FILE = "weights.npy"
EPOCHS = 10
LEARNING_RATE = 0.01

# ========================
# ĐỌC DỮ LIỆU
# ========================
def load_data(csv_file, dataset_dir):
    images = []
    labels = []

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(dataset_dir, row["filename"])
            label = SHAPES.index(row["label"])

            img = Image.open(img_path).convert("L")
            img = np.array(img) / 255.0           # Normalize [0, 1]
            img = img.reshape(1, 64, 64)           # (1, H, W)

            images.append(img)
            labels.append(label)

    return images, labels

# ========================
# TRAIN
# ========================
def train(model, images, labels):
    n = len(images)
    for epoch in range(EPOCHS):
        # Shuffle dữ liệu mỗi epoch
        indices = np.random.permutation(n)
        total_loss = 0
        correct = 0

        for idx in indices:
            img = images[idx]
            label = labels[idx]

            # Forward
            loss = model.forward(img, label)
            total_loss += loss

            # Backward
            model.backward()

            # Tính accuracy
            pred, _ = model.predict(img)
            if pred == label:
                correct += 1

        avg_loss = total_loss / n
        accuracy = correct / n * 100
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    model.save(WEIGHTS_FILE)

# ========================
# PREDICT ẢNH MỚI
# ========================
def predict_image(model, image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = img.reshape(1, 64, 64)

    pred_idx, probs = model.predict(img)
    pred_shape = SHAPES[pred_idx]

    print(f"\nKết quả dự đoán: {pred_shape} ({probs[pred_idx]*100:.2f}%)")
    print("Xác suất các lớp:")
    for i, shape in enumerate(SHAPES):
        print(f"  {shape:12s}: {probs[i]*100:.2f}%")

    return pred_shape

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    model = CNN(learning_rate=LEARNING_RATE)

    print("=== LOADING DATA ===")
    images, labels = load_data(CSV_FILE, DATASET_DIR)
    print(f"Tổng số ảnh: {len(images)}")

    # Nếu đã có weights thì hỏi có muốn train lại không
    if os.path.exists(WEIGHTS_FILE):
        choice = input(f"\nĐã tìm thấy '{WEIGHTS_FILE}'. Bạn có muốn train lại không? (y/n): ")
        if choice.lower() == "y":
            print("\n=== BẮT ĐẦU TRAIN ===")
            train(model, images, labels)
        else:
            model.load(WEIGHTS_FILE)
    else:
        print("\n=== BẮT ĐẦU TRAIN ===")
        train(model, images, labels)

    # Predict ảnh mới (tùy chọn)
    print("\n=== PREDICT ẢNH MỚI ===")
    img_path = input("Nhập đường dẫn ảnh cần dự đoán (Enter để bỏ qua): ").strip()
    if img_path and os.path.exists(img_path):
        predict_image(model, img_path)
    elif img_path:
        print("Không tìm thấy file ảnh!")