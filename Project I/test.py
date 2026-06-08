# main.py
from model import CNN
from utils import load_data, evaluate, predict_image, confusion_matrix

if __name__ == "__main__":
    model = CNN()
    model.load("weights.npy")

    images, labels = load_data("data/labels.csv", "dataset")
    predict_image(model, "img_0.png")

    # predict_image(model, "my_shape.png")  # uncomment to test a single image