# config.py
SHAPES      = ["circle", "ellipse", "square", "triangle", "rectangle", "hexagon", "octagon"]
LABEL_MAP   = {s: i for i, s in enumerate(SHAPES)}
NUM_CLASSES = len(SHAPES)
IMAGE_SIZE  = 64