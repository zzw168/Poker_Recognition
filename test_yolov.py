from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("best.pt")

# Open an image using PIL
source = Image.open("1.jpg")

# Run inference on the source
results = model(source)  # list of Results objects