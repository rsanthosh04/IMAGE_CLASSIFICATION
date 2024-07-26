import torch
from torchvision import models, transforms
from PIL import Image
import requests
import cv2
import numpy as np

# Load the pre-trained model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Function to predict the image class
def predict_image_class(image):
    image_t = preprocess(image)
    batch_t = torch.unsqueeze(image_t, 0)
    with torch.no_grad():
        output = model(batch_t)
    _, index = torch.max(output, 1)
    predicted_class = labels[index[0]]
    return predicted_class

# Open the video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
        break
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    predicted_class = predict_image_class(image)
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

