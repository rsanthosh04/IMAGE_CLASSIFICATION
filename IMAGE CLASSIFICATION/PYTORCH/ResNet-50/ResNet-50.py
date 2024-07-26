import torch
from torchvision import models, transforms
from PIL import Image
import requests
import time

# Load the pre-trained ResNet50 model
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
model.eval()

# Define the preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the input image
image_path = "C:\Users\santh\OneDrive\Pictures\Screenshots.jpeg"
img = Image.open(image_path)

# Preprocess the input image
image_t = preprocess(img)

# Prepare the input for the model
batch_t = torch.unsqueeze(image_t, 0)

# Record the start time
start_time = time.time()

# Make predictions
with torch.no_grad():
    output = model(batch_t)

# Record the end time
end_time = time.time()

# Calculate the inference time
inference_time = end_time - start_time

# Get the predicted class
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()
_, index = torch.max(output, 1)
predicted_class = labels[index[0]]

print(f"Predicted class: {predicted_class}")
print(f"Inference time: {inference_time:.4f} seconds")
