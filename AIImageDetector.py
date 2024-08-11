import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class AIImageDetector(nn.Module):
    def __init__(self):
        super(AIImageDetector, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)

def analyze_frequency_domain(img_path):
    img = cv2.imread(img_path, 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum

def detect_ai_generated(img_path, model):
    # Deep learning-based detection
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        prediction = model(img_tensor).item()
    
    # Frequency domain analysis
    freq_spectrum = analyze_frequency_domain(img_path)
    freq_features = extract_frequency_features(freq_spectrum)
    
    # Metadata analysis
    metadata = extract_metadata(img_path)
    
    # Combine all features and make final decision
    final_score = combine_features(prediction, freq_features, metadata)
    
    is_ai_generated = final_score > 0.5
    confidence = abs(final_score - 0.5) * 200  # Scale to 0-100%
    
    return is_ai_generated, confidence, final_score

# Additional helper functions (not implemented here):
# def extract_frequency_features(freq_spectrum):
# def extract_metadata(img_path):
# def combine_features(dl_prediction, freq_features, metadata):

def main():
    model = AIImageDetector()
    # Assume model is trained and weights are loaded
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    
    img_path = "path_to_image.jpg"
    is_ai_generated, confidence, score = detect_ai_generated(img_path, model)
    
    print(f"AI-generated: {is_ai_generated}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Score: {score:.4f}")

if __name__ == "__main__":
    main()