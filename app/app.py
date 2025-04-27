from flask import Flask, request, render_template, jsonify,send_from_directory  
from werkzeug.utils import secure_filename
import os
from PIL import Image
import pytesseract
import pyttsx3
import torch
from torchvision import transforms
from torchvision.models import resnet18
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model for image classification
model = resnet18(pretrained=True)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load labels for ImageNet
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Function for text-to-speech
def text_to_speech(text):
    engine = pyttsx3.init()
    audio_path = './uploads/output_audio.mp3'
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path
# Function for image prediction
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
    return labels[predicted.item()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded!"})

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file!"})
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform image classification
        classification = predict_image(filepath)

        # Perform OCR
        extracted_text = pytesseract.image_to_string(Image.open(filepath))
        
        # Combine results
        result_text = f"Image Content: {classification}"
        if extracted_text.strip():
            result_text += f"\nExtracted Text: {extracted_text.strip()}"
        # Generate audio
        audio_path = text_to_speech(result_text)

        return render_template('result.html', classification=classification, extracted_text=extracted_text.strip(), audio_path=audio_path)

    return render_template('index.html')



@app.route('/uploads/<filename>')
def upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == "__main__":
    app.run(debug=True)
