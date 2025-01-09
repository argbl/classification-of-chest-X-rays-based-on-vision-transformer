import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from transformers import ViTForImageClassification
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

try:
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(class_names)  
    )
    model.load_state_dict(torch.load('model.pth', map_location="cpu"), strict=False)
    model.eval() 
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:

        image = Image.open(file.stream).convert("RGB")
        image = transform(image).unsqueeze(0)  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
        model.to(device)

        with torch.no_grad():
            outputs = model(image).logits
            _, predicted_class = torch.max(outputs, 1)
            predicted_label = class_names[predicted_class.item()]

        return jsonify({'prediction': predicted_label})

    except UnidentifiedImageError:
        return jsonify({'error': 'Invalid image file'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
