from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

model = load_model('final_model_mangrove.h5')
app = Flask(__name__)

clas = ['Avicennia Alba', 'Rhizophora Apiculata', 'Sonneratia Alba']

def predict_image_class(img, model):
    try:
        img = img.convert('RGB')
        img = img.resize((224, 224)) 
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        pred = clas[predicted_class[0]]
        conf = np.max(prediction)
        return pred, str(conf)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html', error=False)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.files['file']
        img = Image.open(BytesIO(f.read()))
        preds, conf = predict_image_class(img, model)
        if preds:
            return jsonify({'predict': preds, 'conf':conf})
        else:
            return jsonify({'error': 'Unable to process the image. Please try again with a different image.'})
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({'error': 'Unable to process the image. Please try again with a different image.'})

if __name__ == '__main__':
    app.run(debug=True)
