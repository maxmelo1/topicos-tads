from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np

from joblib import load

model = load('svm_mnist_model.joblib')

app = Flask(__name__)
CORS(app)

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    
    image = ((np.array(image.resize((8,8))) / 255.0)*16).astype(np.uint8).astype(float)
    # print(image.reshape(1, -1).shape)
    print(image)

    # import matplotlib.pyplot as plt
    # plt.imshow(image, cmap='gray')
    # plt.show()
    
    return model.predict(image.reshape(1, -1))[0]

@app.route('/infer_img', methods=['POST'])
def process_image():
    try:
        image_bytes = request.files['image'].read()

        result = predict(image_bytes)
        # print(result)
        
        return jsonify({
            'digit': int(result),
            'status': 'success'
        })

    except Exception as e:
        print(e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)