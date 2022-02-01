import os
import io
import numpy as np
import onnxruntime
from PIL import Image
import json

# Imports for the REST API
from flask import Flask, request, jsonify

# Imports for image procesing
from PIL import Image

app = Flask(__name__)

onnx_model_path = './model.onnx'

labels_file = './labels.json'
with open(labels_file) as f:
    classes = json.load(f)
print(classes)

try:
    session = onnxruntime.InferenceSession(onnx_model_path)
    print("ONNX model loaded...")
except Exception as e: 
    print("Error loading ONNX file: ",str(e))

# 4MB Max image size limit
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 

# Default route just shows simple text
@app.route('/')
def index():
    return 'AutoML model host harness'

# Like the CustomVision.ai Prediction service /image route handles either
#     - octet-stream image file 
#     - a multipart/form-data with files in the imageData parameter
@app.route('/image', methods=['POST'])
def predict_image_handler(project=None, publishedName=None):
    try:
        imageData = None
        if ('imageData' in request.files):
            imageData = request.files['imageData']
        elif ('imageData' in request.form):
            imageData = request.form['imageData']
        else:
            imageData = io.BytesIO(request.get_data())

        img = Image.open(imageData)
        results = predict_image(img)
        return jsonify(results)
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image', 500


def preprocess(image, resize_size, crop_size_onnx):

    image = image.convert('RGB')
    # resize
    image = image.resize((resize_size, resize_size))

    # center crop
    left = (resize_size - crop_size_onnx)/2
    top = (resize_size - crop_size_onnx)/2
    right = (resize_size + crop_size_onnx)/2
    bottom = (resize_size + crop_size_onnx)/2
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image)

    # HWC -> CHW
    np_image = np_image.transpose(2, 0, 1)# CxHxW

    # normalize the image
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(np_image.shape).astype('float32')
    for i in range(np_image.shape[0]):
        norm_img_data[i,:,:] = (np_image[i,:,:]/255 - mean_vec[i]) / std_vec[i]
    np_image = np.expand_dims(norm_img_data, axis=0)# 1xCxHxW
    return np_image


def get_predictions_from_ONNX(onnx_session,img_data):

    sess_input = onnx_session.get_inputs()
    sess_output = onnx_session.get_outputs()
    # predict with ONNX Runtime
    output_names = [ output.name for output in sess_output]
    scores = onnx_session.run(output_names=output_names, input_feed={sess_input[0].name: img_data})
    return scores[0]


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def predict_image(img):
    # you can modify resize_size based on your trained model
    resize_size = 256 
    height_onnx_crop_size = 224

    # height and width will be the same for classification
    crop_size_onnx = height_onnx_crop_size 
    img_data = preprocess(img, resize_size, crop_size_onnx)  # (1, 3, 224, 224)

    scores = get_predictions_from_ONNX(session, img_data)

    conf_scores = softmax(scores).tolist()
    print(conf_scores)

    class_idx = np.argmax(conf_scores)
    #print("predicted class:",(class_idx, classes[class_idx]))

    response = {
        "label": classes[class_idx],
        "probability": conf_scores[class_idx] 
    }
    print("Results: " + str(response))
    return response

if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=80)

