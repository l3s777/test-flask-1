# encoding: utf-8

import io
import json

from flask import Flask, jsonify, abort, request

# import torch
# import torch.nn.functional as F
from PIL import Image
# from torch import nn
# from torchvision import transforms as T
# from torchvision.models import resnet50

# from torch.autograd import Variable
# import torchvision.models as models
# import torchvision.transforms as transforms

from io import BytesIO
from pathlib import Path

import numpy as np

import base64

import re

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf


# Initialize our Flask application and the PyTorch model.
app = Flask(__name__)
model = None
use_gpu = True


graph = tf.get_default_graph()
with graph.as_default():
    # load model at very first
   model = load_model('model_design.h5')

# call model to predict an image
def api(f1):
    # data = image.load_img(full_path, target_size=(150, 150, 3))
    buff = BytesIO(base64.b64decode(f1, ' /'))
    im = Image.open(buff)

    im = np.expand_dims(im, axis=0)
    im = im * 1.0 / 255

    with graph.as_default():
        predicted = model.predict(im)
        return predicted


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST'])
def upload_file():

    res = {"success": False}
    f1 = re.sub('^data:image/.+;base64,', '', request.args.get("data"))
    # file = request.files['image']
    # full_name = os.path.join(UPLOAD_FOLDER, file.filename)
    # file.save(full_name)

    indices = {0: 'Cat', 1: 'Dog'}
    result = api(f1)

    predicted_class = np.asscalar(np.argmax(result, axis=1))
    accuracy = round(result[0][predicted_class] * 100, 2)
    label = indices[predicted_class]

    print(label)
    print(accuracy)

    return jsonify(res)


    # return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)



# with open('imagenet_class.txt', 'r') as f:
#     idx2label = eval(f.read())


# def load_model():
#     """Load the pre-trained model, you can use your model just as easily.
#
#     """
#     global model
#     model = resnet50(pretrained=True)
#     model.eval()
#     # if use_gpu:
#     #     model.cuda()


# def prepare_image(image, target_size):
#     """Do image preprocessing before prediction on any data.
#
#     :param image:       original image
#     :param target_size: target image size
#     :return:
#                         preprocessed image
#     """
#
#     if image.mode != 'RGB':
#         image = image.convert("RGB")
#
#     # Resize the input image nad preprocess it.
#     image = T.Resize(target_size)(image)
#     image = T.ToTensor()(image)
#
#     # Convert to Torch.Tensor and normalize.
#     image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
#
#     # Add batch_size axis.
#     image = image[None]
#     # if use_gpu:
#     #     image = image.cuda()
#     return torch.autograd.Variable(image, volatile=True)


# def process_image(data):
#
#     res = {"success": False}
#
#
#     buff = BytesIO(base64.b64decode(data, ' /'))
#     im = Image.open(buff)
#
#     image = prepare_image(im, target_size=(224, 224))
#
#     vgg = models.vgg16(pretrained=True)  # This may take a few minutes.
#
#     prediction = vgg(image)  # Returns a Tensor of shape (batch, num class labels)
#
#     prediction = prediction.data.numpy().argmax()  # Our prediction will be the index of the class label with the largest value.
#
#     print(prediction)  # Converts the index to a string using our labels dict
#     res["success"] = True
#     res["label"] = "gato"
#     res["type"] = "siames"
#     res["prob"] = 0.7
#     print(jsonify(res))



# @app.route("/predict", methods=["POST"])
# def predict():
#     # Initialize the data dictionary that will be returned from the view.
#     result = {"label": "gato", "type": "none", "prob": 0.7}
#
#     data = re.sub('^data:image/.+;base64,', '', request.args.get("data"))
#     process_image(data)
#
#     return jsonify(result)

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    # load_model()
    app.run()
