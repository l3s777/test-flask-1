# encoding: utf-8

import io
import json

import os


from flask import Flask, jsonify, abort, request

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from torch.autograd import Variable
from torchvision.models import squeezenet1_1

from PIL import Image

import requests
from io import BytesIO
from pathlib import Path

import numpy as np

import base64

import re


# Initialize our Flask application and the PyTorch model.
app = Flask(__name__)
model = None
use_gpu = True


def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model
    model = squeezenet1_1(pretrained=True)
    model.eval()

def process_image(data):

    res = {"success": False}

    print("Prediction in progress")

    buff = BytesIO(base64.b64decode(data, ' /'))
    image = Image.open(buff)

    # image = prepare_image(im, target_size=(224, 224))

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)

    index = output.data.numpy().argmax()
    print(index)

    index_file = "class_index_map.json"
    indexpath = os.path.join(os.getcwd(), index_file)
    print(indexpath)

    class_map = json.loads(open(indexpath).read())
    prediction = class_map[str(index)][1]

    return prediction


@app.route("/predict", methods=["POST"])
def predict():

    result = {"res": None}

    data = re.sub('^data:image/.+;base64,', '', request.args.get("data"))
    pred = process_image(data)

    if (len(pred.split("_")) == 1 ):
            result = {"label": pred}
    else:
            result = {"label": pred.split("_")[1], "type": pred.split("_")[0]}

    return jsonify(result)


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run()
