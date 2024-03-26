from django.shortcuts import render
import numpy as np
import base64
import json
from django.http import JsonResponse
from PIL import Image
import io
import cv2
import os

def ReLU(Z):
    return np.maximum(Z,0)

def deriv_of_ReLU(Z):
    return Z > 0

def derivative_leaky_ReLU(Z, alpha=0.15):
    return np.where(Z > 0, 1, alpha)

def leaky_ReLU(Z, alpha=0.15):
    return np.where(Z > 0, Z, alpha * Z)

def softMax(Z):
    """
    The change ensures numerical stability by subtracting the maximum value from Z before exponentiating
    """
    e = np.exp(Z - np.max(Z))
    return e / e.sum(axis=0)

def forward_prop(X, W1, b1, W2, b2, W3, b3, W4, b4):
    Z1 = W1.dot(X) + b1
    A1 = leaky_ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = leaky_ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = leaky_ReLU(Z3)
    Z4 = W4.dot(A3) + b4
    A4 = softMax(Z4)

    return Z1, A1, Z2, A2, Z3, A3, Z4, A4

def get_predictions(A4):
    return np.argmax(A4, 0)

def make_predictions(X, W1 ,b1, W2, b2, W3, b3, W4, b4):
    _, _, _, _, _, _, _, A4 = forward_prop(X, W1, b1, W2, b2, W3, b3, W4, b4)
    predictions = get_predictions(A4)
    return predictions

def load_model():
    try:
        print("Loading the model...")
        base_dir = os.path.dirname(__file__)  # Gets the directory of the current file
        model_path = os.path.join(base_dir, 'model_parameters.npz')  # Adjusts the path to the model file
        data = np.load(model_path)
        print("Loaded the model successfully...")
        return data['W1'], data['b1'], data['W2'], data['b2'], data['W3'], data['b3'], data['W4'], data['b4']
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle the exception or re-raise it
        raise


def mnistrecognizer(request):
    return render(request, 'mnistrecognizer.html', {}) 

def predict(request):
    print('Calling the prediction function in mnistrecognizer app...')
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            image_bytes = image_file.read()
            image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            img_reshaped = img.reshape((784, 1))
            W1, b1, W2, b2, W3, b3, W4, b4 = load_model()

            prediction = make_predictions(img_reshaped, W1, b1, W2, b2, W3, b3, W4, b4)
            print(img_reshaped)
            return JsonResponse({'prediction': prediction.tolist(), 'status': 'success'})
        
        else:
            return JsonResponse({'message': 'No image provided', 'status': 'error'})

    else:
        return JsonResponse({'message': 'Invalid request method', 'status': 'error'})