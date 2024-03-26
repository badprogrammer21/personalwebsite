from django.shortcuts import render
import numpy as np
import base64
import json
from django.http import JsonResponse
from PIL import Image
import io
from torchvision import models
import cv2
import os
import torch
from django.http import HttpResponse
from PIL import Image
from torchvision import transforms

RESCALE_SIZE = 224

def simpsons(request):
    
    return render(request, 'simpsons.html', {

    }) 


def predict_one_sample(model, inputs, device="cpu"):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs

def predict(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            im_val = Image.open(image_file)

            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, 'simpsons_model.pth') 

            device = torch.device('cpu')
            print(device)
            model = models.resnet50(pretrained=True)
            num_features = 2048
            model.fc = torch.nn.Linear(num_features, 42)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("The model is loaded....")
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
            im_val.load()
            im_val = np.array(im_val.resize((RESCALE_SIZE, RESCALE_SIZE)))

            im_val = np.array(im_val / 255, dtype='float32')
            im_val = transform(im_val)
            print("Transforming the picture...")
            prob_pred = predict_one_sample(model, im_val.unsqueeze(0))
            predicted_proba = np.max(prob_pred)*100
            y_pred = np.argmax(prob_pred)
            prediction = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'carl_carlson',\
                           'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'disco_stu', 'edna_krabappel', 'fat_tony', \
                            'gil', 'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lionel_hutz', 'lisa_simpson',\
                             'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover', 'moe_szyslak',\
                            'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner', 'professor_john_frink',\
                            'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel', 'snake_jailbird', \
                            'troy_mcclure', 'waylon_smithers'][y_pred]

            prediction = " ".join(map(lambda x: x.capitalize(), prediction.split('_')))

            return JsonResponse({'prediction': prediction + " " + str(int(np.max(prob_pred)*100)) + "%", 'status': 'success'})
        
        else:
            return JsonResponse({'message': 'No image provided', 'status': 'error'})

    else:
        return JsonResponse({'message': 'Invalid request method', 'status': 'error'})
    