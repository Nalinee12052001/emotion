#uploadajax/main.py
from keras.models import load_model
from typing import Union
from keras.utils import load_img, img_to_array
from urllib import request
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi import FastAPI, Request, UploadFile, File,Form,Body
from fastapi.templating import Jinja2Templates
from IPython.display import HTML

from flask import Flask, render_template

import os,uuid,aiofiles,cv2,base64
import numpy as np
import matplotlib.pyplot as plt


IMAGEDIR = "images/"
 
app = FastAPI()
templates = Jinja2Templates(directory="templates")
 
@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})
 

def facecrop(image,image_name) :  
    facedata = 'haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)
    try:
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)

        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            sub_face = img[y:y+h, x:x+w]

            cv2.imwrite('crop/'+image_name, sub_face)
            #print ("Writing: " + image)

    except Exception as e:
        print (e)

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
    
@app.post("/upload")
async def upload_image(request: Request):
    data = await request.json()
    image_data = data['image']
    extension = data['extension'] 

    img_data = base64.b64decode(image_data)
    filename = str(uuid.uuid4()) + "." + extension

    with open(os.path.join("images", filename), "wb") as f:
        f.write(img_data)

    emotion_model = load_model('model.h5')
    file_original = f"{IMAGEDIR}{filename}"
    facecrop(file_original,filename)
    file_crop='crop/'+filename

    true_image = load_img(file_crop) 
    # print(file)
    img = load_img(file_crop, color_mode="grayscale", target_size=(48, 48))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    print(x.shape)

    objects = ('angry😈', 'disgust😖', 'fear😨', 'happy😁', 'sad😥', 'surprise😧', 'neutral😶')
    custom = emotion_model.predict(x)
    print(objects[np.argmax(custom[0])])
    print(emotion_analysis(custom[0]))

    return {"filename": upload_image,"class":objects[np.argmax(custom[0])]}








