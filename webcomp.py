from flask import Flask, request, render_template, send_file
import base64
import io
from PIL import Image
import numpy as np
import face_recognition


app = Flask(__name__) # Initialize Flask and serve static folder
known_face_encodings = np.loadtxt('encodings.txt')
images = ["billnye.jpeg", "janegoodall.jpeg", "neiltyson.jpg", "sallyride.jpg", "sagan.jpg", "elonmusk.jpg", "jobsGood.jpg", "curieGood.jpg", "michiokaku.jpg", "swGood.jpg", "billgates.jpg", "byron.jpg", "takahashi.jpg", "jemison.jpg", "stofan.jpg"]


def getbestindex(w):
    # returns the index of the lowest value of an array
    value = 0
    for i in range(1, len(w)):
        if w[i] < w[value]:
            value = i
    return value


# returns the url of closest scientist to image's face
def getmatch(img):
    unknown_image = img
    unknown_encoding_array = face_recognition.face_encodings(unknown_image) # generate feature map of user's face
    if len(unknown_encoding_array) > 0:
        unknown_encoding = unknown_encoding_array[0] # we only encoded one image so we only need first result
        results = face_recognition.face_distance(known_face_encodings, unknown_encoding) # get distance from each scientists feature map to user's feature map
        guessindex = getbestindex(results) # get the index of the user's nearest neighbor scientist
        return images[guessindex] # return the path to the image representing user's famous scientists match
    else:
        return "noface.jpg"


@app.route('/')
def root():
    # Serve index.html under /
    return render_template("index.html")


# takes a width by height by 4 image and turns it to a width by height by 3 representation by removing alpha values
# Source: http://stackoverflow.com/a/9459208/284318
def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


@app.route("/findface", methods=["POST"])
def findface():
    start = request.form['image'].index(',') # find beggining of image representation
    imagestr = request.form['image'][start:] # get string representation of image
    image_bytes = io.BytesIO(base64.decodebytes(imagestr.encode('utf-8'))) # create bit buffer and have it hold image
    im = Image.open(image_bytes) # get PIL representation of image
    
    # shape image to be given to face-recognition
    im = pure_pil_alpha_to_color_v2(im) 
    arr = np.array(im) 

    return getmatch(arr) # return path to image representing user's scientist match
