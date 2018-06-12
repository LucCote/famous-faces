from flask import Flask, request, render_template, send_file
import base64
import io
from PIL import Image
import numpy as np
import face_recognition


app = Flask(__name__) # Initialize Flask and serve static folder
guess = "none"
known_face_encodings = np.loadtxt('encodings.txt')
images = ["billnye.jpeg", "janegoodall.jpeg", "neiltyson.jpg", "sallyride.jpg", "sagan.jpg", "elonmusk.jpg", "jobsGood.jpg", "curieGood.jpg", "michiokaku.jpg", "swGood.jpg", "billgates.jpg", "byron.jpg", "takahashi.jpg", "jemison.jpg", "stofan.jpg"]


def getbestindex(w):
    # returns the index of the lowest value of an array
    value = 0
    for i in range(1, len(w)):
        if w[i] < w[value]:
            value = i
    return value


def getmatch(img):
    # returns the url of closest scientist to image's face
    unknown_image = img
    unknown_encoding_array = face_recognition.face_encodings(unknown_image)
    if len(unknown_encoding_array) > 0:
        unknown_encoding = unknown_encoding_array[0]
        results = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        guessindex = getbestindex(results)
        global guess
        return images[guessindex]
    else:
        return "noface.jpg"


@app.route('/')
def root():
    # Serve index.html under /
    return render_template("index.html")


def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)
    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


@app.route("/findface", methods=["POST"])
def findface():
    start = request.form['image'].index(',')
    imagestr = request.form['image'][start:]
    image_bytes = io.BytesIO(base64.decodebytes(imagestr.encode('utf-8')))
    im = Image.open(image_bytes)
    im = pure_pil_alpha_to_color_v2(im)
    arr = np.array(im)
    return getmatch(arr)
