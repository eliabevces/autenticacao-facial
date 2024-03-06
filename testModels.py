# load model and test it
import os
import sys
from viola_jones import detect_face
import numpy as np
import cv2 as cv
from sklearn.metrics import accuracy_score


def get_images_and_label(path):
    # Get the path of all the files in the folder
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # Create empty face list
    faces = []
    # Create empty ID list
    ids = []
    # Looping through all the image paths and loading the Ids and the images
    for image_path in image_paths:
        # Loading the image and converting it to gray scale
        pil_image = cv.imread(image_path)
        # Now we are converting the PIL image into numpy array
        image_array = np.array(pil_image, 'uint8')
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
        # Getting the label from the image
        id = os.path.split(image_path)[0].split('/')[-1]
        # extract the face from the training image sample
        faces.append(image_array)
        ids.append(id)
    return faces, ids

detect_face('data/fotos')


faces_path = 'data/detectadas/treino/'


recognizerEigenface = cv.face.EigenFaceRecognizer_create()
recognizerEigenface.read('models/recognizerEigenface.yml')

recognizerFisherface = cv.face.FisherFaceRecognizer_create()
recognizerFisherface.read('models/recognizerFisherface.yml')

recognizerLBPH = cv.face.LBPHFaceRecognizer_create()
recognizerLBPH.read('models/recognizerLBPH.yml')


faces = []
labels = []
for path in [os.path.join(test_path, f) for f in os.listdir(test_path)]:
    i, l = get_images_and_label(path)
    faces.extend(i)
    labels.extend(l)


faces = [cv.resize(face, (200, 200)) for face in faces]

labels = [int(i) for i in labels]
predsEigenface = []
predsFisherface = []
predsLBPH = []

for face in faces:
    predEigenface = recognizerEigenface.predict(face)
    predsEigenface.append(predEigenface[0])
    predFisherface = recognizerFisherface.predict(face)
    predsFisherface.append(predFisherface[0])
    predLBPH = recognizerLBPH.predict(face)
    predsLBPH.append(predLBPH[0])

print('Eigenface accuracy: ', accuracy_score(labels, predsEigenface))
print('Fisherface accuracy: ', accuracy_score(labels, predsFisherface))
print('LBPH accuracy: ', accuracy_score(labels, predsLBPH))
