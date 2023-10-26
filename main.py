# Description: Main file for face recognition
import os
import sys
from viola_jones import detect_face
import numpy as np
import cv2 as cv
from datetime import datetime



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

train_path = 'data/detectadas/lfw/'

faces_train = []
labels_train = []
faces_test = []
labels_test = []
for path in [os.path.join(train_path, f) for f in os.listdir(train_path)]:
    i, l = get_images_and_label(path)
    faces_train.extend(i[:int(len(i)*0.7)])
    labels_train.extend(l[:int(len(l)*0.7)])
    faces_test.extend(i[int(len(i)*0.7):])
    labels_test.extend(l[int(len(l)*0.7):])

print('Train: ', len(faces_train))
print('Test: ', len(faces_test))

faces_train = [cv.resize(face, (200, 200)) for face in faces_train]
faces_test = [cv.resize(face, (200, 200)) for face in faces_test]

labels_train = [int(i) for i in labels_train]
labels_test = [int(i) for i in labels_test]

recognizerEigenface = cv.face.EigenFaceRecognizer_create()
recognizerFisherface = cv.face.FisherFaceRecognizer_create()
recognizerLBPH = cv.face.LBPHFaceRecognizer_create()

# train
print('Training models...')
print('Eigenface')
start = datetime.now()
recognizerEigenface.train(faces_train, np.array(labels_train))
print("time: ", datetime.now() - start)
print('Fisherface')
start = datetime.now()
recognizerFisherface.train(faces_train, np.array(labels_train))
print("time: ", datetime.now() - start)
print('LBPH')
start = datetime.now()
recognizerLBPH.train(faces_train, np.array(labels_train))
print("time: ", datetime.now() - start)

# save models
recognizerEigenface.save('models/recognizerEigenface.yml')
recognizerFisherface.save('models/recognizerFisherface.yml')
recognizerLBPH.save('models/recognizerLBPH.yml')


# test
total = 0
correct = 0
for i in range(len(faces_test)):
    label, confidence = recognizerEigenface.predict(faces_test[i])
    if label == labels_test[i]:
        correct += 1
    total += 1
print('Eigenface: ', correct/total)

total = 0
correct = 0
for i in range(len(faces_test)):
    label, confidence = recognizerFisherface.predict(faces_test[i])
    if label == labels_test[i]:
        correct += 1
    total += 1
print('Fisherface: ', correct/total)

total = 0
correct = 0
for i in range(len(faces_test)):
    label, confidence = recognizerLBPH.predict(faces_test[i])
    if label == labels_test[i]:
        correct += 1
    total += 1
print('LBPH: ', correct/total)