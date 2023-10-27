# Description: Main file for face recognition
import os
import sys
from viola_jones import detect_face
import numpy as np
import cv2 as cv
from datetime import datetime
from sklearn.model_selection import KFold



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
        ids.append(int(id))
    return faces, ids   

train_path = 'data/detectadas/lfw/'

faces = []
labels = []
for path in [os.path.join(train_path, f) for f in os.listdir(train_path)]:
    i, l = get_images_and_label(path)
    faces.extend(i)
    labels.extend(l)

print('faces: ', len(faces))
print('labels: ', len(labels))


faces_train = []
labels_train = []
faces_test = []
labels_test = []

recognizerEigenface = cv.face.EigenFaceRecognizer_create()
recognizerFisherface = cv.face.FisherFaceRecognizer_create()
recognizerLBPH = cv.face.LBPHFaceRecognizer_create()

acertosEigenface = 0
acertosFisherface = 0
acertosLBPH = 0
testes_totais = 0
kf = KFold(n_splits=2, shuffle=True, random_state=42)
for train_index, test_index in kf.split(faces):
    faces_train = []
    labels_train = []
    faces_test = []
    labels_test = []
    faces_train.extend([cv.resize(faces[i], (200, 200)) for i in train_index])
    labels_train.extend([labels[i] for i in train_index])
    faces_test.extend([cv.resize(faces[i], (200, 200)) for i in test_index])
    labels_test.extend([labels[i] for i in test_index])

    print('faces_train: ', len(faces_train))
    print('faces_test: ', len(faces_test))


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

    # test
    total = 0
    correct = 0
    for i in range(len(faces_test)):
        label, confidence = recognizerEigenface.predict(faces_test[i])
        if label == labels_test[i]:
            correct += 1
            acertosEigenface += 1
        total += 1
    print('Eigenface: ', correct/total)

    total = 0
    correct = 0
    for i in range(len(faces_test)):
        label, confidence = recognizerFisherface.predict(faces_test[i])
        if label == labels_test[i]:
            correct += 1
            acertosFisherface += 1
        total += 1
    print('Fisherface: ', correct/total)

    total = 0
    correct = 0
    for i in range(len(faces_test)):
        label, confidence = recognizerLBPH.predict(faces_test[i])
        if label == labels_test[i]:
            correct += 1
            acertosLBPH += 1
        total += 1
    print('LBPH: ', correct/total)
    testes_totais += total

print('Eigenface: ', acertosEigenface/testes_totais)
print('Fisherface: ', acertosFisherface/testes_totais)
print('LBPH: ', acertosLBPH/testes_totais)



# save models
recognizerEigenface.save('models/recognizerEigenface.yml')
recognizerFisherface.save('models/recognizerFisherface.yml')
recognizerLBPH.save('models/recognizerLBPH.yml')


