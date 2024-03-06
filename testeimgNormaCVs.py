# Description: Main file for face recognition
import os
import sys
from viola_jones import detect_face
import numpy as np
import cv2 as cv
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve,accuracy_score,f1_score,precision_score,recall_score
import mtcnn
import pandas as pd
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import RocCurveDisplay

from sklearn.preprocessing import Normalizer


train_path = 'data/detectadas/lfw/'
data_path = "./data/lfw/"
results_file = 'resultadosLFWCVsNorma.txt'
compressed_file = 'LFW-dataset-grey.npz'

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
        image_array = cv.cvtColor(pil_image, cv.COLOR_BGR2GRAY)
        image_array = np.array(image_array, 'uint8')
        # Getting the label from the image
        imge = Image.fromarray(image_array)
        imge = imge.resize((160, 160))
        id = os.path.split(image_path)[0].split('/')[-1]
        # extract the face from the training image sample
        faces.append(image_array)
        ids.append(id)
    return faces, ids   

# detect_face(data_path, 'famosos')


f = open(results_file, 'a')


list = [os.path.join(train_path, f) for f in os.listdir(train_path)]
faces_train = []
labels_train = []
faces_test = []
labels_test = []

for path in list:
    i, l = get_images_and_label(path)
    faces_train.extend(i[:int(len(i)*0.7)])
    labels_train.extend(l[:int(len(l)*0.7)])
    faces_test.extend(i[int(len(i)*0.7):])
    labels_test.extend(l[int(len(l)*0.7):])

print('Train: ', len(faces_train))
f.write('Train: ' + str(len(faces_train)) + '\n')
print('Test: ', len(faces_test))
f.write('Test: ' + str(len(faces_test)) + '\n\n')


faces_train = [cv.resize(face, (160, 160)) for face in faces_train]
faces_test = [cv.resize(face, (160, 160)) for face in faces_test] 

faces_train, labels_train = shuffle(faces_train, labels_train)

labels_train = [int(i) for i in labels_train]
labels_test = [int(i) for i in labels_test]
faces_test, labels_test = shuffle(faces_test, labels_test)

np.savez_compressed(compressed_file, faces_train, labels_train, faces_test, labels_test)
data = np.load(compressed_file)
faces_train, labels_train, faces_test, labels_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
recognizerEigenface = cv.face.EigenFaceRecognizer_create(80)
recognizerFisherface = cv.face.FisherFaceRecognizer_create(80)
recognizerLBPH = cv.face.LBPHFaceRecognizer_create(10, 14, 12, 12)

# train
print('LBPH')
f.write('LBPH' + '\n')
start = datetime.now()
recognizerLBPH.train(faces_train, np.array(labels_train))
# recognizerLBPH.read('models/recognizerLBPH.yml')
print("time: ", datetime.now() - start)
f.write("time: " + str(datetime.now() - start) + '\n\n')

# test
total = []
correct = []

for i in range(len(faces_test)):
    label, confidence = recognizerLBPH.predict(faces_test[i])
    for k in range(10):
        if confidence > k*10:
            if len(total) < k+1:
                total.append(0)
                correct.append(0)
            if label == labels_test[i]:
                correct[k] += 1
            total[k] += 1

for j in range(10):
    try:
        print('LBPH ', j*10, ' accuracy: ', correct[j]/total[j], "total: ", total[j])
        f.write('LBPH ' + str(j*10) + ' accuracy: ' + str(correct[j]/total[j]) + "total: " + str(total[j]) + '\n')
    except:
        print('LBPH ', j*10, ' accuracy: 0')
        f.write('LBPH ' + str(j*10) + ' accuracy: 0' + '\n')

f.write('\n')


# test
total = []
correct = []
# train
print('Training models...')
print('Eigenface')
f.write('Eigenface' + '\n')
start = datetime.now()
recognizerEigenface.train(faces_train, np.array(labels_train))
# recognizerEigenface.read('models/recognizerEigenface.yml')
print("time: ", datetime.now() - start)
f.write("time: " + str(datetime.now() - start) + '\n\n')

for i in range(len(faces_test)):
    label, confidence = recognizerEigenface.predict(faces_test[i])
    for k in range(10):
        if confidence > k*10:
            if len(total) < k+1:
                total.append(0)
                correct.append(0)
            if label == labels_test[i]:
                correct[k] += 1
            total[k] += 1

for j in range(10):
    try:
        print('Eigenface ', j*10, ' accuracy: ', correct[j]/total[j], "total: ", total[j])
        f.write('Eigenface ' + str(j*10) + ' accuracy: ' + str(correct[j]/total[j]) + "total: " + str(total[j]) + '\n')
    except:
        print('Eigenface ', j*10, ' accuracy: 0')
        f.write('Eigenface ' + str(j*10) + ' accuracy: 0' + '\n')

f.write('\n')

# test
total = []
correct = []
# train
print('Fisherface')
f.write('Fisherface' + '\n')
start = datetime.now()
recognizerFisherface.train(faces_train, np.array(labels_train))
# recognizerFisherface.read('models/recognizerFisherface.yml')
print("time: ", datetime.now() - start)
f.write("time: " + str(datetime.now() - start) + '\n\n')

for i in range(len(faces_test)):
    label, confidence = recognizerFisherface.predict(faces_test[i])
    for k in range(10):
        if confidence > k*10:
            if len(total) < k+1:
                total.append(0)
                correct.append(0)
            if label == labels_test[i]:
                correct[k] += 1
            total[k] += 1

for j in range(10):
    try:
        print('Fisherface ', j*10, ' accuracy: ', correct[j]/total[j], "total: ", total[j])
        f.write('Fisherface ' + str(j*10) + ' accuracy: ' + str(correct[j]/total[j]) + "total: " + str(total[j]) + '\n')
    except:
        print('Fisherface ', j*10, ' accuracy: 0')
        f.write('Fisherface ' + str(j*10) + ' accuracy: 0' + '\n')


f.close()
