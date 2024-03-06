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


train_path = 'data/detectadas/famososcor/'
data_path = "./data/famosos/"
results_file = 'resultadosFamososFacenet.txt'
compressed_file = 'FAMOSOS-COR-dataset.npz'

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
        # image_array = cv.cvtColor(pil_image, cv.COLOR_BGR2GRAY)
        image_array = np.array(pil_image, 'uint8')
        # Getting the label from the image
        imge = Image.fromarray(image_array)
        imge = imge.resize((160, 160))
        id = os.path.split(image_path)[0].split('/')[-1]
        # extract the face from the training image sample
        faces.append(image_array)
        ids.append(id)
    return faces, ids   

# detect_face(data_path, 'famososcor')


f = open(results_file, 'a')


# list = [os.path.join(train_path, f) for f in os.listdir(train_path)]
# faces_train = []
# labels_train = []
# faces_test = []
# labels_test = []

# for path in list:
#     i, l = get_images_and_label(path)
#     faces_train.extend(i[:int(len(i)*0.7)])
#     labels_train.extend(l[:int(len(l)*0.7)])
#     faces_test.extend(i[int(len(i)*0.7):])
#     labels_test.extend(l[int(len(l)*0.7):])

# print('Train: ', len(faces_train))
# f.write('Train: ' + str(len(faces_train)) + '\n')
# print('Test: ', len(faces_test))
# f.write('Test: ' + str(len(faces_test)) + '\n\n')


# faces_train = [cv.resize(face, (160, 160)) for face in faces_train]
# faces_test = [cv.resize(face, (160, 160)) for face in faces_test] 

# faces_train, labels_train = shuffle(faces_train, labels_train)

# labels_train = [int(i) for i in labels_train]
# labels_test = [int(i) for i in labels_test]
# faces_test, labels_test = shuffle(faces_test, labels_test)

# np.savez_compressed(compressed_file, faces_train, labels_train, faces_test, labels_test)
data = np.load(compressed_file)
faces_train, labels_train, faces_test, labels_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
model = load_model('facenet_keras.h5')
def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]
    
# convert each face in the train set into embedding
emdTrainX = []
for face in faces_train:
    emd = get_embedding(model, face)
    emdTrainX.append(emd)
    
emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)

# convert each face in the test set into embedding
emdTestX = []
for face in faces_test:
    emd = get_embedding(model, face)
    emdTestX.append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)

in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)

out_encoder = LabelEncoder()
out_encoder.fit(labels_train)
trainy_enc = out_encoder.transform(labels_train)
testy_enc = out_encoder.transform(labels_test)

model = SVC(kernel='linear', probability=True)

model.fit(emdTrainX_norm, trainy_enc)

#predict\
yhat_train = model.predict(emdTrainX_norm)
yhat_test = model.predict(emdTestX_norm)

#score
score_train = accuracy_score(trainy_enc, yhat_train)
score_test = accuracy_score(testy_enc, yhat_test)

#summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
f.write('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100) + '\n')

print(f1_score(testy_enc, yhat_test, average="macro"))
f.write('F1: ' + str(f1_score(testy_enc, yhat_test, average="macro")) + '\n')
print(precision_score(testy_enc, yhat_test, average="macro"))
f.write('Precision: ' + str(precision_score(testy_enc, yhat_test, average="macro")) + '\n')
print(recall_score(testy_enc, yhat_test, average="macro"))
f.write('Recall: ' + str(recall_score(testy_enc, yhat_test, average="macro")) + '\n')


from random import choice
# select a random face from test set
selection = choice([i for i in range(faces_test.shape[0])])
random_face = faces_test[selection]
random_face_emd = emdTestX_norm[selection]
random_face_class = testy_enc[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = np.expand_dims(random_face_emd, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
all_names = out_encoder.inverse_transform([0,1,2,3,4])
#print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Predicted: \n%s \n%s' % (all_names, yhat_prob[0]*100))
print('Expected: %s' % random_face_name[0])
# plot face
plt.imshow(random_face)
title = '%s (%.3f)' % (predict_names[0], class_probability)
plt.title(title)
plt.show()

f.close()