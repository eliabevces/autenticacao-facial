# Description: Main file for face recognition
import os
import sys
from viola_jones import detect_face
import numpy as np
import cv2 as cv



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

train_path = 'data/detectadas/treino/'
test_path = 'data/detectadas/teste/'

faces = []
labels = []
for path in [os.path.join(train_path, f) for f in os.listdir(train_path)]:
    i, l = get_images_and_label(path)
    faces.extend(i)
    labels.extend(l)

faces = [cv.resize(face, (200, 200)) for face in faces]

# recognizer = cv.face.EigenFaceRecognizer_create()
recognizer = cv.face.FisherFaceRecognizer_create()
# recognizer = cv.face.LBPHFaceRecognizer_create()

labels = [int(i) for i in labels]
recognizer.train(faces, np.array(labels))

for path in [os.path.join(test_path, f) for f in os.listdir(test_path)]:
    i, l = get_images_and_label(path)
    for image in i:
        image = cv.resize(image, (200, 200))

        print(l[0])
        label, confidence = recognizer.predict(image)
        print(f'Label: {label} with confidence: {confidence}')
        cv.imshow(f'{label}', image)
        print( str(label) + ' ' + l[0])

        cv.waitKey(0)

        if str(label) == l[0]:
            print('Correct!')
        else:
            print('Incorrect!')

