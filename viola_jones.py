import cv2 as cv
import os

def detect_face(pasta_pessoas):

    dir_treino = pasta_pessoas + '/treino/'
    dir_teste = pasta_pessoas + '/teste/'
    faces_treino = os.listdir(dir_treino)
    faces_teste = os.listdir(dir_teste)

    for i in faces_treino:
        images = [dir_treino + i + '/' + j for j in os.listdir(dir_treino + i)]

        for image in images:
            # Read image from your local file system
            original_image = cv.imread(image)

            # Convert color image to grayscale for Viola-Jones
            grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

            # Load the classifier and create a cascade object for face detection
            face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

            detected_faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
            
            # create directory for each person
            if not os.path.exists('data/detectadas/treino/' + i):
                os.makedirs('data/detectadas/treino/' + i)
            for (column, row, width, height) in detected_faces:
                cv.imwrite('data/detectadas/treino/' + i + '/' + str(images.index(image)) + '.jpg', grayscale_image[row:row + height, column:column + width])

    for i in faces_teste:
        images = [dir_teste + i + '/' + j for j in os.listdir(dir_teste + i)]

        for image in images:
            # Read image from your local file system
            original_image = cv.imread(image)

            # Convert color image to grayscale for Viola-Jones
            grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

            # Load the classifier and create a cascade object for face detection
            face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

            detected_faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
            
            # create directory for each person
            if not os.path.exists('data/detectadas/teste/' + i):
                os.makedirs('data/detectadas/teste/' + i)
            for (column, row, width, height) in detected_faces:
                cv.imwrite('data/detectadas/teste/' + i + '/' + str(images.index(image)) + '.jpg', grayscale_image[row:row + height, column:column + width])

    print('Successfully saved')

        

