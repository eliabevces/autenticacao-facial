import cv2 as cv
import os

def detect_face(pasta_pessoas):

    faces = os.listdir(pasta_pessoas)

    for i in faces:
        images = [pasta_pessoas + i + '/' + j for j in os.listdir(pasta_pessoas + i)]

        for image in images:
            # Read image from your local file system
            original_image = cv.imread(image)

            # Convert color image to grayscale for Viola-Jones
            grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

            # Load the classifier and create a cascade object for face detection
            face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

            detected_faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
            
            # create directory for each person
            if not os.path.exists('data/detectadas/lfw/' + i):
                os.makedirs('data/detectadas/lfw/' + i)
            for (column, row, width, height) in detected_faces:
                cv.imwrite('data/detectadas/lfw/' + i + '/' + str(images.index(image)) + '.jpg', grayscale_image[row:row + height, column:column + width])

    print('Successfully saved')

        

