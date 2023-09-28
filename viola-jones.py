import cv2 as cv

# Read image from your local file system
original_image = cv.imread('data/test_img.jpg')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# Load the classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

detected_faces = face_cascade.detectMultiScale(grayscale_image)

for (column, row, width, height) in detected_faces:
    cv.imwrite('data/detected_faces/detectedface' + str(row) + '.jpg', original_image[row:row + height, column:column + width])

    
