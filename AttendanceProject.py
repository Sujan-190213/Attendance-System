import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'ImagesAttendance'
images = []  # list of all image
classNames = []  # list of all image name
myList = os.listdir(path)  # image list in 'ImagesAttendance' directory
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # spliting pair => [0] means root and [1] means jpg or jpeg
print(classNames)


# Encoding of images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Mark Attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()  # read all line that are currently present in csv file ( reason: if somebody arrive, we ignore it )
        nameList = []
        for line in myDataList:
            entry = line.split(',')  # split name and time separately
            nameList.append(entry[0]) # name is the first element of the list, second element time
        # non-repeated parson (check)
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)  # encoding for known faces
print('Encoding Complete') # it takes times so for ensuring complition we print a message

cap = cv2.VideoCapture(0)  # takes image from webcam

while True:
    success, img = cap.read()   # iamge data
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing(small)  to one-fourth of the original image for better performance
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # calculate face-locations and face-encodings
    facesCurFrame = face_recognition.face_locations(imgS)  # face_locations returns 4 values [ top, right, bottom, left]
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)   # gives 128 measurement of resize image along with location of the faces

# Finding the matches
    # grab one facelocation from facescurrentFrame list then grab one encodeface from encodescurrentFrame
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # array of face distance
        # print(faceDis)
        matchIndex = np.argmin(faceDis)  # finding the lowest face distance and its return the index value

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)

            # draw rectangle in faces area
            y1, x2, y2, x1 = faceLoc   # [ top, right, bottom, left ]
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # resizing one-fourth * 4 = original shape
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # entry the parson name with time in csv file
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# stop webcam through esc key
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
