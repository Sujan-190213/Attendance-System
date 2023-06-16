import cv2
import numpy as np
import face_recognition

# load first image
imgElon = face_recognition.load_image_file('ImagesAttendance/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

# load first image
imgTest = face_recognition.load_image_file('ImagesAttendance/Elon Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# detecting first image face in rectangular box
faceLoc = face_recognition.face_locations(imgElon)[0]   # face_locations returns 4 values [ top, right, bottom, left]
encodeElon = face_recognition.face_encodings(imgElon)[0]  # gives 128 measurement of an image
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)  # { faceLoc[3] => x1, faceLoc[0] => y1 }, { faceLoc[1] => x2, faceLoc[2] => y2 }

# detecting second image face in rectangular box
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)





# compare two ImagesAttendance
results = face_recognition.compare_faces([encodeElon],encodeTest)  # function first argument => list of none faces, second argument => test face

# sometimes we found similarity in two different images ( from 128 measurement ). So for this we need region distance (face)
# face distance between two ImagesAttendance
faceDis = face_recognition.face_distance([encodeElon],encodeTest)  # lower the distance better the match is
print(results,faceDis)

cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)




# show two ImagesAttendance
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)

cv2.waitKey(0)  # distroy windwo to press any key
