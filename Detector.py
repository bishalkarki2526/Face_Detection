from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
from PIL import ImageFont,ImageDraw,Image
import mysql.connector as mariadb
import numpy
from PIL import Image

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer//trainerData.yml")




display_window = cv2.namedWindow("Faces")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getProfile(id):
    mariadb_connection = mariadb.connect(user='root',
                                       password='root',
                                       database='Face_Info')
    cursor = mariadb_connection.cursor()
    cmd="SELECT * FROM Face_Information WHERE ID="+str(id)
    cursor.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    mariadb_connection.close()
    return profile

id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        
        cv2.putText(image,str(id),(x,y+h),font,1,(0,255,0))
        profile=getProfile(id)
        if(profile == None):
            cv2.putText(image,"Unknown",(x,y+h+25),font,1,(0,255,0))
            
        else:
            
            cv2.putText(image,str(profile[0]),(x,y+h),font,1,(0,255,0))
            cv2.putText(image,str(profile[1]),(x,y+h+25),font,1,(0,255,0))
            cv2.putText(image,str(profile[2]),(x,y+h+50),font,1,(0,255,0))
            cv2.putText(image,str(profile[3]),(x,y+h+75),font,1,(0,255,0))
    
    
    cv2.imshow("Faces", image)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if key == 15:
        camera.close()
        cv2.destroyAllWindows()
        break
