from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import io
import numpy
import mysql.connector as mariadb

camera = PiCamera()
stream = io.BytesIO()
camera.resolution = (320, 240)
camera.framerate = 30
Raw_Capture = PiRGBArray(camera, size=(320, 240))


def insertorUpdate(Id,Name):
    mariadb_connection=mariadb.connect(user='root',
                                       password='root',
                                       database='Face_Info')
    cursor = mariadb_connection.cursor()
    cmd="SELECT * FROM Face_Information WHERE ID="+str(Id)
    cursor.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE Face_Information SET Name="+str(Name)+" WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO Face_Information(ID,Name) VALUES("+str(Id)+","+str(Name)+")"
    cursor.execute(cmd)
    mariadb_connection.commit()
    mariadb_connection.close()



id=int(input('Enter your id:'))
name=input('Enter your name:')

insertorUpdate(id,name)

Display_Window = cv2.namedWindow("Faces")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera.capture(stream, format='bgr')
detector=cv2.CascadeClassifier('Classifiers/face.xml')
time.sleep(1)

buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)
image1 = cv2.imdecode(buff, 1)
SampleNo=0

for frame in camera.capture_continuous(Raw_Capture, format="bgr", use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        SampleNo=SampleNo+1
        cv2.imwrite("dataset/face."+ str(id) +"."+str(SampleNo)+ ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Facess",image)
    cv2.imshow("Faces", image)
    key = cv2.waitKey(1)
    Raw_Capture.truncate(0)
    if(SampleNo>20):
        print("sample has been completed")
        break
        
        
camera.close()
cv2.destroyAllWindows()
        
    


