import cv2
import os
import numpy as np
from PIL import Image 

recognizer = cv2.face.LBPHFaceRecognizer_create()
#cascadePath = "Classifiers/face.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataset'

def get_images_and_labels(path):
     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
     
     faces = []
     
     IDs = []
     for image_path in image_paths:
         
         faceImg = Image.open(image_path).convert('L')
         faceNp = np.array(faceImg, 'uint8')
         ID = int(os.path.split(image_path)[-1].split(".")[1])
         #nbr=int(''.join(str(ord(c)) for c in nbr))
         faces.append(faceNp)
         print (ID)
         
         IDs.append(ID)
         cv2.imshow("trainning",faceNp)
         cv2.waitKey(10)
     return IDs,faces

    



IDs,faces=get_images_and_labels(path)
recognizer.train(faces, np.array(IDs))
recognizer.save('trainer/trainerData.yml')
cv2.destroyAllWindows()
