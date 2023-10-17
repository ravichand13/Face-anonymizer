import argparse
import mediapipe as mp
import cv2
import os
#read image

image_path=os.path.join('..', 'ps', 'rg.jpg')
img=cv2.imread(image_path)
H,W,_=img.shape


#detect faces #(if there is no face it gives error)
mp_face_detection=mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out=face_detection.process(img_rgb)


    #print(out.detections)
#model_selection 0->within 2mts range 1->long range,min_detect_confidence->50% orcan keep any value
#  (out.detections) gives some coordinates,boundry,pts etc
    #in form of locations_data->relative_bounding_box


    for detections in out.detections:
        location_data=detections.location_data
        bbox=location_data.relative_bounding_box

        x1, y1, w, h=bbox.xmin,bbox.ymin,bbox.width,bbox.height
        x1 = int(x1 * W)
        y1 = int(y1 * H)
        w = int(w * W)
        h = int(h * H)

        img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        #blur faces
        img[y1: y1 + h, x1: x1 + w, :] = cv2.blur(img[y1: y1 + h, x1: x1+w,:], (7, 7))

#show image
cv2.imshow('img',img)
cv2.waitKey(0)

#we can use argparse lib

"""
args=argparse.ArgumentParser()
args.add_argument("--mode",default="image")
args.add_argument("--filePath",default="./ps/rg.jpg")
args.parse_args()

if args.mode in ["image"]
    #read image code(like above read image)
    #set from img_rgb to rest was a fxn and pass img and facedetection to it
    #write output/show for image
    
    #lly
elif args.mode in ["video"]
    #create some default output fxn with arguments for easy data entry like outpt loc & fps
elif args.mode in ["webcam"]
"""

#for only blurring comment rectangle line
