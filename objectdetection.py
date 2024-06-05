import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

net = cv.dnn.readNet("yolov3.weights","yolov3.cfg")

classes =[]
with open('object.names','r') as f:
    classes = f.read().splitlines()

img = cv.imread("b.jpeg")

(height,width, _ ) = img.shape
newImg = img[:,:,::-1]

blob = cv.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)

net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layers_outputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layers_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if(confidence>0.5):
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)
print(len(boxes))
# print(boxes)

indexes = cv.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
# print(indexes.flatten())

font = cv.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0,255,size=(len(boxes),3))

for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i],2));
    color= colors[i]
    cv.rectangle(img,(x,y),(x+w,y+h),color,2)
    cv.putText(img,label + " " + confidence,(x,y+20),font,1,(255,255,255),2)
cv.imshow("Image",img)
cv.waitKey(0)
cv.destroyAllWindows()