"""
    Use python to read 
    檢測圖片只需要 obj.names, yolov3.cfg 以及 weights 檔就夠
    可直接利用 Opencv 內建的 darknet 來讀取網路並產生出預測
"""
# TO_DETECTING_IMAGE_DIR_PATH = GITHUB_CODEBASE_DIR_PATH+"/to_detect_images"


import os
import cv2
import numpy as np
import glob
# from google.colab.patches import cv2_imshow

import pprint
pp = pprint.PrettyPrinter(indent=4)


def detecting_one_image(net, output_layers, img):
    # Detecting objects
    # cv::dnn::blobFromImage (InputArray image, double scalefactor=1.0, const Size &size=Size(), const Scalar &mean=Scalar(), bool swapRB=false, bool crop=false, int ddepth=CV_32F)
    blob = cv2.dnn.blobFromImage(img,  # InputArray image
                                 0.00392,  # double scalefactor=1.0
                                 (416, 416),  # const Size &size=Size()
                                 (0, 0, 0),  # const Scalar &mean=Scalar()
                                 True,  # bool swapRB=false
                                 crop=False 
                                 )
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    return outs

# Load Yolo
# net = cv2.dnn.readNet(GDRIVE_WEIGHTS_DIR_PATH+"/yolov3_last.weights", GDRIVE_CFG_DIR_PATH+"/yolov3.cfg")
net = cv2.dnn.readNet("yolov4-person_best.weights", "yolov4-person.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load label names
with open(GDRIVE_CFG_DIR_PATH+"/obj.names", "r") as f:
  classes = [line.strip() for line in f.readlines()]

# Generate display colors
colors = np.random.uniform(0, 255, size=(len(classes), 3))

for fpath in glob.glob(os.path.join(TO_DETECTING_IMAGE_DIR_PATH, "*.jpg")):
  print("fpath", fpath)

  # Loading image
  img = cv2.imread(fpath)
  height, width, channels = img.shape

  if width>800: # resize for display purpose
    dim = (800, int(800*height/width))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    height, width, channels = img.shape

  outs = detecting_one_image(net, output_layers, img)

  # Showing informations on the screen
  for out in outs:
    for detection in out:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > 0.3:
        # Object detected
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)

        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        label = "(%.2f) %s" % (confidence, classes[class_id])

        cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_id], 2)
        cv2.putText(img, label, (x, y+h-5), cv2.FONT_HERSHEY_PLAIN, 1, colors[class_id], 1)

  cv2_imshow(img)
