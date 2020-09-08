import cv2
import numpy as np
import time

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
# with open("tiny_classes.txt", "r") as f:
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# vc = cv2.VideoCapture("HK_WALK_360p.mp4")
vc = cv2.VideoCapture("crowd.mp4")

# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
# net = cv2.dnn.readNet("yolov4-person_best.weights", "yolov4-person.cfg")
net = cv2.dnn.readNetFromDarknet("yolov4-person.cfg","yolov4-person_best.weights")
# net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg","yolov4-tiny.conv.29")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/256)

ids = []

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    # 畫一條直線
    cv2.line(frame, (0,150), (frame.shape[1], 150), (0,0,255), 1)

    if len(ids) == 0:
        ids = list(range(len(boxes)))
    print("ids = ", ids)

    start_drawing = time.time()
    for i, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)

        # 根據直線高低決定BBox顏色
        color = (255,0,0) if box[1] > 150 else (0,255,0)

        # 決定id



        label += "_id{}" .format(i)


        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("detections", frame)