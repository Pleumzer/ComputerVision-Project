import cv2
import numpy as np
from TrackCar import *

end = 0

trackCar = EuclideanDistTracker()

video_path = "demo.mp4"
read_video = cv2.VideoCapture(video_path)

object_detector = cv2.createBackgroundSubtractorMOG2(history=None, varThreshold=None)

default_fps = read_video.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / default_fps)

kernalOp = np.ones((3,3), np.uint8)
kernalOp2 = np.ones((5,5), np.uint8)
kernalCl = np.ones((11,11), np.uint8)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((5,5), np.uint8)

while True:
    ret, frame = read_video.read()
    if not ret:
        break

    height, width, _ = frame.shape
    left_half = frame[:, width // 2:, :]
    roi = frame[330:620, 661:1280]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

    fgmask = fgbg.apply(roi)
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)

    # กำหนดขนาดของ kernel สำหรับ morphological operations
    kernel_erode = np.ones((3, 3), np.uint8)  # กำหนด kernel สำหรับ erosion
    kernel_dilate = np.ones((5, 5), np.uint8)  # กำหนด kernel สำหรับ dilation

    # ทำการ erosion เพื่อลบ noise
    e_img = cv2.erode(mask2, kernel_erode)

    contours, _ = cv2.findContours(e_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # THRESHOLD
        if area > 2800:
            x, y, w, h = cv2.boundingRect(cnt)
            # กรอบใหญ่
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])

    boxes_ids = trackCar.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        if (trackCar.getsp(id) < trackCar.limit()):
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 0, 255), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

        s = trackCar.getsp(id)
        if (trackCar.f[id] == 1 and s != 0):
            trackCar.capture(roi, x, y, h, w, s, id)
    # วาดเส้นบนภาพ
    cv2.line(left_half, (0, 330), (660, 330), (0, 0, 255), 2)
    cv2.line(left_half, (0, 350), (660, 350), (0, 0, 255), 2)
    cv2.line(left_half, (0, 600), (660, 600), (0, 0, 255), 2)
    cv2.line(left_half, (0, 620), (660, 620), (0, 0, 255), 2)

    cv2.imshow("With Lines", left_half)
    cv2.imshow("Mask", mask2)
    cv2.imshow("Roi", roi)

    key = cv2.waitKey(frame_delay) & 0xFF
    if key == 27:
        break

read_video.release()
cv2.destroyAllWindows()
