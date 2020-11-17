import os
import cv2
import numpy as np

FACE_DETECTOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_detection', 'face_detector')

prototype = os.path.join(FACE_DETECTOR_DIR, 'deploy.prototxt')
weights = os.path.join(FACE_DETECTOR_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
net = cv2.dnn.readNet(prototype, weights)

camera = cv2.VideoCapture(0)

def gen_frames():
    process_this_frame = True
    while True:
        success, frame = camera.read()  # read the camera frame
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                x_face = []
                face = frame[startY:endY, startX:endX]
                
                try:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (64, 64))
                    face = face/255
                    x_face.append(face)
                    x_face = np.array(x_face)

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255,0), 2)
                except Exception as e:
                    print(f"Unexpected failure: {e}")

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
