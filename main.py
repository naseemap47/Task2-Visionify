from ultralytics import YOLO
import cv2
import random
import numpy as np
import os


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



# Load a model
model = YOLO("yolov8n.pt")
class_names = model.names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# create a video capture object for the camera or video
cap = cv2.VideoCapture('task/5738755-hd_1920_1080_24fps.mp4')


# Write Video
save = True
if save:
    # Get the width and height of the video.
    original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Create directlry for saving video output
    os.makedirs('video_output', exist_ok=True)
    out_vid = cv2.VideoWriter(f"video_output/{len(os.listdir('video_output'))}.mp4", 
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (original_video_width, original_video_height))

# set full screen window
cv2.namedWindow('Track Person', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Track Person', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # take frame from video or camera.
    success, img = cap.read()
    if not success:
        print("[INFO] Failed Load Video")
        break
    
    # Get detection output from model with tracker
    results = model.track(img, tracker="bytetrack.yaml")
    # for one video or camera results[0]
    # otherwise for mutilple camera or video (batching)
    for result in results:
        bboxs = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls
        track_ids = result.boxes.id.int().cpu().tolist()
        for bbox, cnf, cs, track_id in zip(bboxs, conf, cls, track_ids):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            # Person class id: 0
            if int(cs) == 0:
                # plot bounding box on the top of person with track id, class name and detection confidence.
                plot_one_box(
                    [xmin, ymin, xmax, ymax], img,
                    colors[int(cs)], f'{track_id} {class_names[int(cs)]} {float(cnf):.3}',
                    3
                )
    # Write Video
    if save:
        out_vid.write(img)
    
    cv2.imshow('Track Person', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if save:
    out_vid.release()
cv2.destroyAllWindows()
