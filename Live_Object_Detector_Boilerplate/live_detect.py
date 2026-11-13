# Live Object Detector (student starter)
# Run it from inside this folder (labels.json and detector.onnx must be next to it):
#     python live_detect.py
# Lines marked (given) are done for you. Your job is TODO 1 to TODO 5. Do them in order.
# To stop: click the video window and press q. If nothing responds, press Ctrl+C in the terminal.

import json

import cv2
import numpy as np

# (given) settings. You change MIN_SCORE in step 5.
MIN_SCORE = 0.5
SIZE = 640
STRIDES = [8, 16, 32]
FONT = cv2.FONT_HERSHEY_SIMPLEX

with open("labels.json") as file:
    coco = json.load(file)["coco"]
NAMES = [coco[key] for key in sorted(coco, key=int)]

net = cv2.dnn.readNetFromONNX("detector.onnx")

cells = [SIZE // stride for stride in STRIDES]
GRID = np.concatenate([np.indices((n, n))[::-1].reshape(2, -1).T for n in cells])
STRIDE = np.concatenate([np.full((n * n, 1), s) for n, s in zip(cells, STRIDES)])


# (given) detect(frame) looks at one picture and returns a list of (name, box).
def detect(frame):
    # (given) A: fit the picture into a 640 by 640 square, padded with grey, so nothing is stretched.
    height, width = frame.shape[:2]
    scale = min(SIZE / height, SIZE / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small

    # (given) B: run the model. output has 8400 rows (guesses). Each row has 85 numbers:
    #          4 for a box, 1 for "is there an object here?", and 80 name scores.
    net.setInput(cv2.dnn.blobFromImage(canvas))
    output = net.forward()[0]

    # (given) C: turn each guess's 4 box numbers into a box in pixels.
    centres = (output[:, :2] + GRID) * STRIDE
    sizes = np.exp(output[:, 2:4]) * STRIDE

    # TODO 1: score every guess.
    #   output[:, 4:5] is "is there an object here?" (1 number for each guess).
    #   output[:, 5:] is the 80 name scores (80 numbers for each guess).
    #   A guess is only good if BOTH are high, so multiply them. Call the result scores.
    #   Check: print(scores.shape) should show (8400, 80). Delete the print afterwards.
    scores = output[:, 4:5] * output[:, 5:]
    class_ids = scores.argmax(axis=1)
    confidences = scores.max(axis=1)
    # TODO 2: for every guess, find its best name and how sure it is.
    #   scores.argmax(axis=1) gives, for each row, the position of the biggest number. Call it class_ids.
    #   scores.max(axis=1) gives the biggest number in each row. Call it confidences.


    # (given) D: change each box into (x, y, w, h) on your ORIGINAL picture.
    boxes = (np.hstack([centres - sizes / 2, sizes]) / scale).astype(int)
    # TODO 3: keep the sure boxes and merge duplicates. Call the result keep.
    #   cv2.dnn.NMSBoxesBatched(boxes, confidences, class_ids, score_threshold, nms_threshold)
    #   Give it boxes, confidences and class_ids as lists (add .tolist()), then MIN_SCORE, then 0.5.
    #   It returns the positions of the guesses to keep.
    keep = cv2.dnn.NMSBoxesBatched(
        boxes.tolist(), confidences.tolist(), class_ids.tolist(),MIN_SCORE, 0.5
    )
    # TODO 4: return a list of (name, box) for every kept position i.
    #   The name is NAMES[class_ids[i]] and the box is boxes[i]. A list comprehension does it: [ ... for i in keep]
    return [(NAMES[class_ids[i]], boxes[i]) for i in keep]
   # placeholder: replace this line with your own


# (given) the camera loop: one picture at a time. TODO 5 is inside it.
video = cv2.VideoCapture(0)

# (given) self-check: runs detect() on the first camera picture, so you can test TODO 1 to 4 before TODO 5.
# Point the camera at something (a cup, a bottle, yourself). When TODO 1 to 4 are right you see, for example:
#     objects found: ['cup']      (if it shows [] hold something in front of the camera and run again)
ok, first = video.read()
if ok:
    print("objects found:", [name for name, box in detect(first)])

while True:
    ok, frame = video.read()
    if not ok:
        break
    for name, (x, y, w, h) in detect(frame):
        # (given) a different colour for each name
        hsv = np.uint8([[[NAMES.index(name) * 29 % 180, 255, 255]]])
        colour = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])
        # TODO 5a: draw the box around the object.
        #   cv2.rectangle(picture, corner1, corner2, colour, line_width)
        #   corner1 is the top-left corner (x, y). corner2 is the opposite corner (x + w, y + h).
        #   Use the ready-made colour and a line width of 2.
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(frame, name, (x, y - 6), FONT, 0.6, colour, 2)        # TODO 5b: write the name just above the box, in the same colour.
        #   cv2.putText(picture, text, position, FONT, size, colour, thickness)
        #   text is name. position is the top-left corner, moved 6 pixels up. Use size 0.6 and thickness 2.
    cv2.imshow("Live detect", frame)
    if cv2.waitKey(1) & 255 == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
