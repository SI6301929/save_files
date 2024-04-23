import cv2
import numpy as np
from PIL import Image

# プロパティの設定
#video_capture = cv2.VideoCapture(1)
video_capture = cv2.VideoCapture(0)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

# cv2.VideoCapture()と違ってcap.start()を忘れずに
while True:
    _, frame = video_capture.read()
    print()
    #frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #im =Image.fromarray(frame_cvt)
    cv2.imshow(" ",frame)

    # Press Q to stop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ストリーミング停止
video_capture.release()
cv2.destroyAllWindows()
