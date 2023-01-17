import cv2
import matplotlib.pyplot as plt
from time import perf_counter


cam = cv2.VideoCapture("./SampleVideo.mp4")

is_equal = True
plt.ion()


ret, frame = cam.read()
while cam.isOpened:
    ret, frame = cam.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_equal:
            start = perf_counter()
            frame = cv2.equalizeHist(frame)
            end = perf_counter()
            print("OpenCV time: " + str(end - start))

        plt.hist(frame.ravel(), 256, [0, 256])
        cv2.imshow('frame', frame)
        plt.show()

        if cv2.waitKey() == 32:
            is_equal = not is_equal

        if cv2.waitKey() == ord('q'):
            break
    else:
        break

cam.release()
cv2.destroyAllWindows()
