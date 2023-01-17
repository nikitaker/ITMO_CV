import cv2
import matplotlib.pyplot as plt
import numpy as np
import numba
from time import perf_counter


@numba.njit
def calculate(hist_t):
    cdf = hist_t.cumsum()
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = cdf.astype(np.uint8)  # Transform from float64 back to unit8
    return cdf


def histogram_equalization(image):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = calculate(hist)
    result = cdf[image]
    return result



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
            frame = histogram_equalization(frame)
            end = perf_counter()
            print("Numba time: " + str(end - start))

        plt.hist(frame.ravel(), 256, [0, 256])

        cv2.imshow('frame', frame)
        plt.show()

        if cv2.waitKey() == 32:
            is_equal = not is_equal

        if cv2.waitKey() == ord('q'):
            break

    # Break the loop
    else:
        break

cam.release()
cv2.destroyAllWindows()
