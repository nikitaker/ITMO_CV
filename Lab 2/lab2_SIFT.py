import cv2
import numpy as np

# read the images
source = cv2.imread("source.png")
template = cv2.imread("test5.png")


# create SIFT object
sift = cv2.SIFT_create()  # opencv-contrib-python

# detect SIFT features in both images
keypoints_1, descriptors_1 = sift.detectAndCompute(template, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(source, None)

# create feature matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# match descriptors of both images
matches = bf.match(descriptors_1, descriptors_2)
# sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

gmatches = matches[:20]

## extract the matched keypoints
src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in gmatches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in gmatches]).reshape(-1, 1, 2)

## find homography matrix and do perspective transform
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h, w = template.shape[:2]
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

## draw found regions
source = cv2.polylines(source, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)

## draw match lines
res = cv2.drawMatches(template, keypoints_1, source, keypoints_2, gmatches, None, flags=2)

cv2.imshow("SIFT", res)
cv2.waitKey()
cv2.destroyAllWindows()
