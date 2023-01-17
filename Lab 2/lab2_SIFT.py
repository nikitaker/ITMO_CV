import cv2

# read the images
source = cv2.imread("source.png")
template = cv2.imread("another.png")

# create SIFT object
sift = cv2.SIFT_create()  # opencv-contrib-python

# detect SIFT features in both images
keypoints_1, descriptors_1 = sift.detectAndCompute(source, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(template, None)

# create feature matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# match descriptors of both images
matches = bf.match(descriptors_1, descriptors_2)
# sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)
# draw first 50 matches
matched_img = cv2.drawMatches(source, keypoints_1, template, keypoints_2, matches[:15], template)


cv2.imshow('image', matched_img)
# save the image
cv2.imwrite("matched_images.jpg", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
