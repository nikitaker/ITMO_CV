import cv2

source = cv2.imread("source.png")
template = cv2.imread("another.png")
(templateHeight, templateWidth) = template.shape[:2]

result = cv2.matchTemplate(source, template, cv2.TM_SQDIFF)
# If you are using cv.TM_SQDIFF as comparison method, minimum value gives the best match.
(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

topLeft = minLoc
botRight = (topLeft[0] + templateWidth, topLeft[1] + templateHeight)

cv2.rectangle(source, topLeft, botRight, 255, 2)

cv2.imshow("Main_image", source)
cv2.imshow("Template", template)
cv2.waitKey(0)
