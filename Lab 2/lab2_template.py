import numpy as np
import cv2

source = cv2.imread("source.png")
template = cv2.imread("another.png")
(templateHeight, templateWidth) = template.shape[:2]

# ищем template на основном изображении,
result = cv2.matchTemplate(source, template, cv2.TM_SQDIFF)

# ищем минимум в хитмапе
(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)


topLeft = minLoc
botRight = (topLeft[0] + templateWidth, topLeft[1] + templateHeight)
roi = source[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

mask = np.zeros(source.shape, dtype="uint8")

new_image = cv2.addWeighted(source, 0.5, mask, 1, 0)


# выделяем нужную нам область на изображении
new_image[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi

# display the images
cv2.imshow("Main_image", new_image)
cv2.imshow("Template", template)
cv2.imwrite("template_matched.png", new_image)
cv2.waitKey(0)
