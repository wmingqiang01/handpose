import cv2
img = cv2.imread('背景.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('beijing_gray.png', img)