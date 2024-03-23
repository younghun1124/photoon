import cv2
import numpy as np

def color_quantization(image, k):
    # 이미지를 float32 타입으로 변환하고 3차원 배열을 2차원 배열로 재구성합니다.
    Z = image.reshape((-1,3))
    Z = np.float32(Z)

    # 정의한 클러스터 개수(k)와 반복 종료 조건을 설정합니다.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # UINT8로 다시 변환합니다.
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))

    return result_image

# Load the image
img = cv2.imread('image2.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply median blur to reduce noise
gray = cv2.medianBlur(gray, 5)
# Detect edges using adaptive thresholding
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

# 이미지의 색상을 제한합니다.
color = color_quantization(img, 10)

# Combine the color image with the edges mask
cartoon = cv2.bitwise_and(color, color, mask=edges)
# Display the cartoon image
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
