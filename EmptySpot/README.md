This is a very basic example of an empty-spot-detector with openCV <br>
```python
import cv2

input_image = 'img_01.jpg'
image = cv2.imread(input_image)
# Convert to grayScale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Canny Edge Detector
image_edges = cv2.Canny(image_gray,0,50)
image_blur = cv2.GaussianBlur(image_edges, (15, 15), 0)
image_Thresholding = cv2.threshold(image_blur, 20, 255, cv2.THRESH_BINARY)
image_dilate = cv2.dilate(image_Thresholding[1], None, iterations=2)
image_mediablur = cv2.medianBlur(image_dilate,5)
#invert colors
final_image = cv2.bitwise_not(image_mediablur)

#detecting contours
contours = cv2.findContours(final_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

contour_list = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 300 :
        contour_list.append(contour)

cv2.drawContours(image, contour_list,  -1, (0,0,255), 2)
cv2.imshow('Objects Detected',image)
cv2.waitKey(0)
```
![alt text](https://drive.google.com/uc?export=view&id=115o21PtFqeaIQTdbIHH-_Dn2Yg1-Q30M)<br>
![alt text](https://drive.google.com/uc?export=view&id=1VxU3rI96q56FHAkFUk3Td_1NHRfewDOV)<br>
![alt text](https://drive.google.com/uc?export=view&id=1UH1q_t42nGB0ena3llCmnQKrVvTD7Xll)<br>
![alt text](https://drive.google.com/uc?export=view&id=1mTVNBBNOddgIkamIUGvRccq4wL-OaGpm)<br>
![alt text](https://drive.google.com/uc?export=view&id=1Kx-VR1TKHR1aRFzvJMEOI_MX0kWJYdPI)
