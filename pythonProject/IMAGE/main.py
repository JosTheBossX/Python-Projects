import cv2
import numpy as np


num_down = 2  #number of downsampling steps
num_bilateral = 7   #number of bilateral filtersteps

img_rgb = cv2.imread("IMAGE/Ayushmann2.jpg")
print(img_rgb.shape)        #prints the dimensionsof the picture
cv2.imshow('THISH',img_rgb)
#resizing so as to get optimal results after unsampling is done.
img_rgb = cv2.resize(img_rgb, (800,800))

#downsample image using gaussian pyramid
img_color = img_rgb
for _ in range(num_down):
    img_color = cv2.pyrDown(img_color)

#repeatedly apply small bilateral filter instead of applying one large filter
for _ in range(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d=9,
                                    sigmaColor=9, sigmaSpace=7)


#upsample image to original size
for _ in range(num_down):
    img_color = cv2.pyrUp(img_color)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)
img_edge = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,
                                 blockSize=9, C=2)


img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

img_cartoon = cv2.bitwise_and(img_color, img_edge)


stack = np.hstack([img_rgb,img_cartoon])
cv2.imshow('Stacked Images', stack)
cv2.imshow('final',img_cartoon)
cv2.waitKey(0)
if cv2.waitKey(1)==27:
    cv2.destroyAllWindows()
