import cv2
import numpy as np
import pywt

# загрузка изображения
img = cv2.imread('src/image.jpg', cv2.IMREAD_GRAYSCALE)

# приведение изображения к размеру 512х512
img = cv2.resize(img, (512, 512))

# вейвлет-разложение изображения с использованием вейвлета Хаара
coeffs = pywt.dwt2(img, 'haar')

# вейвлет-разложение изображения с использованием вейвлета Добеши второго порядка
coeffs2 = pywt.dwt2(img, 'db2')

# удаление высокочастотных составляющих
coeffs = list(coeffs)
coeffs[0] = np.zeros_like(coeffs[0])
img_haar = pywt.idwt2(tuple(coeffs), 'haar')

coeffs2 = list(coeffs2)
coeffs2[0] = np.zeros_like(coeffs2[0])
img_db2 = pywt.idwt2(tuple(coeffs2), 'db2')

# сравнение результатов с классической низкочастотной фильтрацией изображения
kernel = np.ones((5,5),np.float32)/25
img_filtered = cv2.filter2D(img,-1,kernel)

# отображение результатов
cv2.imshow('Original Image', img)
cv2.imshow('Haar Wavelet', img_haar)
cv2.imshow('Daubechies Wavelet', img_db2)
cv2.imshow('Filtered Image', img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

# d