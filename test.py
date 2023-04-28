import cv2
import pywt

# загрузка изображения
img = cv2.imread('src/image.jpg')

# изменение размера до 512х512
img = cv2.resize(img, (512, 512))

# проведение 4 уровней вейвлет-разложения с помощью вейвлета Хаара
coeffs = pywt.wavedec2(img, 'haar', level=4)

# выделение границ объекта
coeffs_H = list(coeffs)
coeffs_H[0] *= 0
for i in range(1, len(coeffs_H)):
    coeffs_H[i] = pywt.threshold(coeffs_H[i], coeffs_H[i].max() * 0.1, mode='soft')
img_H = pywt.waverec2(coeffs_H, 'haar')

# удаление высокочастотных составляющих
coeffs_L = list(coeffs)
for i in range(1, len(coeffs_L)):
    coeffs_L[i] *= 0
img_L = pywt.waverec2(coeffs_L, 'haar')

# сравнение с классической низкочастотной фильтрацией изображения
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# отображение результатов
cv2.imshow('Original Image', img)
cv2.imshow('Object Boundaries', img_H)
cv2.imshow('Low Frequency Components', img_L)
cv2.imshow('Blur Filtered Image', img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()