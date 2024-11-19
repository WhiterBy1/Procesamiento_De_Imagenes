import cv2
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# Leer la imagen
image = io.imread('images/Dall.jpg')

# Convertir a escala de grises si es necesario
if len(image.shape) == 3:
    image = rgb2gray(image)

# Calcular el umbral de Otsu
thresh = threshold_otsu(image)

# Crear la imagen binaria
binary = (image > thresh).astype('uint8') * 255

# Mostrar la imagen usando OpenCV
cv2.imshow('Imagen Binaria', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Imagen', image)
cv2.waitKey(0)
cv2.destroyAllWindows()