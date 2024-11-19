import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.filters import median, gaussian, threshold_otsu
from skimage.morphology import erosion, dilation, disk
from skimage.metrics import mean_squared_error, structural_similarity
import itertools
import sys
import os

# Definir métricas de similitud
def calculate_metrics(imageA, imageB):
    # Convertir imágenes booleanas a float
    if imageA.dtype == bool:
        imageA = imageA.astype(float)
    if imageB.dtype == bool:
        imageB = imageB.astype(float)

    data_range = imageA.max() - imageA.min()
    mse = mean_squared_error(imageA, imageB)
    try:
        ssim, _ = structural_similarity(imageA, imageB, full=True, channel_axis=-1, data_range=data_range)
    except TypeError:
        ssim = structural_similarity(imageA, imageB, data_range=data_range)
    return mse, ssim

# Definir filtros y sus parámetros automáticamente
filters = {
    'median': {
        'function': lambda image, footprint: median(image, footprint),
        'parameters': {'footprint': [disk(r) for r in range(1, 5)]}
    },
    'gaussian': {
        'function': gaussian,
        'parameters': {'sigma': [1, 2, 3]}
    },
    'erosion': {
        'function': erosion,
        'parameters': {'footprint': [disk(r) for r in range(1, 5)]}
    },
    'dilation': {
        'function': dilation,
        'parameters': {'footprint': [disk(r) for r in range(1, 5)]}
    },
    'thresholding': {
        'function': lambda image, thresh: image > thresh,
        'parameters': {'thresh': ['otsu', 0.1, 0.2, 0.3, 0.4, 0.5]}
    }
}

# Función para aplicar un filtro a imágenes RGB o escala de grises
def apply_filter(image, filter_func, params):
    if image.ndim == 2:  # Imagen en escala de grises
        return filter_func(image, **params)
    elif image.ndim == 3:  # Imagen RGB
        channels = []
        for i in range(image.shape[2]):
            channel = filter_func(image[:, :, i], **params)
            channels.append(channel)
        return np.stack(channels, axis=-1)
    else:
        raise ValueError("Imagen con formato no soportado.")

# Función para encontrar la mejor combinación de parámetros
def find_best_parameters(input_image, target_image, filter_name):
    filter_info = filters[filter_name]
    param_names = list(filter_info['parameters'].keys())
    param_values = list(filter_info['parameters'].values())
    combinations = list(itertools.product(*param_values))

    best_mse = float('inf')
    best_ssim = -1
    best_params = None
    best_result = None

    for params in combinations:
        param_dict = dict(zip(param_names, params))

        # Manejar el umbral de Otsu
        if filter_name == 'thresholding' and param_dict['thresh'] == 'otsu':
            param_dict['thresh'] = threshold_otsu(input_image)

        try:
            processed_image = apply_filter(input_image, filter_info['function'], param_dict)
            
            # Convertir a float si es booleano
            if processed_image.dtype == bool:
                processed_image = processed_image.astype(float)

            mse, ssim = calculate_metrics(processed_image, target_image)

            if mse < best_mse:
                best_mse = mse
                best_ssim = ssim
                best_params = param_dict
                best_result = processed_image
        except Exception as e:
            print(f"Error aplicando parámetros {param_dict}: {e}")

    if best_result is None:
        print(f"No se pudo encontrar una configuración válida para '{filter_name}'.")
        best_result = input_image.copy()

    print(f"\nMejor parámetro para '{filter_name}': {best_params}")
    print(f"MSE: {best_mse:.4f}, SSIM: {best_ssim:.4f}")
    return best_result

# Menú interactivo para selección del filtro
def menu():
    print("\nSeleccione el filtro que desea aplicar:")
    for idx, filter_name in enumerate(filters.keys()):
        print(f"{idx + 1}. {filter_name}")

    while True:
        try:
            selected_filter = int(input("Ingrese el número del filtro: "))
            if 1 <= selected_filter <= len(filters):
                filter_name = list(filters.keys())[selected_filter - 1]
                break
            else:
                raise ValueError
        except ValueError:
            print("Opción inválida. Intente nuevamente.")
    
    return filter_name

# Función para cargar y validar imágenes
def load_image(path):
    if not os.path.exists(path):
        print(f"Error: El archivo '{path}' no existe.")
        sys.exit(1)
    image = img_as_float(io.imread(path))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image

# Programa principal
def main():
    input_path = 'Quiz/benign (3).png' #input("Ingrese la ruta de la imagen de entrada: ")
    target_path = 'Quiz/benign (3)_mask.png'#input("Ingrese la ruta de la imagen objetivo: ")


    input_image = load_image(input_path)
    target_image = load_image(target_path)

    if input_image.shape != target_image.shape:
        print("Error: Las imágenes deben tener el mismo tamaño.")
        sys.exit(1)

    while True:
        mode = input("\nSeleccione el modo:\n1. Automatizado\n2. Manual\nIngrese 1 o 2: ")
        if mode in ['1', '2']:
            break
        else:
            print("Opción no válida. Intente nuevamente.")

    if mode == '1':
        print("\nEjecutando proceso automatizado...")
        best_mse = float('inf')
        best_ssim = -1
        best_filter = None
        best_result = None

        for filter_name in filters.keys():
            result_image = find_best_parameters(input_image, target_image, filter_name)
            mse, ssim = calculate_metrics(result_image, target_image)

            if mse < best_mse:
                best_mse = mse
                best_ssim = ssim
                best_filter = filter_name
                best_result = result_image

        print(f"\nMejor filtro: {best_filter}")
        print(f"MSE: {best_mse:.4f}, SSIM: {best_ssim:.4f}")
        result_image = best_result

    elif mode == '2':
        filter_name = menu()
        result_image = find_best_parameters(input_image, target_image, filter_name)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Imagen de Entrada')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(target_image, cmap='gray')
    plt.title('Imagen Objetivo')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result_image, cmap='gray')
    plt.title('Imagen Procesada')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
