# import os
# import nibabel as nib
# import numpy as np
# from PIL import Image

# def nii_to_png(input_dir, output_dir):
#     """
#     Конвертирует все .nii/.nii.gz файлы в PNG и сохраняет в новую папку.
    
#     Параметры:
#         input_dir (str): Путь к папке с .nii файлами
#         output_dir (str): Путь для сохранения PNG
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     for filename in os.listdir(input_dir):
#         if filename.endswith(('.nii', '.nii.gz')):
#             nii_path = os.path.join(input_dir, filename)
#             img = nib.load(nii_path)
#             data = img.get_fdata()
            
#             # Удаляем единичные измерения (если есть)
#             data = np.squeeze(data)
            
#             # Проверяем, что осталось 2 или 3 измерения
#             if data.ndim == 3:  # 3D-данные (Z, Y, X)
#                 for z in range(data.shape[2]):
#                     slice_2d = data[:, :, z]
#                     save_slice(slice_2d, output_dir, filename, z)
#             elif data.ndim == 2:  # 2D-данные (Y, X)
#                 save_slice(data, output_dir, filename)
#             else:
#                 print(f"Файл {filename} имеет неожиданную размерность: {data.shape}")

# def save_slice(slice_2d, output_dir, filename, z=None):
#     """Сохраняет 2D-срез в PNG."""
#     # Нормализуем в 0-255 и конвертируем в uint8
#     slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255
#     slice_2d = slice_2d.astype(np.uint8)
    
#     # Создаем имя файла
#     base_name = os.path.splitext(filename)[0]
#     if filename.endswith('.nii.gz'):
#         base_name = os.path.splitext(base_name)[0]
    
#     if z is not None:
#         png_filename = f"{base_name}_slice{z:03d}.png"
#     else:
#         png_filename = f"{base_name}.png"
    
#     png_path = os.path.join(output_dir, png_filename)
#     Image.fromarray(slice_2d).save(png_path)

# # Пример вызова
# input_folder = "D:\\Task01_BrainTumour\\Task01_BrainTumour\\imagesTr\\BRATS_001.nii"
# output_folder = "D:\\Task01_BrainTumour\\Task01_BrainTumour\\png_data"
# nii_to_png(input_folder, output_folder)
# import os
# import nibabel as nib  # для чтения .nii
# import numpy as np
# from PIL import Image  # для сохранения PNG

# # Загрузите NIfTI-файл
# nii_path = "D:\\Task01_BrainTumour\\Task01_BrainTumour\\imagesTr\\BRATS_001.nii\\BRATS_001.nii"  # или .nii.gz
# img = nib.load(nii_path)
# data = img.get_fdata()  # 3D-массив в формате [срезы, высота, ширина]

# # Нормализация в 8-битный диапазон (0-255)
# data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
# data_normalized = data_normalized.astype(np.uint8)

# # Создайте папку для PNG
# output_dir = "D:\\Task01_BrainTumour\\Task01_BrainTumour\\images"
# os.makedirs(output_dir, exist_ok=True)

# # Сохраните каждый срез как PNG
# for slice_idx in range(data.shape[2]):  # если оси поменяны, используйте 0 или 1
#     slice_2d = data_normalized[:, :, slice_idx]  # или data[:, slice_idx, :] и т.д.
#     im = Image.fromarray(slice_2d)
#     im.save(os.path.join(output_dir, f"slice_{slice_idx:04d}.png"))


# import os
# import nibabel as nib  # чтение .nii
# import numpy as np
# from tifffile import imsave, imwrite  # для сохранения TIFF (поддержка 16 бит)
# # Альтернатива: from PIL import Image (но без сжатия LZW)

# # --- Настройки ---
# nii_path = "D:\\Task01_BrainTumour\\Task01_BrainTumour\\imagesTr\\BRATS_001.nii\\BRATS_001.nii"  # или .nii.gz
# output_dir = "D:\\Task01_BrainTumour\\Task01_BrainTumour\\images"     # куда сохранять

# def convert_to_grayscale_tiffs(nii_path, output_dir, method='first_channel', bit_depth=8):
#     """
#     Конвертирует 4D NIfTI в серию одноканальных grayscale TIFF-файлов.
    
#     Параметры:
#         nii_path (str): Путь к файлу .nii или .nii.gz
#         output_dir (str): Папка для сохранения TIFF
#         method (str): Способ преобразования:
#             'first_channel' - взять первый канал (по умолчанию)
#             'mean' - усреднить все каналы
#             'max' - взять максимум по каналам
#         bit_depth (int): 8 или 16 бит
#     """
#     # Загрузка данных
#     img = nib.load(nii_path)
#     data = img.get_fdata().astype(np.float32)  # (240, 240, 155, 4)
    
#     # Преобразование в grayscale
#     if method == 'first_channel':
#         gray_data = data[..., 0]  # Берём первый канал
#     elif method == 'mean':
#         gray_data = np.mean(data, axis=-1)  # Усредняем каналы
#     elif method == 'max':
#         gray_data = np.max(data, axis=-1)  # Берём максимум
#     else:
#         raise ValueError("Неизвестный метод. Используйте: 'first_channel', 'mean' или 'max'")
    
#     # Нормализация
#     gray_data = (gray_data - gray_data.min()) / (gray_data.max() - gray_data.min())
#     if bit_depth == 8:
#         gray_data = (gray_data * 255).astype(np.uint8)
#     else:
#         gray_data = (gray_data * 65535).astype(np.uint16)
    
#     # Сохранение
#     os.makedirs(output_dir, exist_ok=True)
#     for z in range(gray_data.shape[2]):
#         imwrite(
#             os.path.join(output_dir, f"slice_{z:04d}.tif"),
#             gray_data[:, :, z],
#             photometric='minisblack'  # Обязательно для grayscale
#         )

# # Пример использования
# convert_to_grayscale_tiffs(
#     nii_path=nii_path,
#     output_dir=output_dir,
#     method='mean',  # Усреднение каналов
#     bit_depth=8     # 8-битные изображения
# )
import numpy as np
from PIL import Image
data = np.load('D:\\DATA\\demo data\\gt\\TotalSegmentator_midslice_res256_s0716_seg.npy')

# Преобразование в бинарное изображение (0 и 255)
binary_image = (data * 255).astype(np.uint8)  # если данные в диапазоне [0, 1]
# или
# binary_image = np.where(data > 0, 255, 0).astype(np.uint8)  # если данные логические

# Сохранение в PNG
img = Image.fromarray(binary_image, mode='L')  # 'L' — grayscale (8-bit)
img.save('D:\\DATA\\demo data\\gt\\output.png')