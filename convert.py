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


import os
import nibabel as nib  # чтение .nii
import numpy as np
from tifffile import imsave  # для сохранения TIFF (поддержка 16 бит)
# Альтернатива: from PIL import Image (но без сжатия LZW)

# --- Настройки ---
nii_path = "D:\\Task01_BrainTumour\\Task01_BrainTumour\\imagesTr\\BRATS_001.nii\\BRATS_001.nii"  # или .nii.gz
output_dir = "D:\\Task01_BrainTumour\\Task01_BrainTumour\\images"     # куда сохранять
