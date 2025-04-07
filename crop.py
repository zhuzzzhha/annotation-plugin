import glob
import os
import numpy as np
from PIL import Image  # Для работы с изображениями (если нужно из файла)

def crop_and_save_images(input_folder, output_folder, top_left, bottom_right):
    """
    Обрезает все изображения в указанной папке, сохраняет обрезанные изображения
    в другую папку.

    Args:
        input_folder (str): Путь к папке с исходными изображениями.
        output_folder (str): Путь к папке, куда будут сохранены обрезанные изображения.
        top_left (tuple): Координаты верхнего левого угла обрезки (x, y).
        bottom_right (tuple): Координаты правого нижнего угла обрезки (x, y).
    """

    # Создаем выходную папку, если она не существует
    os.makedirs(output_folder, exist_ok=True)

    # Получаем список файлов изображений (jpg, jpeg, png, tif, tiff, bmp)
    image_files = (
        glob.glob(os.path.join(input_folder, "*.jpg"))
        + glob.glob(os.path.join(input_folder, "*.jpeg"))
        + glob.glob(os.path.join(input_folder, "*.png"))
        + glob.glob(os.path.join(input_folder, "*.tif"))
        + glob.glob(os.path.join(input_folder, "*.tiff"))
        + glob.glob(os.path.join(input_folder, "*.bmp"))
    )

    if not image_files:
        print(f"В папке '{input_folder}' не найдено изображений.")
        return

    for image_path in image_files:
        try:
            # Открываем изображение с помощью PIL
            image = Image.open(image_path)

            # Обрезаем изображение (используем функцию crop_image, которую мы определили ранее)
            cropped_image = crop_image(image, top_left, bottom_right)

            if cropped_image is not None:
                # Формируем путь для сохранения обрезанного изображения
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{name}_cropped{ext}")

                # Сохраняем обрезанное изображение. PIL требует Image object, поэтому numpy array конвертируем обратно.
                cropped_image_pil = Image.fromarray(cropped_image)
                cropped_image_pil.save(output_path)  # Сохраняем как PIL Image.

                print(f"Изображение '{filename}' успешно обрезано и сохранено в '{output_path}'.")
            else:
                print(f"Не удалось обрезать изображение '{filename}'.")

        except Exception as e:
            print(f"Ошибка при обработке изображения '{image_path}': {e}")



def crop_image(image, top_left, bottom_right):
    """
    Обрезает изображение по заданным верхнему левому и правому нижнему углам.

    Args:
        image (numpy.ndarray or PIL.Image.Image): Исходное изображение. Может быть представлено
            как NumPy array (например, прочитанное с помощью skimage.io.imread)
            или как объект PIL Image.
        top_left (tuple): Координаты верхнего левого угла в формате (x, y).
        bottom_right (tuple): Координаты правого нижнего угла в формате (x, y).

    Returns:
        numpy.ndarray or PIL.Image.Image: Обрезанное изображение, в том же формате, что и входное.
            Возвращает None, если координаты обрезки недействительны.
    """

    x1, y1 = top_left
    x2, y2 = bottom_right

    # Проверка координат на валидность
    if x1 >= x2 or y1 >= y2:
        print("Ошибка: Некорректные координаты обрезки. Верхний левый угол должен быть выше и левее правого нижнего.")
        return None

    if isinstance(image, np.ndarray):
        # Обрезка NumPy array
        height, width = image.shape[:2]  # Учитываем, что у изображения может быть несколько каналов (RGB, RGBA)

        # Проверка границ
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped_image = image[y1:y2, x1:x2]  # Важно: y идет первым!
        return cropped_image
    elif isinstance(image, Image.Image):
        # Обрезка PIL Image
        width, height = image.size

        # Проверка границ
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped_image = image.crop((x1, y1, x2, y2)) # PIL требует tuple (left, top, right, bottom)
        return np.array(cropped_image) # возвращаем в формате numpy, для единообразия
    else:
        print("Ошибка: Неподдерживаемый тип изображения.  Используйте NumPy array или PIL Image.")
        return None



if __name__ == '__main__':
    # Пример использования:
    input_folder = "input_images"  # Замените на путь к папке с вашими изображениями
    output_folder = "cropped_images"  # Замените на путь к папке, куда сохранить обрезанные изображения
    top_left = (390, 250)
    bottom_right = (995, 850)

    input_folder = "D:\\reconstruction\\mask_versions\\3"
    output_folder = "D:\\cropped_reconstrucction_gt"
    # 2. Вызываем функцию обрезки и сохранения
    crop_and_save_images(input_folder, output_folder, top_left, bottom_right)