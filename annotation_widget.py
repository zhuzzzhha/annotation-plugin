from enum import Enum
import pathlib
import cv2
from matplotlib.colors import LinearSegmentedColormap
import napari.layers
import torch
import torch.nn.functional as F
from datetime import datetime
from magicgui import magic_factory, magicgui
from collections import deque
from itertools import product
from matplotlib import pyplot as plt
import napari
import numpy as np
import os
from qtpy import QtWidgets
from PIL import Image
from skimage.io import imread
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QGridLayout,
    QMessageBox)
from napari.utils.colormaps import (
    DirectLabelColormap,
    label_colormap,
)

from click_counter import ClickCounter
from graph_widget import myPyQtGraphWidget
from label_widget import LabelItem
from model import ScribblePromptUNet
from plotter_widget import IoUPlotter, BrPlotter
from segment_anything.sam import ScribblePromptSAM
from segment_widget import SegmentItem
from timer_widget import TimerWidget
from skimage.transform import resize
from skimage.draw import line
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.segmentation import find_boundaries

class COLORS(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

index_to_color_map = {
    0: "red",
    1: "green",
    2: "blue"
}

file_dir = pathlib.Path(os.path.dirname(__file__))
exp_dir = file_dir / "checkpoints" 

model_dict = {
    'ScribblePrompt-Unet': 'ScribblePrompt_unet_v1_nf192_res128.pt'
}


class AnnotationWidget(QtWidgets.QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        #Layers
        self.feedback_layer = None
        self.segments_layer = None
        self.gt_mask = None      

        #Paths
        self.versions_directory_path = None
        self.current_version_dir_path = None
        self.current_version_volume = None

        self.segements_count = 0
        self.gt_volume = None
        self.fp_feedback_data_array = []
        self.version_to_init_volume = dict()
        self.data_without_fp = dict()
        self.has_points = False

        #Statistics
        self.iou = []
        self.clicks = []
        self.segment_points = []
        self.pos_points = []
        self.neg_points = []

        self.brush_color_index = 0
        self.label_items_array = []
        self.colormap = label_colormap(num_colors=4)
        self.segment_items_array = []
        
        #Widgets
        self.brush_iou_plotter = IoUPlotter(self.viewer)
        self.viewer.window.add_dock_widget(self.brush_iou_plotter, area='bottom', name="IoU/clicks")
        self.segment_iou_plotter = IoUPlotter(self.viewer)
        self.brightness_plotter = BrPlotter(self.viewer)
        self.viewer.window.add_dock_widget(self.segment_iou_plotter, area='bottom', name="IoU/clicks")
        self.viewer.window.add_dock_widget(self.brightness_plotter, area="bottom", name="Brightness")
        
        #Colors dict
        self.colors = self.colormap.colors[1:5]
        self.colors[0] = np.array([0.0, 1.0, 0.0, 1.0])  # Зеленый
        self.colors[1] = np.array([1.0, 0.0, 0.0, 1.0])  # Красный
        self.colors[2] = np.array([0.0, 0.0, 1.0, 1.0])  # Синий
        self.color_dict = dict(enumerate(self.colors, start=1))
        self.network_color_dict = {1: self.colors[2]}
        self.network_color_dict[None] = "transparent"
        self.network_color_dict[0] = "transparent"
        self.color_dict[None] = "transparent"
        self.color_dict[0] = "transparent"

        self.label_click_counter = None
        self.point_click_counter = None
        self.shapes_layer = None

        # GUI elements
        init_image = QLabel("Выберите исходное томографическое изображение")
        self.init_image_button = QPushButton("Выберите папку с томографией")
        self.select_gt_directory_button = QPushButton("Выберите папку с эталонной сегментацией")
        fp_checkbox_info = QLabel("Hide previous FP masks from versions: ")
        self.checkbox_grid_label = QGridLayout()
        self.create_labels_button = QPushButton("Start editing annotations")
        self.clicks_count_label = QLabel("Paint clicks count: ")
        self.clicks_count_segments = QLabel("Segments clicks count: ")
        self.save_feedback_version = QPushButton("Сохранить текущую маску")
        self.timer_widget = TimerWidget(self.viewer)
        self.create_label_item_array()
        self.prediction_button = QPushButton("Сформировать версию маски")
        self.add_segment_button = QPushButton("Create new segment")

        #Layout
        self.boxLayout = QVBoxLayout()
        self.boxLayout.setContentsMargins(0, 20, 0, 0)
        self.gridLayout = QGridLayout()
        self.segment_grid_layout = QGridLayout()
        self.setLayout(self.boxLayout)
        self.boxLayout.addWidget(init_image)
        self.boxLayout.addWidget(self.init_image_button)
        self.boxLayout.addWidget(self.select_gt_directory_button)
        self.boxLayout.addLayout(self.checkbox_grid_label)
        self.boxLayout.addWidget(self.save_feedback_version)
        self.boxLayout.addWidget(self.timer_widget)
        self.boxLayout.addWidget(self.prediction_button)
        self.boxLayout.addLayout(self.gridLayout)
        self.boxLayout.addLayout(self.segment_grid_layout)

        # Connections
        self.init_image_button.clicked.connect(self.on_select_init_image)
        self.select_gt_directory_button.clicked.connect(self.on_select_gt_directory)
        self.create_labels_button.clicked.connect(self.on_create_feedback_layer)
        self.save_feedback_version.clicked.connect(self.on_save_current_feedback)
        self.viewer.layers.events.inserted.connect(self.on_points_inserted)
        self.prediction_button.clicked.connect(self.on_prediction_click)

    def on_prediction_click(self):
        img = self.get_current_slice_image_data()
        original_shape = (img.height, img.width)
        img = torch.tensor(np.asarray(img.resize((128,128)).convert('L')))/255
        h,w = img.shape[-2:]
        img = img[None,None,...].float()

        pos_scribbles = np.zeros((h, w))

        neg_scribbles = np.zeros((h, w))

        pos_scribbles = self.get_scribbles_from_labels(1, (h,w))
        neg_scribbles = self.get_scribbles_from_labels(2, (h,w))
        neg_points = []
        for idx, point in enumerate(self.segment_points):
            if idx%2 != 0:
                self.neg_points.extend(self.dilation_around_point(point, 4, (128,128)))
        point_coords = torch.tensor([[ [66, 26], *self.neg_points, *self.pos_points]])
        point_labels = torch.tensor(np.array([[0] * (point_coords.shape[1])], dtype=np.int32))
        point_labels[:, len(self.neg_points)+1:] = 1

        
        if pos_scribbles is None and neg_scribbles is None:
            scribbles = None
        else: 
            scribbles = np.stack([pos_scribbles, neg_scribbles])
            scribbles = torch.from_numpy(scribbles).unsqueeze(0)

        sp_unet = ScribblePromptUNet(version="v1")
        mask_unet = sp_unet.predict(img=img, scribbles=scribbles, point_coords=point_coords, point_labels=point_labels)
        mask = F.interpolate(mask_unet, size=original_shape, mode='bilinear').squeeze()

        
        bins = np.arange(0.1, 1.1, 0.1
        quantized_mask = np.digitize(mask.cpu().numpy(), bins)  # Разбиваем на метки 1, 2, ..., 10

        # Отображаем в napari (каждый label = свой цвет)
        self.colormap_mask = label_colormap(num_colors=10)
        colors = self.colormap_mask.colors[1:10]
        colors[0] = np.array([0.133, 0.545, 0.133, 1.0])
        colors[1] = np.array([0.2, 0.7, 0.1, 1.0])  
        colors[2] = np.array([0.4, 0.8, 0.2, 1.0]) 
        colors[3] = np.array([0.6, 0.9, 0.3, 1.0])  
        colors[4] = np.array([0.8, 1.0, 0.4, 1.0]) 
        colors[5] = np.array([1.0, 1.0, 0.5, 1.0]) 
        colors[6] = np.array([1.0, 0.9, 0.0, 1.0])  
        colors[7] = np.array([1.0, 0.7, 0.0, 1.0])  
        colors[8] = np.array([1.0, 0.4, 0.0, 1.0])  



        self.network_mask.data = quantized_mask
        self.network_mask.colormap = self.colormap_mask

        self.network_mask.refresh()
        iou = self.calculate_iou(mask, self.gt_volume)
        biou = self.boundary_iou(mask, self.gt_volume)
        dice = self.dice_coefficient(mask, self.gt_volume)
        if self.label_click_counter is not None:
            label_clicks = self.label_click_counter.get_click_count()
            self.brush_iou_plotter.update_plot(iou, label_clicks)                    
        if self.point_click_counter is not None:
            segment_clicks = self.point_click_counter.get_click_count()
            seg_clicks = self.seg_click_counter.get_click_count()
            self.segment_iou_plotter.update_plot(iou, segment_clicks)                    


    def get_points_from_points_layer(self, class_value, target_size):
        points_data = self.segments_layer.data

        return points_data

    def get_scribbles_from_labels(self, class_value, target_size):
        slice_idx = self.viewer.dims.current_step[0]
        if self.feedback_layer is not None:
            labels_data = self.feedback_layer.data[slice_idx]
            scribbles = (labels_data == class_value).astype(np.uint8)
            resized_scribbles = resize(scribbles, target_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            return resized_scribbles

    def get_current_slice_image_data(self) -> Image:
        """Получение текущего среза исходной томографии"""
        image = Image.open(self.init_image_files[0])
        return image

    def on_select_init_image(self):
        """Инициализация исходного томографического изображения"""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_():
            self.init_image_path = dialog.selectedFiles()[0]
            self.init_image_files = []
            for filename in os.listdir(self.init_image_path):
                file_path = os.path.join(self.init_image_path, filename)
                if os.path.isfile(file_path):
                    self.init_image_files.append(file_path)
            self.viewer.open(self.init_image_path)
            self.init_image = self.viewer.layers.selection.active
            self.init_image.blending = 'additive'
            shape = self.init_image.data.shape
            empty_labels = np.zeros(shape, dtype=np.uint8)
            self.network_mask = self.viewer.add_labels(empty_labels, name="Output mask")
            self.network_mask.face_color = index_to_color_map[2]


    def on_select_versions_directory(self):
        """Инициализация списка файлов с версиями разметки"""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_():
            self.versions_directory_path = dialog.selectedFiles()[0]
            self.fill_feedback_versions_list()

    def on_select_gt_directory(self):
        """Инициализация объема с ground truth"""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_():
            self.versions_gt_directory_path = dialog.selectedFiles()[0]
        self.gt_files_list = [os.path.join(self.versions_gt_directory_path, d) for d in os.listdir(self.versions_gt_directory_path) if os.path.isfile(os.path.join(self.versions_gt_directory_path, d))]
        first_image_path = self.gt_files_list[0]
        self.viewer.open(self.versions_gt_directory_path)
        first_image = imread(first_image_path)
        image_height, image_width = first_image.shape[:2] 

        if len(first_image.shape) == 2: 
            self.gt_volume = np.zeros((len(self.gt_files_list), image_height, image_width), dtype=first_image.dtype)
        else:  
            num_channels = first_image.shape[2]
            self.gt_volume = np.zeros((len(self.gt_files_list), image_height, image_width, num_channels), dtype=first_image.dtype)
        
        for i, filename in enumerate(self.gt_files_list):
            image_path = os.path.join( self.versions_gt_directory_path, filename)
            image = imread(image_path)

            if image.shape[:2] != (image_height, image_width):
                print(f"Ошибка: Размер изображения '{filename}' не соответствует размеру первого изображения.")
                return None

            self.gt_volume[i] = image
        self.gt_volume = self.gt_volume[0]
        self.gt_mask = self.viewer.add_labels(self.gt_volume, name="GT mask")
        self.gt_mask.face_color = index_to_color_map[1]


        
    def on_color_index_changed(self, value):
        """Вызывается при изменении индекса цвета кисти."""
        self.brush_color_index = value

        if 'Labels' in self.viewer.layers:
           self.feedback_layer = self.viewer.layers['Labels']
           self.feedback_layer.face_color = index_to_color_map[self.brush_color_index]

    def compute_history_fp_volume(self, version_name):
        value = self.version_to_init_volume.get(version_name)
        value = value
        fp_data = self.fp_feedback_data_array[0]
        for version in self.fp_feedback_data_array[1:]:
            fp_data |= version
        indices = np.where((value == 255) & fp_data == 1)
        self.data_without_fp[version_name] = value.copy()
        self.data_without_fp[version_name][indices] = 0
        

    def on_create_feedback_layer(self):
        """Создание слоя обратной связи и начало процесса разметки."""
        self.timer_widget.start_time = datetime.now()
        if self.init_image is None:
            print("Please select an init image first.")
            return

        shape = self.init_image.data.shape
        labels_data = np.zeros(shape, dtype=np.uint8)

        self.feedback_layer = self.viewer.add_labels(labels_data, name="Current feedback")

        self.feedback_layer.face_color = index_to_color_map[self.brush_color_index]

        self.feedback_layer.mode = 'paint'
        self.viewer.layers.selection.active = self.feedback_layer

        self.initialise_widget()
        self.label_click_counter = ClickCounter(self.feedback_layer, self.clicks_count_label)


    def on_save_current_feedback(self):
        """Сохранение объема с текущей обратной связью"""
        for layer in self.viewer.layers:
            if layer.name == "Current feedback":
                filepath = os.path.join(self.versions_directory_path, "feedback")
                feedback_versions = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
                filepath = os.path.join(filepath, str(len(feedback_versions) + 1))
                if np.save(filepath, layer.data):
                   #TODO Show message
                   msg = QMessageBox()
                   msg.setWindowTitle("info")
                   msg.setText("Feedback was saved")

    def fill_feedback_versions_list(self):
        """Формирование объема из файлов с обратной связью"""
        if self.versions_directory_path:
            self.feedback_versions_path = os.path.join(self.versions_directory_path, "feedback")
            self.feedback_versions_list = [f for f in os.listdir(self.feedback_versions_path)]

            for i, file in enumerate(self.feedback_versions_list):
                self.fp_feedback_data_array.append(np.load(os.path.join(self.feedback_versions_path, file)))


    def on_feedback_checkbox_click(self, state):
        self.compute_history_fp_volume(self.current_version_name)
        if state == 0:
             self.viewer.layers.selection.active.data =  self.version_to_init_volume[self.current_version_name]
        elif state == 1:
             self.viewer.layers.selection.active.data = self.data_without_fp[self.current_version_name]
        
            
    def create_label_item_array(self):
        """
        Создает список label_items с количеством классов, соответствующим слою labels."""
        if self.feedback_layer is not None and self.feedback_layer.data.max() > 0:
            for i in range(self.feedback_layer.data.max()):
                entry = LabelItem(i + 1, self.feedback_layer, self.color_dict)
                entry.set_color_dictionary(self.color_dict)
                self.label_items_array.append(entry)

    def create_segment_item(self):
        """
        Создает список segment_item"""
        points_count = len(self.segments_layer.data)
  
        if points_count > len(self.segment_items_array):
            index = len(self.segment_items_array)
            entry = SegmentItem(index, self.segments_layer)
            self.segment_items_array.append(entry)

            cur_qWidgets_list = entry.get_qWidget_list()
            for j in range(len(cur_qWidgets_list)):
                self.segment_grid_layout.addWidget(cur_qWidgets_list[j], index, j)
        


    def create_standard_label_item_array(self):
        """
        Создает список Label_items TP, FP, unlabelled."""
        names = ["TP", "FP", "unlabelled"]
        if self.feedback_layer is not None:
            entry = LabelItem(1, self.feedback_layer, self.color_dict, (0.0, 1.0, 0.0), names[0])
            entry.set_color_dictionary(self.color_dict)
            self.label_items_array.append(entry)

            entry = LabelItem(2, self.feedback_layer, self.color_dict, (1.0, 0.0, 0.0), names[1])
            entry.set_color_dictionary(self.color_dict)
            self.label_items_array.append(entry)


    def initialise_widget(self):
        """Инициализация виджета класса разметки."""

        self.label_items_array = []
        self.create_label_item_array()
        self.create_standard_label_item_array()
        for i in range(
            len(self.label_items_array)
        ):  
            cur_qWidgets_list = self.label_items_array[i].get_qWidget_list()
           
            for j in range(len(cur_qWidgets_list)):
                self.gridLayout.addWidget(cur_qWidgets_list[j], i, j)
       
        self.feedback_layer.colormap = self.colormap

    def calculate_iou(self, pred_mask, gt_mask):
        """
        Вычисление IoU (Intersection over Union) между ground truth изображением и разметкой.
        """

        pred_mask = (pred_mask > 0.5).numpy().astype(np.uint8)

        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        if union > 0:
            iou = intersection / union

        return iou
    
    def on_create_segment(self):
        """Добавляет отрезок в Points layer.  Первая точка - зеленая, вторая - красная."""
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(self.segments_layer)


    def on_point_added(self):
        """Обработка события добавления точки в слой отрезков"""    
        self.has_points = True
        points = self.segments_layer.data
        selected = self.segments_layer.selected_data
        self.segments_layer.blending = 'translucent'

        if len(points) %2 != 0:
            self.segments_layer.face_color[-1] = [1.0, 0.0, 0.0, 1.0]

    def on_points_inserted(self, event):
        self.segments_layer = event.value
        if isinstance(event.value, napari.layers.Points):
            self.point_click_counter = ClickCounter(self.segments_layer, self.clicks_count_segments)
        if self.shapes_layer is None and str(event.value) == "Shapes":
            self.seg_click_counter = ClickCounter(self.shapes_layer, self.clicks_count_segments)
            self.shapes_layer = event.value
            self.shapes_layer.events.data.connect(self.print_shape)
        self.segments_layer.face_color = index_to_color_map[1]
        self.segments_layer.blending = 'translucent'
        self.segments_layer.events.data.connect(self.on_point_added)

    def print_shape(self):
        if len(self.shapes_layer.data) and len(self.shapes_layer.data[0]) > 1 and len(self.shapes_layer.data[0])%2 ==0:
            first, second = self.shapes_layer.data[-1]
            print(f"first {first} second {second}")
            brightness, points = self.bresenham_line(*first, *second)
            max_diff_idx, beg, end = self.find_boundary_with_margin(brightness)
            pos_idx = [idx for idx in range(0, beg+1)]
            neg_idx = [idx for idx in range(end, len(brightness))]
            pos_points = [[points[idx][1], points[idx][0]] for idx in pos_idx]
            neg_points = [[points[idx][1], points[idx][0]] for idx in neg_idx]
            self.pos_points.extend(pos_points)
            self.neg_points.extend(neg_points)
            self.brightness_plotter.update_plot(brightness)

    def bresenham_line(self, x0, y0, x1, y1):
        print(f"x0 {x0} y0 {y0} x1 {x1} y1 {y1}")
        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        brightness_values = []
        img = self.get_current_slice_image_data()
        for point in points:
            brightness = img.getpixel((point[1], point[0]))  # napari использует (row, col) = (y, x)
            brightness_values.append(brightness)
        return brightness_values, points
    


    def boundary_iou(self, pred_mask, gt_mask, dilation_radius=3):
        """
        Вычисляет Boundary IoU между предсказанной и истинной масками.
        
        Параметры:
            pred_mask (ndarray): Предсказанная бинарная маска (0 и 1)
            gt_mask (ndarray): Истинная бинарная маска (0 и 1)
            dilation_radius (int): Радиус для расширения границ (по умолчанию 3 пикселя)
        
        Возвращает:
            float: Значение Boundary IoU [0, 1]
        """
        # Проверка размеров
        print(f"pred_mask.shape {pred_mask.shape} gt_mask.shape {gt_mask.shape}")
        assert pred_mask.shape == gt_mask.shape, "Маски должны иметь одинаковый размер"
        
        # Находим границы масок
        pred_mask = (pred_mask > 0.5).numpy().astype(np.uint8)

        pred_boundary = find_boundaries(pred_mask, mode='inner')
        gt_boundary = find_boundaries(gt_mask, mode='inner')
        
        # Расширяем границы с заданным радиусом
        if dilation_radius > 0:
            struct = np.ones((2*dilation_radius+1, 2*dilation_radius+1))
            pred_boundary = binary_dilation(pred_boundary, structure=struct)
            gt_boundary = binary_dilation(gt_boundary, structure=struct)
        
        # Вычисляем пересечение и объединение границ
        intersection = np.logical_and(pred_boundary, gt_boundary).sum()
        union = np.logical_or(pred_boundary, gt_boundary).sum()
        
        # Избегаем деления на ноль
        boundary_iou = intersection / union if union > 0 else 0.0
        
        return boundary_iou
    
    def dice_coefficient(self, pred_mask, gt_mask, epsilon=1e-6):
        """
        Вычисляет Dice Coefficient между предсказанной и истинной масками.
        
        Параметры:
            pred_mask (ndarray): Предсказанная бинарная маска (0 и 1)
            gt_mask (ndarray): Истинная бинарная маска (0 и 1)
            epsilon (float): Малое число для избежания деления на ноль
        
        Возвращает:
            float: Значение Dice Coefficient [0, 1]
        """
        # Проверка размеров
        assert pred_mask.shape == gt_mask.shape, "Маски должны иметь одинаковый размер"
        
        # Преобразование в бинарные массивы (на случай, если значения не 0/1)
        pred_mask = (pred_mask > 0.5).numpy().astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        
        # Вычисление пересечения и объединения
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        sum_masks = pred_mask.sum() + gt_mask.sum()
        
        # Формула Dice: (2 * |X ∩ Y|) / (|X| + |Y|)
        dice = (2. * intersection + epsilon) / (sum_masks + epsilon)
        
        return dice

    def find_boundary_with_margin(self, intensity_profile, margin_ratio=0.1):
        """y
        Находит границу между объектом и фоном по максимальному перепаду яркостин
        и добавляет отступы в обе стороны.
        
        Параметры:
            intensity_profile (np.array): Массив значений яркости вдоль отрезка
            margin_ratio (float): Доля отрезка для отступа (по умолчанию 1/5)
        
        Возвращает:
            tuple: (индекс границы, индекс левой границы с отступом, индекс правой границы с отступом)
        """
        # 1. Вычисляем разницы между соседними пикселями
        diffs = np.abs(np.diff(intensity_profile))
        
        # 2. Находим точку с максимальным перепадом яркости
        max_diff_idx = np.argmax(diffs)
        
        # 3. Вычисляем размер отступа (1/5 длины отрезка)
        pos_margin = 0.3
        neg_margin = 0.1
        margin_right = int(len(intensity_profile) * pos_margin)
        margin_left = int(len(intensity_profile) * neg_margin)
        
        # 4. Определяем границы с отступами
        left_bound = max(0, max_diff_idx - margin_right)
        right_bound = min(len(intensity_profile)-1, max_diff_idx + margin_left)
        
        return max_diff_idx, left_bound, right_bound


    def find_connected_components(self, matrix, start_point, connectivity=8):
        """
        Находит все точки, связанные с начальной точкой, с заданной степенью связности.
        
        Параметры:
            matrix (numpy.ndarray): 2D матрица (бинарное изображение, где 1 - объект, 0 - фон)
            start_point (tuple): (y, x) - начальная точка для поиска компоненты
            connectivity (int): 4 или 8 (по умолчанию 8)
        
        Возвращает:
            list: список координат (y, x) точек компоненты связности
        """
        if connectivity not in [4, 8]:
            raise ValueError("Связность должна быть 4 или 8")
        
        # Проверка валидности начальной точки
        y, x = start_point
        if matrix[y, x] == 0:
            return []
        
        # Направления для поиска соседей
        if connectivity == 4:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # 8-связность
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        rows, cols = matrix.shape
        visited = np.zeros_like(matrix, dtype=bool)
        component = []
        queue = deque([start_point])
        visited[start_point] = True
        
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < rows and 0 <= nx < cols and 
                    matrix[ny, nx] == 1 and not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        
        return component


    def dilation_around_point(self, center, radius, shape):
        """
        Выполняет дилатацию вокруг точки и возвращает список координат точек, входящих в дилатацию.

        Параметры:
        -----------
        center : tuple
            Координаты центральной точки (x, y).
        radius : int
            Радиус дилатации (в пикселях).
        shape : tuple
            Размеры изображения или области (height, width), чтобы не выходить за границы.

        Возвращает:
        -----------
        list of tuples
            Список координат (x, y) точек, входящих в дилатацию.
        """
        x_center, y_center = center
        height, width = shape

        # Создаем сетку координат вокруг центра в пределах радиуса
        points = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = x_center + dx
                y = y_center + dy

                # Проверяем, что точка внутри изображения и в пределах круга (если нужна круговая дилатация)
                if 0 <= x < width and 0 <= y < height and (dx**2 + dy**2) <= radius**2:
                    points.append((x, y))

        return points