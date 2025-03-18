import pathlib
import cv2
import torch
import torch.nn.functional as F
from datetime import datetime
from magicgui import magicgui
from matplotlib import pyplot as plt
import napari
import numpy as np
from vedo import Volume
import os
from pathlib import Path
from qtpy import QtWidgets, QtCore
from PIL import Image
from skimage.io import imread
from skimage.measure import label
from napari.layers import Image as napari_img
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QCheckBox,
    QMessageBox)
from qtpy.QtCore import Qt
from napari.utils.colormaps import (
    DirectLabelColormap,
    label_colormap,
)

from click_counter import ClickCounter
from label_widget import LabelItem
from model import ScribblePromptUNet, show_mask, show_scribbles
from plotter_widget import IoUPlotter
from predictor import Predictor
from segment_widget import SegmentItem
from timer_widget import TimerWidget
from skimage.transform import resize

_maxLabels = 5
index_to_color_map = {
    0: "red",
    1: "green",
    2: "purple"
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
        self.image_layer = self.viewer.layers.selection.active
        self.feedback_layer = None
        self.shapes_layer = self.viewer.add_shapes([], name="Segments")
        self.segements_count = 0
        self.fp_feedback_data_array = []
        self.iou = []
        self.clicks = []

        self.versions_directory_path = None
        self.versions_list = []
        self.current_version_dir_path = None
        self.current_version_volume = None

        self.brush_color_index = 0
        self.label_items_array = []
        self.segment_items_array = []
        self.version_to_init_volume = dict()
        self.data_without_fp = dict()
        self.ground_truth_path = pathlib.Path("D:\\reconstruction\\mask_versions\\3")
        self.iou_plotter = IoUPlotter(self.viewer)
        self.viewer.window.add_dock_widget(self.iou_plotter, area='right', name="IoU/clicks")

        
        self.colormap = label_colormap(num_colors=3)

        # Получаем цвета из colormap, начиная с 1 (чтобы избежать фона)
        colors = self.colormap.colors[1:4]

        # Назначаем красный и зеленый цвета первым двум классам
        colors[0] = np.array([0.0, 1.0, 0.0, 1.0])  # Красный
        colors[1] = np.array([1.0, 0.0, 0.0, 1.0])  # Зеленый

        # Создаем словарь colormap
        self.color_dict = dict(enumerate(colors, start=1))

        # Добавляем transparent цвет для None и 0
        self.color_dict[None] = "transparent"
        self.color_dict[0] = "transparent"

        # GUI elements
        init_image = QLabel("Select initial tomography image")
        self.init_image_button = QPushButton("Select tomography directory")
        version_directory_info = QLabel("Select mask version directory:")
        self.select_directory_button = QPushButton("Select Directory")
        mask_version_info = QLabel("Select version:")
        gt_directory_info = QLabel("Select mask version directory:")
        self.select_gt_directory_button = QPushButton("Select Ground truth directory")

        self.versions_combobox = QComboBox()
        self.versions_combobox.addItem("No version selected")
        fp_checkbox_info = QLabel("Hide previous FP masks from versions: ")
        self.checkbox_grid_label = QGridLayout()
        self.fp_versions_checkboxes = []


        self.class_label = QLabel("Select Class:")
        self.class_combobox = QComboBox()
        self.class_combobox.addItem("TP") 
        self.class_combobox.addItem("FP")
        self.class_combobox.addItem("Unlabelled")

        self.create_labels_button = QPushButton("Start editing annotations")
        self.clicks_count_label = QLabel("Paint clicks count: ")
        self.save_feedback_version = QPushButton("Save current feedback")
        self.timer_widget = TimerWidget(self.viewer)
        self.create_label_item_array()
        self.prediction_button = QPushButton("Make prediction")

        self.boxLayout = QVBoxLayout()
        self.boxLayout.setContentsMargins(0, 20, 0, 0)
        self.gridLayout = QGridLayout()
        self.segment_grid_layout = QGridLayout()
        self.setLayout(self.boxLayout)

        self.boxLayout.addWidget(init_image)
        self.boxLayout.addWidget(self.init_image_button)
        self.boxLayout.addWidget(version_directory_info)
        self.boxLayout.addWidget(self.select_directory_button)
        self.boxLayout.addWidget(gt_directory_info)
        self.boxLayout.addWidget(self.select_gt_directory_button)
        self.boxLayout.addWidget(mask_version_info)
        self.boxLayout.addWidget(self.versions_combobox)
        self.boxLayout.addWidget(fp_checkbox_info)
        self.boxLayout.addLayout(self.checkbox_grid_label)
        self.boxLayout.addWidget(self.class_combobox)
        self.boxLayout.addWidget(self.create_labels_button)
        self.boxLayout.addWidget(self.save_feedback_version)
        self.boxLayout.addWidget(self.clicks_count_label)
        self.boxLayout.addWidget(self.timer_widget)
        self.boxLayout.addWidget(self.prediction_button)
        self.boxLayout.addLayout(self.gridLayout)
        self.boxLayout.addLayout(self.segment_grid_layout)


        # Connections
        self.init_image_button.clicked.connect(self.on_select_init_image)
        self.select_directory_button.clicked.connect(self.on_select_directory)
        self.versions_combobox.currentIndexChanged.connect(self.on_version_selected)
        self.select_gt_directory_button.clicked.connect(self.on_gt_directory)
        self.class_combobox.currentIndexChanged.connect(self.on_color_index_changed)
        self.create_labels_button.clicked.connect(self.on_create_feedback_layer)
        self.save_feedback_version.clicked.connect(self.on_save_current_feedback)
        self.shapes_layer.events.data.connect(self.create_segment_item)
        self.prediction_button.clicked.connect(self.on_prediction_click)

    def on_prediction_click(self):
        img = self.get_current_slice_image_data()
        original_shape = (img.height, img.width)
        img = torch.tensor(np.asarray(img.resize((128,128)).convert('L')))/255
        h,w = img.shape[-2:]
        img = img[None,None,...].float()

        pos_scribbles = np.zeros((h, w))
        # pos_scribbles = cv2.line(pos_scribbles, (25,37), (35,20), color=1, thickness=1)

        neg_scribbles = np.zeros((h, w))
        # neg_scribbles = cv2.line(neg_scribbles, (10,100), (110,40), color=1, thickness=1)

        pos_scribbles = self.get_scribbles_from_labels(1, (h,w))
        neg_scribbles = self.get_scribbles_from_labels(2, (h,w))
        #points = self.get_points_from_points_layer(1, (h,w))

        scribbles = np.stack([pos_scribbles, neg_scribbles])
        scribbles = torch.from_numpy(scribbles).unsqueeze(0)


        sp_unet = ScribblePromptUNet(version="v1")
        mask_unet = sp_unet.predict(img=img, scribbles=scribbles)
        mask = F.interpolate(mask_unet, size=original_shape, mode='bilinear').squeeze()
        
        mask = mask > 0.5
        mask = mask.numpy().astype(np.uint8)
        slice_idx = self.viewer.dims.current_step[0]
        self.network_mask.data[slice_idx] = mask
        self.network_mask.refresh()
        iou = self.calculate_iou()
        clicks = self.click_counter.get_click_count()
        self.iou_plotter.update_plot(iou, clicks)

    def get_points_from_points_layer(self, class_value, target_size):
        slice_idx = self.viewer.dims.current_step[0]
        points_data = self.points_layer.data
        resized_points = resize(points_data, target_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        return resized_points

    def get_scribbles_from_labels(self, class_value, target_size):
        slice_idx = self.viewer.dims.current_step[0]
        if self.feedback_layer is not None:
            labels_data = self.feedback_layer.data[slice_idx]
            scribbles = (labels_data == class_value).astype(np.uint8)
            resized_scribbles = resize(scribbles, target_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            return resized_scribbles


    def get_current_slice_image_data(self) -> Image:
        slice_idx = self.viewer.dims.current_step[0]
        image = Image.open(self.init_image_files[slice_idx])
        return image

    def on_select_init_image(self):
        """Открывает диалоговое окно выбора томографии."""
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
            shape = self.init_image.data.compute().shape
            empty_labels = np.zeros(shape, dtype=np.uint8)
            self.network_mask = self.viewer.add_labels(empty_labels, name="Output mask")
            self.points_layer = self.viewer.add_points([], name="Points")


    def on_select_directory(self):
        """Открывает диалоговое окно выбора директории."""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_():
            self.versions_directory_path = dialog.selectedFiles()[0]
            self.fill_versions_list()
            self.fill_feedback_versions_list()

    def on_gt_directory(self):
        """Открывает диалоговое окно выбора директории c ground truth."""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        if dialog.exec_():
            self.versions_gt_directory_path = dialog.selectedFiles()[0]
        self.gt_files_list = [os.path.join(self.versions_gt_directory_path, d) for d in os.listdir(self.versions_gt_directory_path) if os.path.isfile(os.path.join(self.versions_gt_directory_path, d))]
        first_image_path = self.gt_files_list[0]
        first_image = imread(first_image_path)
        image_height, image_width = first_image.shape[:2] # обрабатываем и чб и цветные картинки

        # 3. Создаем NumPy array для хранения объемного изображения.
        #    Размерность: (количество файлов, высота, ширина, количество каналов (если есть))
        if len(first_image.shape) == 2: # Ч/Б изображение
            self.gt_volume = np.zeros((len(self.gt_files_list), image_height, image_width), dtype=first_image.dtype)
        else:  # Цветное изображение
            num_channels = first_image.shape[2]
            self.gt_volume = np.zeros((len(self.gt_files_list), image_height, image_width, num_channels), dtype=first_image.dtype)


        # 4. Читаем все PNG файлы и добавляем их в объемное изображение.
        for i, filename in enumerate(self.gt_files_list):
            image_path = os.path.join( self.versions_gt_directory_path, filename)
            image = imread(image_path)

            # Проверка соответствия размеров изображения (важно!)
            if image.shape[:2] != (image_height, image_width):
                print(f"Ошибка: Размер изображения '{filename}' не соответствует размеру первого изображения.")
                return None

            self.gt_volume[i] = image
        


    def on_color_index_changed(self, value):
        """Вызывается при изменении индекса цвета кисти."""
        self.brush_color_index = value

        if 'Labels' in self.viewer.layers:
           self.feedback_layer = self.viewer.layers['Labels']
           self.feedback_layer.face_color = index_to_color_map[self.brush_color_index]

    def compute_history_fp_volume(self, version_name):
        value = self.version_to_init_volume.get(version_name)
        value = value.compute()
        fp_data = self.fp_feedback_data_array[0]
        for version in self.fp_feedback_data_array[1:]:
            fp_data |= version
        indices = np.where((value == 255) & fp_data == 1)
        self.data_without_fp[version_name] = value.copy()
        self.data_without_fp[version_name][indices] = 0
        

    def on_create_feedback_layer(self):
        """Создает слой feedback."""
        self.timer_widget.start_time = datetime.now()
        if self.image_layer is None or not isinstance(self.image_layer, napari.layers.Image):
            print("Please select an image layer first.")
            return

        shape = self.image_layer.data.shape
        labels_data = np.zeros(shape, dtype=np.uint8)

        self.feedback_layer = self.viewer.add_labels(labels_data, name="Current feedback")

        self.feedback_layer.face_color = index_to_color_map[self.brush_color_index]

        self.feedback_layer.mode = 'paint'
        self.viewer.layers.selection.active = self.feedback_layer

        self.initialise_widget()

        self.click_counter = ClickCounter(self.feedback_layer, self.clicks_count_label)

    def on_save_current_feedback(self):
        for layer in self.viewer.layers:
            if layer.name == "Current feedback":
                filepath = os.path.join(self.versions_directory_path, "feedback")
                feedback_versions = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
                filepath = os.path.join(filepath, str(len(feedback_versions) + 1))
                if np.save(filepath, layer.data):
                   #TODO Show message
                   msg = QMessageBox()  # Pass the parent widget
                   msg.setWindowTitle("info")
                   msg.setText("Feedback was saved")

    def fill_versions_list(self):
        """Заполняет выпадающий список папками с версиями из выбранной директории."""
        self.versions_combobox.clear()
        if self.versions_directory_path:
            try:
                self.versions_list = [d for d in os.listdir(self.versions_directory_path) if os.path.isdir(os.path.join(self.versions_directory_path, d)) and d!="feedback"]
                if self.versions_list:
                    self.versions_combobox.addItems(self.versions_list)
                else:
                    self.versions_combobox.addItem("No folders found in directory")
            except OSError as e:
                print(f"Error reading directory: {e}")
                self.versions_combobox.addItem("Error reading directory")
        else:
            self.versions_combobox.addItem("No version selected")


    def fill_feedback_versions_list(self):
        if self.versions_directory_path:
            self.feedback_versions_path = os.path.join(self.versions_directory_path, "feedback")
            self.feedback_versions_list = [f for f in os.listdir(self.feedback_versions_path)]

            for i, file in enumerate(self.feedback_versions_list):
                self.fp_feedback_data_array.append(np.load(os.path.join(self.feedback_versions_path, file)))

            check_box = QCheckBox()
            check_box.clicked.connect(self.on_feedback_checkbox_click)
            self.fp_versions_checkbox = check_box
            self.checkbox_grid_label.addWidget(QLabel(str(1)), 0, 1)
            self.checkbox_grid_label.addWidget(self.fp_versions_checkbox, 1, 1)

    def on_feedback_checkbox_click(self, state):
        self.compute_history_fp_volume(self.current_version_name)
        if state == 0:
             self.viewer.layers.selection.active.data =  self.version_to_init_volume[self.current_version_name]
        elif state == 1:
             self.viewer.layers.selection.active.data = self.data_without_fp[self.current_version_name]

    def on_version_selected(self, index):
        """Вызывается при выборе версии из выпадающего списка."""
        if index >= 0 and self.versions_list:  # Проверяем, что есть папки   
            selected_file = self.versions_list[index]
            self.current_version_dir_path = os.path.join(self.versions_directory_path, self.versions_list[index])
            self.current_version_name = self.versions_list[index]
            self.load_stack()
            self.current_version_volume = self.image_layer.data
        else:
            print("No version selected")

    def load_stack(self):
        """Загружает все файлы в директории как стек."""
        self.viewer.open(self.current_version_dir_path)
        self.image_layer = self.viewer.layers.selection.active
        self.version_to_init_volume[self.current_version_name] = self.viewer.layers.selection.active.data
        
            
    def create_label_item_array(self):
        """
        Создает список label_items с количеством классов, соответствующим слою labels."""
        if self.feedback_layer is not None and self.feedback_layer.data.max() > 0:
            for i in range(self.feedback_layer.data.max()):
                entry = LabelItem(i + 1, self.feedback_layer, self.color_dict)
                entry.set_color_dictionary(self.color_dict)
                self.label_items_array.append(entry)

    def _on_layer_selection_change(self, *args, **kwargs):
        for layer in self.viewer.layers:
            if self._delete_action in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.remove(self._delete_action)

        self.active_layer = self.viewer.layers.selection.active
        if self.active_layer is not None:
            if self._delete_action not in self.active_layer.mouse_drag_callbacks:
                if isinstance(self.active_layer, napari.layers.Shapes):
                    self.active_layer.mouse_drag_callbacks.append(self._delete_action)

    def _delete_action(self, source_layer, event):
        layer_value = source_layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if layer_value is not None:
            new_labels = source_layer.data
            new_labels[new_labels == layer_value] = 0
            source_layer.data = new_labels

    def create_segment_item(self):
        """
        Создает список segment_item"""
        shapes_count = len(self.shapes_layer.data)
        if shapes_count > len(self.segment_items_array):
            index = len(self.segment_items_array)
            entry = SegmentItem(index, self.shapes_layer)
            self.segment_items_array.append(entry)

            cur_qWidgets_list = entry.get_qWidget_list()
            for j in range(len(cur_qWidgets_list)):
                self.segment_grid_layout.addWidget(cur_qWidgets_list[j], index, j)
            self.update_histogram_on_shape_change(index)


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
        self.create_label_item_array()  # populates the label_items_array
        self.create_standard_label_item_array()
        for i in range(
            len(self.label_items_array)
        ):  # basically the table rows (i+1 later to jump header)
            cur_qWidgets_list = self.label_items_array[i].get_qWidget_list()
            # basically go over the columns
            for j in range(len(cur_qWidgets_list)):
                self.gridLayout.addWidget(cur_qWidgets_list[j], i, j)
        # update the colors
        self.feedback_layer.colormap = self.colormap

    def bresenham_line(self, x0, y0, x1, y1):
        """Алгоритм Брезенхема для растеризации отрезка."""
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
        return points
    
    def plot_histogram(self, pixel_values):
        """Построение гистограммы."""
        fig, ax = plt.subplots()
        ax.hist(pixel_values, bins=256, range=(0, 256), color='black', alpha=0.75)
        ax.set_title("Гистограмма интенсивности пикселей отрезка")
        ax.set_xlabel("Интенсивность")
        ax.set_ylabel("Частота")
        plt.show()
        return fig

    def update_histogram_on_shape_change(self, index):
        if len(self.shapes_layer.data) > 0:
            # Получаем координаты отрезка
            line = self.shapes_layer.data[index]
            z = self.viewer.dims.current_step[0]
            print(f'z {z}')
            
            x0, y0 = int(line[0][0]), int(line[0][1])
            x1, y1 = int(line[1][0]), int(line[1][1])

            print(f'x0 {x0} y0 {y0} x1 {x1} y1{y1}')

            # Получаем пиксели, через которые проходит отрезок
            points = self.bresenham_line(x0, y0, x1, y1)
            image_data = self.init_image.data.compute()
            pixel_values = [image_data[z, x, y] for x, y in points if 0 <= y < image_data.shape[2] and 0 <= x < image_data.shape[1]]
            print(pixel_values)
            clipped_array = np.clip(pixel_values, 0, 1)

            # Scale the values to the range [0, 255]
            scaled_array = (clipped_array - 0) / (1 - 0) * 255

            # Convert to uint8
            uint8_array = scaled_array.astype(np.uint8)

            # Обновляем гистограмму
            self.plot_histogram(uint8_array)

    def calculate_iou(self):
        """
        Вычисляет IoU (Intersection over Union) между ground truth изображением и разметкой в napari.
        """

        # 2. Получение размеченного изображения из слоя Labels
        labels_image =  self.network_mask.data

        intersection = np.logical_and(self.gt_volume, labels_image).sum()
        union = np.logical_or(self.gt_volume, labels_image).sum()

        # 6. Вычисление IoU
        if union > 0:
            iou = intersection / union

        return iou
