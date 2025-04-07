from enum import Enum
import pathlib
import cv2
import napari.layers
import torch
import torch.nn.functional as F
from datetime import datetime
from magicgui import magic_factory, magicgui
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
from plotter_widget import IoUPlotter
from segment_anything.sam import ScribblePromptSAM
from segment_widget import SegmentItem
from timer_widget import TimerWidget
from skimage.transform import resize
from skimage.draw import line

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

        self.brush_color_index = 0
        self.label_items_array = []
        self.colormap = label_colormap(num_colors=4)
        self.segment_items_array = []
        
        #Widgets
        self.brush_iou_plotter = IoUPlotter(self.viewer)
        self.viewer.window.add_dock_widget(self.brush_iou_plotter, area='bottom', name="IoU/clicks")
        self.segment_iou_plotter = IoUPlotter(self.viewer)
        self.viewer.window.add_dock_widget(self.segment_iou_plotter, area='bottom', name="IoU/clicks")
        
        #Colors dict
        colors = self.colormap.colors[1:5]
        colors[0] = np.array([0.0, 1.0, 0.0, 1.0])  # Зеленый
        colors[1] = np.array([1.0, 0.0, 0.0, 1.0])  # Красный
        colors[2] = np.array([0.0, 0.0, 1.0, 1.0])  # Синий
        self.color_dict = dict(enumerate(colors, start=1))
        self.network_color_dict = {1: colors[2]}
        self.network_color_dict[None] = "transparent"
        self.network_color_dict[0] = "transparent"
        self.color_dict[None] = "transparent"
        self.color_dict[0] = "transparent"

        # GUI elements
        init_image = QLabel("Select initial tomography image")
        self.init_image_button = QPushButton("Select tomography directory")
        #version_directory_info = QLabel("Select mask version directory:")
        #self.select_directory_button = QPushButton("Select Directory")
        self.select_gt_directory_button = QPushButton("Select Ground truth directory")
        fp_checkbox_info = QLabel("Hide previous FP masks from versions: ")
        self.checkbox_grid_label = QGridLayout()
        self.create_labels_button = QPushButton("Start editing annotations")
        self.clicks_count_label = QLabel("Paint clicks count: ")
        self.clicks_count_segments = QLabel("Segments clicks count: ")
        self.save_feedback_version = QPushButton("Save current feedback")
        self.timer_widget = TimerWidget(self.viewer)
        self.create_label_item_array()
        self.prediction_button = QPushButton("Make prediction")
        self.add_segment_button = QPushButton("Create new segment")

        #Layout
        self.boxLayout = QVBoxLayout()
        self.boxLayout.setContentsMargins(0, 20, 0, 0)
        self.gridLayout = QGridLayout()
        self.segment_grid_layout = QGridLayout()
        self.setLayout(self.boxLayout)
        self.boxLayout.addWidget(init_image)
        self.boxLayout.addWidget(self.init_image_button)
        #self.boxLayout.addWidget(version_directory_info)
        #self.boxLayout.addWidget(self.select_directory_button)
        self.boxLayout.addWidget(self.select_gt_directory_button)
        self.boxLayout.addWidget(fp_checkbox_info)
        self.boxLayout.addLayout(self.checkbox_grid_label)
        self.boxLayout.addWidget(self.create_labels_button)
        self.boxLayout.addWidget(self.save_feedback_version)
        self.boxLayout.addWidget(self.clicks_count_label)
        self.boxLayout.addWidget(self.clicks_count_segments)
        self.boxLayout.addWidget(self.timer_widget)
        self.boxLayout.addWidget(self.prediction_button)
        self.boxLayout.addWidget(self.add_segment_button)
        self.boxLayout.addLayout(self.gridLayout)
        self.boxLayout.addLayout(self.segment_grid_layout)
        #self.mouseBindings()

        # Connections
        self.init_image_button.clicked.connect(self.on_select_init_image)
        #self.select_directory_button.clicked.connect(self.on_select_versions_directory)
        self.select_gt_directory_button.clicked.connect(self.on_select_gt_directory)
        self.create_labels_button.clicked.connect(self.on_create_feedback_layer)
        self.save_feedback_version.clicked.connect(self.on_save_current_feedback)
        self.viewer.layers.events.inserted.connect(self.on_points_inserted)
        self.prediction_button.clicked.connect(self.on_prediction_click)
        #self.add_segment_button.clicked.connect(self.on_create_segment)

    def on_prediction_click(self):
        img = self.get_current_slice_image_data()
        height = img.height
        width = img.width
        tile_size = 128
        overlap = 16
        img_np = np.array(img)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        print(f"min {img_np.min()} max {img_np.max()}")
        tiles = []
        h, w = 128, 128
        sp_unet = ScribblePromptSAM(version="v1")
        output_mask = np.zeros((height, width))

        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                y_start = y
                x_start = x
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)

                tile = img_np[y_start:y_end, x_start:x_end]
                tile_tensor = torch.tensor(tile, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                tiles.append(tile_tensor)

                pos_scribbles = self.get_scribbles_from_labels(1, y_start, y_end, x_start, x_end)
                neg_scribbles = self.get_scribbles_from_labels(2, y_start, y_end, x_start, x_end)

                if not(pos_scribbles is None and neg_scribbles is None):
                    scribbles = np.stack([pos_scribbles, neg_scribbles])
                    scribbles = torch.from_numpy(scribbles).unsqueeze(0)
                else:
                    scribbles = None
                point_coords=None
                point_labels=None
                if self.has_points:
                    points = self.get_points_from_points_layer()
                    points_coords = []
                    points_labels = []
                    tile_points = []
                    label = False
                    for point in points:
                        label = not label
                        if y_start <= point[1] and point[1] <= y_end and x_start <= point[2] and point[2] <= x_end:
                            tile_points.append(point)
                            points_coords.append([int(point[1] - y_start), int(point[2] - x_start)])
                            points_labels.append(int(label))
                    if len(points_coords):
                        point_coords = torch.tensor(points_coords).unsqueeze(0)
                        point_labels = torch.tensor(points_labels).unsqueeze(0)
                sp_unet = ScribblePromptSAM(version="v1")
                if y_end - y_start < h or x_end - x_start < w:
                    mask_unet = np.zeros((y_end - y_start, x_end - x_start))
                    mask = mask_unet > 0.3    
                    mask = mask.astype(np.uint8)
                else:
                    mask_unet, _, _ = sp_unet.predict(img=tile_tensor, scribbles=scribbles, point_coords=point_coords, point_labels=point_labels)
                    mask = mask_unet.squeeze() > 0.5    
                    mask = mask.numpy().astype(np.uint8)
                output_mask[y_start:y_end, x_start:x_end] = mask

        slice_idx = self.viewer.dims.current_step[0]
    
        self.network_mask.colormap = DirectLabelColormap(color_dict=self.network_color_dict)
        self.network_mask.data[slice_idx] = output_mask
        self.network_mask.refresh()
        iou = self.calculate_iou()
        if self.label_click_counter is not None:
            label_clicks = self.label_click_counter.get_click_count()
            self.brush_iou_plotter.update_plot(iou, label_clicks)                    
        if self.segment_iou_plotter is not None:
            segment_clicks = self.point_click_counter.get_click_count()
            self.segment_iou_plotter.update_plot(iou, segment_clicks)                    


    def get_points_from_points_layer(self):
        points_data = self.segments_layer.data
        return points_data

    def get_scribbles_from_labels(self,
                                  class_value,
                                  y_start,
                                  y_end,
                                  x_start,
                                  x_end):
        """Получение черточек из слоя обратной связи"""
        slice_idx = self.viewer.dims.current_step[0]
        if self.feedback_layer is not None:
            labels_data = self.feedback_layer.data[slice_idx]
            scribbles = (labels_data == class_value).astype(np.uint8)
            tile = scribbles[y_start:y_end, x_start:x_end]
            return tile


    def get_current_slice_image_data(self) -> Image:
        """Получение текущего среза исходной томографии"""
        slice_idx = self.viewer.dims.current_step[0]
        image = Image.open(self.init_image_files[slice_idx])
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
            shape = self.init_image.data.compute().shape
            empty_labels = np.zeros(shape, dtype=np.uint8)
            self.network_mask = self.viewer.add_labels(empty_labels, name="Output mask")
            #self.points_layer = self.viewer.add_points([], name="Points")


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
        first_image = imread(first_image_path)
        image_height, image_width = first_image.shape[:2] # обрабатываем и чб и цветные картинки

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
            #self.update_histogram_on_shape_change(index)

    # def mouseBindings(self):
        
    #     @self.shapes_layer.mouse_drag_callbacks.append
    #     def shape_mouse_drag_callback(layer, event):
    #         ### respond to click+drag """
    #         #print('shape_mouse_drag_callback() event.type:', event.type, 'event.pos:', event.pos, '')
    #         self.lineShapeChange_callback(layer, event)
    #         yield

    #         while event.type == 'mouse_move':
    #             self.lineShapeChange_callback(layer, event)
    #             yield

    # def updateLines(self, sliceNum, index):
    #     """
    #     data: two points that make the line
    #     """
    #     selectedDataList = self.shapes_layer.selected_data
    #     index = selectedDataList[0]
    #     data = self.shapes_layer.data[index]
    #     src = data[0]
    #     dst = data[1]
    #     x, lineProfile, yFit, fwhm, leftIdx, rightIdx = self.plot_widget.analysis.lineProfile(sliceNum, src, dst, linewidth=1, doFit=True)

    #     self.updateLineIntensityPlot(x, lineProfile, yFit, leftIdx, rightIdx)

    # def _getSelectedShape(self):
    #     # selected_data is a list of int with the index into self.shapeLayer.data of all selected shapes
    #     selectedDataList = self.shapes_layer.selected_data
    #     if len(selectedDataList) > 0:
    #         index = selectedDataList.pop() # just the first selected shape
    #         shapeType = self.shapes_layer.shape_types[index]
    #         return shapeType, index, self.shapes_layer.data[index]
    #     else:
    #         return (None, None, None)

    # def lineShapeChange_callback(self, layer, event):
    #     """
    #     Callback for when user clicks+drags to resize a line shape.

    #     Responding to @self.shapeLayer.mouse_drag_callbacks

    #     update pg plots with line intensity profile

    #     get one selected line from list(self.shapeLayer.selected_data)
    #     """

    #     shapeType, index, data = self._getSelectedShape()
    #     z = self.viewer.dims.current_step[0]

    #     if shapeType == 'line':
    #         self.updateLines(z, data)

    # def updateLineIntensityPlot(self, x, oneProfile, fit=None, left_idx=np.nan, right_idx=np.nan): #, ind_lambda):
    #     """
    #     Update the pyqt graph (top one) with a new line profile

    #     Parameters:
    #         oneProfile: ndarray of line intensity
    #     """

    #     # new
    #     self.myPyQtGraphWidget.updateLinePlot(x, oneProfile, fit, left_idx, right_idx)

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

    def calculate_iou(self):
        """
        Вычисление IoU (Intersection over Union) между ground truth изображением и разметкой.
        """

        labels_image =  self.network_mask.data

        intersection = np.logical_and(self.gt_volume, labels_image).sum()
        union = np.logical_or(self.gt_volume, labels_image).sum()

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
        # elif len(points) > 1:
        #     num_points = 10
        #     point1 = points[-1]
        #     point2 = points[-2]
        #     x_values = np.linspace(point1[0], point2[0], num_points).astype(int)
        #     y_values = np.linspace(point1[1], point2[1], num_points).astype(int)
            
        #     z = self.viewer.dims.current_step[0]
        #     # Получаем значения яркости пикселей под отрезком
        #     brightness_values = self.init_image.data.compute()[z]
        #     for idx in range(x_values):
        #         print(f'brightness_values {brightness_values[y_values[idx]][x_values[idx]]}')
    def on_points_inserted(self, event):
        print(f'event.value {event.value}')
        self.segments_layer = event.value
        if isinstance(event.value, napari.layers.Points):
            self.point_click_counter = ClickCounter(self.segments_layer, self.clicks_count_segments)
        self.segments_layer.face_color = index_to_color_map[1]
        self.segments_layer.blending = 'translucent'
        self.segments_layer.events.data.connect(self.on_point_added)