import napari
from napari.utils.notifications import show_info
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Corrected import
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from skimage import io
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QPushButton, QSizePolicy # import QSizePolicy

class IoUPlotter(QWidget):
    """
    Виджет для отображения графика IoU в зависимости от количества кликов в napari.
    """

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.iou_values = []
        self.click_counts = []

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)  

        layout = QVBoxLayout(self)  
        layout.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 


        self.reset_button = QPushButton("Reset Plot")
        self.reset_button.clicked.connect(self.reset_plot)
        layout.addWidget(self.reset_button)


    def update_plot(self, iou, click_count):
        """
        Обновляет график IoU с новыми данными.

        Args:
            iou (float): Новое значение IoU.
            click_count (int): Новое количество кликов.
        """

        self.iou_values.append(iou)
        self.click_counts.append(click_count)

        self.ax.clear()  
        self.ax.plot(self.click_counts, self.iou_values, marker='o')
        self.ax.set_xlabel("Количество кликов")
        self.ax.set_ylabel("IoU")
        self.ax.set_title("IoU в зависимости от количества кликов")
        self.ax.grid(True)
        self.canvas.draw()  
        self.canvas.update() 

    def reset_plot(self):
        """
        Сбрасывает график, удаляя все данные.
        """
        self.iou_values = []
        self.click_counts = []
        self.ax.clear()
        self.ax.set_xlabel("Количество кликов")
        self.ax.set_ylabel("IoU")
        self.ax.set_title("IoU в зависимости от количества кликов")
        self.ax.grid(True)
        self.canvas.draw()
        self.canvas.update()


class BrPlotter(QWidget):
    """
    Виджет для отображения графика
    """

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)  

        layout = QVBoxLayout(self)  
        layout.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 


        self.reset_button = QPushButton("Reset Plot")
        self.reset_button.clicked.connect(self.reset_plot)
        layout.addWidget(self.reset_button)


    def update_plot(self, brightness):
        """
        Обновляет график IoU с новыми данными.

        Args:
            iou (float): Новое значение IoU.
            click_count (int): Новое количество кликов.
        """
        self.pixel_ids = np.arange(len(brightness)) 
        self.ax.clear()  
        self.ax.plot(self.pixel_ids, brightness, marker='o')
        self.ax.set_xlabel("номер пикселя на отрезке")
        self.ax.set_ylabel("Яркость")
        self.ax.grid(True)
        self.canvas.draw()  
        self.canvas.update() 

    def reset_plot(self):
        """
        Сбрасывает график, удаляя все данные.
        """
        self.brightness = []
        self.pixel_ids = []
        self.ax.clear()
        self.ax.set_xlabel("номер пикселя на отрезке")
        self.ax.set_ylabel("Яркость")
        self.ax.grid(True)
        self.canvas.draw()
        self.canvas.update()