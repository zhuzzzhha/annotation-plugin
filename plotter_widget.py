
# import napari
# from napari.utils.notifications import show_info
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from PyQt5 import QtWidgets
# from PyQt5.QtCore import Qt, QTimer
# from skimage import io
# from PyQt5.QtWidgets import QVBoxLayout, QWidget, QPushButton

# class IoUPlotter(QtWidgets.QWidget):
#     """
#     Виджет для отображения графика IoU в зависимости от количества кликов в napari.
#     """

#     def __init__(self, napari_viewer):
#         super().__init__()

#         self.viewer = napari_viewer
#         self.iou_values = []
#         self.click_counts = []

#         # Создаем Figure и Canvas для matplotlib
#         self.figure, self.ax = plt.subplots()
#         self.canvas = FigureCanvas(self.figure)

#         # Настраиваем Layout
#         layout = QVBoxLayout(self)
#         layout.addWidget(self.canvas)

#         # Кнопка для сброса графика
#         self.reset_button = QPushButton("Reset Plot")
#         self.reset_button.clicked.connect(self.reset_plot)
#         layout.addWidget(self.reset_button)


#     def update_plot(self, iou, click_count):
#         """
#         Обновляет график IoU с новыми данными.

#         Args:
#             iou (float): Новое значение IoU.
#             click_count (int): Новое количество кликов.
#         """

#         self.iou_values.append(iou)
#         self.click_counts.append(click_count)

#         self.ax.clear()  # Очищаем предыдущий график
#         self.ax.plot(self.click_counts, self.iou_values, marker='o')
#         self.ax.set_xlabel("Количество кликов")
#         self.ax.set_ylabel("IoU")
#         self.ax.set_title("IoU в зависимости от количества кликов")
#         self.ax.grid(True)
#         self.canvas.draw()  # Перерисовываем canvas

#     def reset_plot(self):
#         """
#         Сбрасывает график, удаляя все данные.
#         """
#         self.iou_values = []
#         self.click_counts = []
#         self.ax.clear()
#         self.ax.set_xlabel("Количество кликов")
#         self.ax.set_ylabel("IoU")
#         self.ax.set_title("IoU в зависимости от количества кликов")
#         self.ax.grid(True)
#         self.canvas.draw()

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

        # Создаем Figure и Canvas для matplotlib
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)  # Corrected canvas instantiation

        # Настраиваем Layout
        layout = QVBoxLayout(self)  # Corrected: Pass 'self' to the layout
        layout.addWidget(self.canvas)
        # Set size policy for the canvas
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # fix for canvas sizing


        # Кнопка для сброса графика
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

        self.ax.clear()  # Очищаем предыдущий график
        self.ax.plot(self.click_counts, self.iou_values, marker='o')
        self.ax.set_xlabel("Количество кликов")
        self.ax.set_ylabel("IoU")
        self.ax.set_title("IoU в зависимости от количества кликов")
        self.ax.grid(True)
        self.canvas.draw()  # Перерисовываем canvas
        self.canvas.update() # Add canvas update

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
        self.canvas.update() # Add canvas update