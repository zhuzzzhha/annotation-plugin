from pathlib import Path

import napari
import numpy as np
import skimage.measure
from napari.resources import _icons
from napari.utils import DirectLabelColormap
from qtpy.QtCore import QSize
from qtpy.QtGui import QColor, QIcon
from qtpy.QtWidgets import QCheckBox, QColorDialog, QPushButton, QSlider

class SegmentItem:
    """
    This object contains QWidgets for a segment tool.
    Each Button has its own methods.
    """

    def __init__(self, index, layer, name=None):
        self.label = index  # number of the label
        self.layer = layer  # associated image/labels layer
        # assign color dictionary
        self.active = False  # state if it is selected for drawing
        self.visible = True  # shown in the viewer or not
        

        # QWidget label name/number
        if name is None:
            name = "Segment #" + str(self.label)
        self.qSegment = QPushButton(name)
        # set label color as background

        # QWidget erase label
        self.qSlider = QSlider()
        self.qSlider.setMaximum(255)
        self.qSlider.setMinimum(0)

    
        # List of QWidgets
        self.qWidget_list = []
        self.qWidget_list.append(self.qSegment)
        self.qWidget_list.append(self.qSlider)

        # Connect the qLabels button for selecting the corresponding label
        self.qSegment.clicked.connect(self._onClick_select_Label)
        


    #          Label_item class methods         #

    def _onClick_restore_label(self):
        """
        Restores the label that has been saved after erasing.
        Implemented up to 4D images
        """
        # 1D image
        if self.mem.shape[0] == 1:
            for x in self.mem:
                self.layer.data[x] = self.label
        # 2D image
        elif self.mem.shape[0] == 2:
            for i in range(self.mem.shape[1]):
                self.layer.data[self.mem[0][i]][self.mem[1][i]] = self.label
        # 3D image
        elif self.mem.shape[0] == 3:
            for i in range(self.mem.shape[-1]):
                self.layer.data[self.mem[0][i]][self.mem[1][i]][
                    self.mem[2][i]
                ] = self.label
        # 4D image
        elif self.mem.shape[0] == 4:
            for i in range(self.mem.shape[-1]):
                t = self.mem[0][i]
                z = self.mem[1][i]
                y = self.mem[2][i]
                x = self.mem[3][i]
                self.layer.data[t][z][y][x] = self.label
        else:
            raise NotImplementedError(
                f"Error: The restore function for {self.mem.shape[0]}"
                f"-dimensional images is not implemented."
            )

        self.layer.refresh()
        self.qRestore.setDisabled(True)
        # delete the memorised label drawing
        self.mem = None
        print(f"Label #{self.label} has been restored.")

    
    import numpy as np


    def _onClick_select_Label(self):
        """
        (clicking button) selects the corresponding label
        """
        self.layer.selected_label = self.label

    def get_qWidget_list(self):
        """
        Getter of the QWidget list.
        :return: array of QWidgets (class variable)
        """
        return self.qWidget_list


    def get_pixel_intensities_along_line(image, start, end):
        """
        Gets the pixel intensities along a line in an image.

        Args:
            image (np.ndarray): The image to sample from.
            start (tuple): The (row, col) coordinates of the starting point of the line.
            end (tuple): The (row, col) coordinates of the ending point of the line.

        Returns:
            np.ndarray: An array of pixel intensities along the line.
        """

        # Get the row and column indices of the pixels along the line
        rr, cc = line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))

        # Ensure that the line stays within the image bounds
        rr = np.clip(rr, 0, image.shape[0] - 1) # row
        cc = np.clip(cc, 0, image.shape[1] - 1) # col

        # Extract the pixel intensities along the line
        intensities = image[rr, cc]

        return intensities
