
import napari
import napari.layers
import numpy as np
import os
from skimage.io import imread

from magicgui import magic_factory

from annotation_widget import AnnotationWidget
from graph_widget import myPyQtGraphWidget


@magic_factory()
def make_annotation_widget(viewer):
    return AnnotationWidget(viewer)

my_selected_set = set()
viewer = napari.Viewer()
annotation_widget = AnnotationWidget(viewer)
#plot_widget = myPyQtGraphWidget(annotation_widget.shapes_layer)
viewer.window.add_dock_widget(annotation_widget, area='right', name="Annotation editing")
#viewer.window.add_dock_widget(plot_widget, area='right', name="Plot")


napari.run()    