
import napari
import napari.layers
import numpy as np
import os
from skimage.io import imread

from magicgui import magic_factory

from annotation_widget import AnnotationWidget


@magic_factory()
def make_annotation_widget(viewer):
    return AnnotationWidget(viewer)

my_selected_set = set()
viewer = napari.Viewer()
annotation_widget = AnnotationWidget(viewer)
viewer.window.add_dock_widget(annotation_widget, area='right', name="Annotation editing")

# line_prof_layer = viewer.add_shapes(
#     np.asarray([(100, 100), (200,300)]),
#     shape_type="line",
#     edge_color="red",
#     name="Line Profile"
# )

# # добавить callback для события мышиного перетаскивания
# @line_prof_layer.mouse_drag_callbacks.append
# def profile_lines_drag(shapes_layer, event):
#     print(event.type)
#     print("Drag")



napari.run()