import napari


class ClickCounter:
    def __init__(self, layer, clicks_count_label):
        self.layer = layer
        self.click_count = 0
        self.clicks_count_label = clicks_count_label
        print(f'layer {type(layer)}')
        if isinstance(layer, napari.layers.Labels):
            self.layer.events.paint.connect(self.on_paint)
        elif isinstance(layer, napari.layers.Points):
            self.layer.events.data.connect(self.on_insert_point)
            self.num_points = 0
        #self.feedback_layer.events.brush_size.connect(self.on_paint)

    def on_paint(self, event):
        # Каждое событие paint считаем кликом
        self.click_count += 1
        self.clicks_count_label.setText(f"Paint clicks count: {self.click_count}")

    def on_insert_point(self):
        points = self.layer.data
        self.click_count += (len(points) - self.num_points)
        self.clicks_count_label.setText(f"Segments clicks count: {self.click_count}")
        self.num_points = len(points)


    def get_click_count(self):
        return self.click_count