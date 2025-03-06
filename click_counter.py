class ClickCounter:
    def __init__(self, feedback_layer, clicks_count_label):
        self.feedback_layer = feedback_layer
        self.click_count = 0
        self.clicks_count_label = clicks_count_label
        self.feedback_layer.events.paint.connect(self.on_paint)
        self.feedback_layer.events.brush_size.connect(self.on_paint)

    def on_paint(self, event):


        # Каждое событие paint считаем кликом
        self.click_count += 1
        self.clicks_count_label.setText(f"Clicks count: {self.click_count}")

    def get_click_count(self):
        return self.click_count