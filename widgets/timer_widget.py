import napari
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel
from qtpy.QtCore import QTimer, QTime
from datetime import datetime

class TimerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)

        self.time_label = QLabel()
        self.start_time = None # save start

        layout = QVBoxLayout()
        layout.addWidget(self.time_label)
        self.setLayout(layout)

        self.update_time()
        self.timer.start(1000)  # Update every 1000 ms (1 second)

    def update_time(self):
        current_time = datetime.now()
        if self.start_time is not None:
            elapsed = current_time - self.start_time # elapsed time
            formatted_time = str(elapsed).split(".")[0] # Get the whole elapsed time
            #formatted_time = QTime.currentTime().toString() # Get current time

            self.time_label.setText(f"Elapsed Time: {formatted_time}") #Elapsed Time label