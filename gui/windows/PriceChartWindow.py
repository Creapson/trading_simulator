from gui.components.Plot import Plot
from gui.components.Window import Window

class PriceChartWindow(Window):
    def __init__(self, 
                 title: str = "Not Defined", 
                 width: int = 800, 
                 height: int = 600, 
                 autosize: bool = False, 
                 no_resize: bool = False):

        self.plot = Plot(label="PriceChartWindow")
        self.price_axis = 0
        self.volume_axis = 0
        self.date_axis = 0
        super().__init__(title, width, height, autosize, no_resize)

    def build(self):
        self.plot.setup()
        self.date_axis = self.plot.add_x_axis("Date")
        self.price_axis = self.plot.add_y_axis("Price USD")
        self.volume_axis = self.plot.add_y_axis("Volume")
        super().build()
