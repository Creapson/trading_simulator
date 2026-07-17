from gui.components.Plot import Plot
from gui.components.Window import Window

class PriceChartWindow(Window):
    def __init__(self, 
                 title: str = "Not Defined", 
                 width: int = 800, 
                 height: int = 600, 
                 autosize: bool = True, 
                 no_resize: bool = False):

        self.plot = Plot()
        super().__init__(title, width, height, autosize, no_resize)

    def build(self):
        self.plot.setup()
        super().build()
