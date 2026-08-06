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
        self.prim_plot = 0
        self.sec_plot = 0
        super().__init__(title, width, height, autosize, no_resize)

    def build(self):
        self.plot.setup(1, 2, row_ratios=[0.7, 0.3])
        self.prim_plot = self.plot.plot_ids[0]
        self.sec_plot = self.plot.plot_ids[1]

        # primary plot
        self.date_axis = self.plot.add_x_axis("Date", parent=self.prim_plot)
        self.price_axis = self.plot.add_y_axis("Price USD", parent=self.prim_plot)
        self.volume_axis = self.plot.add_y_axis(label="Volume", min_limit=0, lock_min=True, parent=self.prim_plot)
        # secoundary plot
        self.occilator_axis = self.plot.add_y_axis(label="Value", parent=self.sec_plot)
        self.occilator_axis = self.plot.add_x_axis(label="Date", parent=self.sec_plot)
        self.plot.show_legend(self.prim_plot)
        self.plot.show_legend(self.sec_plot)
        self.plot.link_all_x(True)
        super().build()
