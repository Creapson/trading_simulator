from typing import Dict, Any
import dearpygui.dearpygui as dpg
from pydantic import BaseModel, Field

class Plot(BaseModel):
    label: str

    plot_id: int = Field(default=0, exclude=True) 
    x_axis_ids: list[int | str] = Field(default_factory=list)
    y_axis_ids: list[int | str] = Field(default_factory=list)

    def uuid(self, text: str) -> str:
        return str(self.id) + "_" + text

    def apply_settings(self, sender, app_data, user_data):
        widget_id, setting = user_data

        match setting:
            case "log":
                dpg.configure_item(widget_id, scale=dpg.mvPlotScale_Log10)
            case _:
                print("Cant apply setting: ", setting)
        pass

    def setup(self):
        if self.id == 0:
            self.id = int(dpg.generate_uuid())

        self.plot_id = int(dpg.generate_uuid())
        self.x_axis_id = int(dpg.generate_uuid()) 
        self.y_axis_id = int(dpg.generate_uuid())

        with dpg.tree_node(label="Settings"):
            dpg.add_button(label="fit_view", callback=self.fit_view)

        dpg.add_plot(label=self.label, tag=self.plot_id)

    def add_x_axis(self, label):
        axis_id = dpg.add_plot_axis(
                dpg.mvXAxis,
                label=label,
                scale=dpg.mvPlotScale_Linear,
                parent=self.plot_id
                )
        self.x_axis_ids.append(axis_id)
        return axis_id

    def add_y_axis(self, label):
        axis_id = dpg.add_plot_axis(
                dpg.mvYAxis, 
                label=label,
                scale=dpg.mvPlotScale_Linear,
                parent=self.plot_id
                )
        self.y_axis_ids.append(axis_id)
        return axis_id

    def remove_x_axis(self, tag):
        self.x_axis_ids.remove(tag)
        dpg.delete_item(tag)

    def remove_y_axis(self, tag):
        self.y_axis_ids.remove(tag)
        dpg.delete_item(tag)

    def clear_plot(self):
        dpg.delete_item(self.y_axis_id, children_only=True)

    def fit_view(self):
        dpg.fit_axis_data(self.x_axis_id)
        dpg.fit_axis_data(self.y_axis_id)

    def add_line_series(self, name:str, 
                        x_values: list[float],
                        y_values: list[float],
                        axis_id):

        print("Adding line seires")
        dpg.add_line_series(
            x_values, y_values, label=name, 
            tag=self.uuid(name),
            parent=axis_id
        )
        self.fit_view()

    def add_candlestick_series(self, 
                               name:str,
                               dates, 
                               opens, 
                               closes, 
                               lows, 
                               highs):
        print("Adding candlestick seires")
        dpg.add_candle_series(
            dates, 
            opens, 
            closes, 
            lows, 
            highs, 
            label=name,
            parent=self.y_axis_id,
            tag=self.uuid(name)
        )
        self.fit_view()
