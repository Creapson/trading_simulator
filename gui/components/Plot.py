from typing import Dict, Any
import dearpygui.dearpygui as dpg
from pydantic import BaseModel, Field

class Plot(BaseModel):
    label: str

    plot_ids: dict[int, int] = Field(default_factory=dict, exclude=True)
    subplot_id: int =Field(default=0, exclude=True) 
    x_axis_ids: list[int | str] = Field(default_factory=list, exclude=True)
    y_axis_ids: list[int | str] = Field(default_factory=list, exclude=True)

    def uuid(self, text: str) -> str:
        return str(self.subplot_id) + "_" + text

    def apply_settings(self, sender, app_data, user_data):
        widget_id, setting = user_data

        match setting:
            case "log":
                dpg.configure_item(widget_id, scale=dpg.mvPlotScale_Log10)
            case _:
                print("Cant apply setting: ", setting)
        pass

    def show_legend(self, plot_id):
        dpg.add_plot_legend(parent=plot_id)

    def setup(
            self, 
            cols:int=1, 
            rows:int=1,
            plot_labels:list[str]=[],
            row_ratios=[],
            col_ratios=[]
              ):
        self.subplot_id = int(dpg.generate_uuid())

        with dpg.tree_node(label="Settings"):
            dpg.add_button(label="fit_view", callback=self.fit_view)

        if rows == cols == 1:
            dpg.add_plot(
                    label=self.label, 
                    tag=self.subplot_id,
                    width=-1,
                    height=-1
            )
        else:
            with dpg.subplots(rows=rows, columns=cols, width=-1, height=-1, row_ratios=row_ratios, column_ratios=col_ratios) as self.subplot_id:
                for row in range(rows):
                    for col in range(cols):
                        idx = (row * cols) + col  # Note: multiplied by cols, not rows
                        
                        label: str = "None"
                        try:
                            label = plot_labels[idx]
                        except IndexError:
                            label = "None"

                        self.plot_ids[idx] = dpg.add_plot(
                            width=-1, 
                            height=-1,
                            label=label,
                        )

    def add_x_axis(
            self, 
            label, 
            tag=None,
            parent=None,
            ):
        if parent is None: parent = self.plot_ids[0]
        if tag is None:
            tag = dpg.generate_uuid()
        else:
            tag=self.uuid(tag)
        axis_id = dpg.add_plot_axis(
                dpg.mvXAxis,
                label=label,
                scale=dpg.mvPlotScale_Linear,
                parent=parent,
                tag=tag
                )
        self.x_axis_ids.append(axis_id)
        return axis_id

    def add_y_axis(
            self, 
            label, 
            tag=None,
            min_limit=float("-inf"),
            max_limit=float("inf"),
            lock_min=False,
            lock_max=False,
            parent=None,
            ):
        if parent is None: parent = self.plot_ids[0]

        if tag is None:
            tag = dpg.generate_uuid()
        else:
            tag = self.uuid(tag)
        axis_id = dpg.add_plot_axis(
                dpg.mvYAxis, 
                label=label,
                scale=dpg.mvPlotScale_Linear,
                parent=parent,
                tag=tag,
                lock_max=lock_max,
                lock_min=lock_min,
                )
        dpg.set_axis_limits_auto(axis_id)
        self.y_axis_ids.append(axis_id)
        dpg.set_axis_limits_constraints(axis_id, min_limit, max_limit)
        return axis_id

    def remove_x_axis(self, tag):
        self.x_axis_ids.remove(self.uuid(tag))
        dpg.delete_item(self.uuid(tag))

    def remove_y_axis(self, tag):
        self.y_axis_ids.remove(self.uuid(tag))
        dpg.delete_item(self.uuid(tag))

    def clear_plot(self):
        dpg.delete_item(self.y_axis_ids, children_only=True)

    def fit_view(self):
        dpg.fit_axis_data(self.x_axis_ids[0])
        dpg.fit_axis_data(self.y_axis_ids[0])

    def link_all_x(self, should_link=True):
        dpg.configure_item(self.subplot_id, link_all_x=should_link)

    def add_line_series(
            self, 
            name:str, 
            x_values: list[float],
            y_values: list[float],
            axis_id
            ):

        print("Adding line seires")
        dpg.add_line_series(
            x_values, y_values, label=name, 
            tag=self.uuid(name),
            parent=axis_id
        )
        self.fit_view()

    def add_candlestick_series(
            self, 
            name:str,
            dates, 
            opens, 
            closes, 
            lows, 
            highs,
            axis_id
            ):
        print("Adding candlestick seires")
        dpg.add_candle_series(
            dates, 
            opens, 
            closes, 
            lows, 
            highs, 
            label=name,
            parent=axis_id,
            tag=self.uuid(name)
        )
        self.fit_view()

    def add_bar_series(
            self,
            name,
            dates,
            values,
            axis_id
            ):
        dpg.add_bar_series(
                x=dates,
                y=values,
                parent=axis_id,
                tag=self.uuid(name)
                )
