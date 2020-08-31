from kerosene.events.handlers.visdom import BaseVisdomHandler
import crayons as crayons
import math
from beautifultable import BeautifulTable
from kerosene.events import TemporalEvent, Monitor, Phase
from kerosene.events.handlers.console import BaseConsoleLogger
from kerosene.events.handlers.visdom import BaseVisdomHandler
from kerosene.loggers.visdom import PlotType
from kerosene.loggers.visdom.data import VisdomData
from kerosene.loggers.visdom.visdom import VisdomLogger
from kerosene.training.events import Event
from kerosene.training.trainers import Trainer


class PlotGPUMemory(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_EPOCH_END, Event.ON_VALID_EPOCH_END, Event.ON_TEST_EPOCH_END,
                        Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END, Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, variable_name, params, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)
        self._variable_name = variable_name
        self._params = params

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = None

        if self.should_handle(event):
            data = self.create_visdom_data(event, trainer)

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self, event: TemporalEvent, trainer):
        return [VisdomData(trainer.name, self._variable_name, PlotType.LINE_PLOT, event.frequency, [event.iteration],
                           trainer.custom_variables[self._variable_name],
                           params={'opts': {'xlabel': str(event.frequency), 'ylabel': "Memory Consumption (MB)",
                                            'title': "GPU {} Memory ".format(self._params.get("local_rank", 0)),
                                            'legend': ["Total", "Free", "Used"],
                                            'name': None}})]


class PlotCustomLoss(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END,
                        Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, loss_name, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)
        self._loss_name = loss_name

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = list()

        if self.should_handle(event):
            data = self.create_visdom_data(event, trainer)

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self, event: TemporalEvent, trainer):
        return [VisdomData(trainer.name, self._loss_name, PlotType.LINE_PLOT, event.frequency, [[event.iteration]],
                           [trainer.custom_variables[self._loss_name]],
                           params={'opts': {'xlabel': str(event.frequency), 'ylabel': self._loss_name,
                                            'title': "{} per {}".format(str(self._loss_name),
                                                                        str(event.frequency)),
                                            'name': str(event.phase),
                                            'legend': [str(event.phase)]}})]


class PlotCustomLinePlotWithLegend(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_EPOCH_END, Event.ON_VALID_EPOCH_END, Event.ON_TEST_EPOCH_END,
                        Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END, Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, variable_name, params, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)
        self._variable_name = variable_name
        self._params = params

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = None

        if self.should_handle(event):
            data = self.create_visdom_data(event, trainer)

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self, event: TemporalEvent, trainer):
        return [VisdomData(trainer.name, self._variable_name, PlotType.LINE_PLOT, event.frequency, [event.iteration],
                           [trainer.custom_variables[self._variable_name]],
                           params={'opts': {'xlabel': str(event.frequency), 'ylabel': self._params.get("ylabel", ""),
                                            'title': self._params.get("title",
                                                                      "{} per {}".format(str(self._variable_name),
                                                                                         str(event.frequency))),
                                            'legend': self._params.get("legend", ["Training", "Validation", "Test"]),
                                            'name': None}})]


class PrintMonitorsTable(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END,
                        Event.ON_TEST_BATCH_END]

    def __init__(self, every=1):
        super().__init__(self.SUPPORTED_EVENTS, every)
        self._monitors = {}
        self._monitors_tables = {}

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        if self.should_handle(event):
            for model, monitor in monitors.items():

                monitor_values = {**monitors[model][event.phase][Monitor.METRICS],
                                  **monitors[model][event.phase][Monitor.LOSS]}

                if model in self._monitors_tables:
                    self._monitors_tables[model].update(monitor_values, event.phase)
                else:
                    self._monitors_tables[model] = MonitorsTable(model)
                    self._monitors_tables[model].update(monitor_values, event.phase)

                self._monitors_tables[model].show()
                print("\n")

            print("\n\n\n")


class MonitorsTable(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._monitors = {}
        self._new_monitors = {}

        self.table = BeautifulTable(200)

    def append(self, values: dict, old_values: dict):
        self.table.rows.append(list(map(lambda key: self.color(values[key], old_values.get(key, None)), values.keys())))

    def update(self, monitors: dict, phase: Phase):
        self.table.clear()

        if phase not in self._monitors:
            self._monitors[phase] = monitors
        self._new_monitors[phase] = monitors

        for key, value in self._monitors.items():
            self.append(self._new_monitors[key], self._monitors[key])

        self.table.columns.header = monitors.keys()
        self.table.rows.header = list(map(lambda key: str(key), self._monitors.keys()))

    def show(self):
        self.table._compute_width()
        topline = "".join(
            ["+", "-" * (self.table._width + 1 + max(list(map(lambda key: len(str(key)), self._monitors.keys())))),
             "+"])
        print(topline)
        spc = (len(topline) - 2 - len(self.model_name)) / 2
        print("%s%s%s%s%s" % ("|", " " * math.ceil(spc), self.model_name, " " * math.floor(spc), "|"))
        print(self.table)

    def color(self, value, old_value):
        if old_value is not None:
            if value > old_value:
                return crayons.green("{} \u2197".format(value), bold=True)
            if value < old_value:
                return crayons.red("{} \u2198".format(value), bold=True)

        return crayons.white("{} \u2192".format(value), bold=True)
