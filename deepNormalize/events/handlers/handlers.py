from kerosene.events import TemporalEvent, Monitor
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
