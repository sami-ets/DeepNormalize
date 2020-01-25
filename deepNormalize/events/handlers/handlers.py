from kerosene.events import TemporalEvent
from kerosene.events.handlers.visdom import BaseVisdomHandler
from kerosene.loggers.visdom import PlotType
from kerosene.loggers.visdom.data import VisdomData
from kerosene.loggers.visdom.visdom import VisdomLogger
from kerosene.training.events import Event
from kerosene.training.trainers import Trainer


class PlotCustomBarBlot(BaseVisdomHandler):
    SUPPORTED_EVENTS = [Event.ON_EPOCH_END, Event.ON_TRAIN_EPOCH_END, Event.ON_VALID_EPOCH_END, Event.ON_TEST_EPOCH_END,
                        Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END, Event.ON_TEST_BATCH_END, Event.ON_BATCH_END]

    def __init__(self, visdom_logger: VisdomLogger, variable_name, plot_type: PlotType, params, every=1):
        super().__init__(self.SUPPORTED_EVENTS, visdom_logger, every)
        self._variable_name = variable_name
        self._plot_type = plot_type
        self._params = params

    def __call__(self, event: TemporalEvent, monitors: dict, trainer: Trainer):
        data = None

        if self.should_handle(event):
            data = self.create_visdom_data(event, trainer)

        if data is not None:
            self.visdom_logger(data)

    def create_visdom_data(self, event: TemporalEvent, trainer):
        if self._plot_type == PlotType.LINE_PLOT and "name" not in self._params['opts'].keys():
            self._params['opts']['name'] = str(event.phase)

        return [VisdomData(trainer.name, self._variable_name, self._plot_type, event.frequency,
                           trainer.custom_variables[self._variable_name]["X"],
                           trainer.custom_variables[self._variable_name]["Y"], self._params)]
