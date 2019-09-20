from kerosene.training.trainers import Trainer
from kerosene.events import Event
from kerosene.events.handlers.console import BaseConsoleLogger


class PrintTrainLoss(BaseConsoleLogger):
    SUPPORTED_EVENTS = [Event.ON_BATCH_END, Event.ON_EPOCH_END, Event.ON_TRAIN_BATCH_END, Event.ON_VALID_BATCH_END]

    def __init__(self, every=1):
        super(PrintTrainLoss, self).__init__(every=every)

    def __call__(self, event: Event, trainer: Trainer) -> str:
        assert event in self.SUPPORTED_EVENTS, "Unsupported event provided. Only {} are permitted.".format(
            self.SUPPORTED_EVENTS)
        if self.should_handle_epoch_data(event, trainer):
            return self.LOGGER.info("".join(list(map(
                lambda model_trainer: "Model: {}, Train Loss: {}, Validation Loss: {} ".format(model_trainer.name,
                                                                                               model_trainer.train_loss.mean().item(),
                                                                                               model_trainer.valid_loss.mean().item()),
                trainer.model_trainers))))
        elif self.should_handle_step_data(event, trainer):
            return self.LOGGER.info("".join(list(map(
                lambda model_trainer: "Model: {}, Train Loss: {}, Validation Loss: {} ".format(model_trainer.name,
                                                                                               model_trainer.train_loss.mean().item(),
                                                                                               model_trainer.valid_loss.mean().item()),
                trainer.model_trainers))))
