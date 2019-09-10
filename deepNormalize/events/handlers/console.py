from kerosene.training.trainers import Trainer
from kerosene.events import Event
from kerosene.events.handlers.console import BaseConsoleLogger


class PrintTrainLoss(BaseConsoleLogger):

    def __call__(self, event: Event, trainer: Trainer) -> str:
        return self.LOGGER.info("".join(list(map(
            lambda model_trainer: "Model: {}, Train Loss: {}, Validation Loss: {} ".format(model_trainer.name,
                                                                                     model_trainer.train_loss.item(),
                                                                                     model_trainer.valid_loss.item()),
            trainer.model_trainers))))
