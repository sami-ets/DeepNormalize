from kerosene.events.preprocessors.base_preprocessor import EventPreprocessor
from kerosene.training.state import TrainerState
from kerosene.events import Event


class PrintTrainLoss(EventPreprocessor):

    def __call__(self, event: Event, state: TrainerState) -> str:
        return "".join(list(map(
            lambda model_state: "Model: {}, Train Loss: {}, Validation Loss: {} ".format(model_state.name,
                                                                                        model_state.train_loss.item(),
                                                                                        model_state.valid_loss.item()),
            state.model_trainer_states)))
