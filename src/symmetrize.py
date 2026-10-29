from torch import nn

from .transforms import apply_group, invert_group


class SymmetrizedForecaster(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        predictions = []
        for group in range(8):
            transformed_inputs = apply_group(inputs, group)
            transformed_predictions = self.model(transformed_inputs)
            predictions.append(invert_group(transformed_predictions, group))
        return sum(predictions) / len(predictions)

