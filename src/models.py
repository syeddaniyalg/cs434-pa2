from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_channels: int = 64
    kernel_size: int = 3
    dilation: int = 1
    fold_factor: int = 4

    def to_dict(self):
        return asdict(self)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        padding = dilation * (kernel_size // 2)
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

    def initial_state(self, x):
        shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        return x.new_zeros(shape), x.new_zeros(shape)

    def forward(self, x, state):
        hidden, cell = state
        gates = self.gates(torch.cat((x, hidden), dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTMForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        folded_channels = config.fold_factor ** 2
        self.cell1 = ConvLSTMCell(
            folded_channels,
            config.hidden_channels,
            config.kernel_size,
            config.dilation,
        )
        self.cell2 = ConvLSTMCell(
            config.hidden_channels,
            config.hidden_channels,
            config.kernel_size,
            config.dilation,
        )
        self.output = nn.Conv2d(config.hidden_channels, folded_channels, kernel_size=1, bias=True)

    def forward(self, inputs):
        if inputs.shape[1:] != (5, 1, 64, 64):
            raise ValueError(f"Expected [B, 5, 1, 64, 64], received {tuple(inputs.shape)}")
        state1 = None
        state2 = None
        predictions = []
        current = None
        for step in range(9):
            frame = inputs[:, step] if step < 5 else current
            folded = F.pixel_unshuffle(frame, self.config.fold_factor)
            if state1 is None:
                state1 = self.cell1.initial_state(folded)
                state2 = self.cell2.initial_state(folded)
            hidden1, cell1 = self.cell1(folded, state1)
            hidden2, cell2 = self.cell2(hidden1, state2)
            state1 = hidden1, cell1
            state2 = hidden2, cell2
            current = torch.sigmoid(F.pixel_shuffle(self.output(hidden2), self.config.fold_factor))
            if step >= 4:
                predictions.append(current)
        return torch.stack(predictions, dim=1)


def config_for(model_name):
    model_name = model_name.upper()
    if model_name not in {"A", "B", "C", "D", "DB"}:
        raise ValueError(model_name)
    dilation = 2 if model_name in {"D", "DB"} else 1
    return ModelConfig(name=model_name, dilation=dilation)


def build_model(model_name):
    return ConvLSTMForecaster(config_for(model_name))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def one_step_reach(model_name):
    config = config_for(model_name)
    radius_per_convolution = (config.kernel_size // 2) * config.dilation
    folded_cells = 2 * radius_per_convolution
    return folded_cells * config.fold_factor


def validate_architectures():
    baseline = build_model("A")
    larger = build_model("D")
    baseline_parameters = parameter_count(baseline)
    larger_parameters = parameter_count(larger)
    if baseline_parameters != 480_784:
        raise AssertionError(baseline_parameters)
    if larger_parameters > 721_176:
        raise AssertionError(larger_parameters)
    if one_step_reach("D") <= one_step_reach("A"):
        raise AssertionError((one_step_reach("A"), one_step_reach("D")))
    return {
        "A_parameters": baseline_parameters,
        "A_reach": one_step_reach("A"),
        "D_parameters": larger_parameters,
        "D_reach": one_step_reach("D"),
    }

