from typing import Tuple

import torch

def make_linspace_points(start: Tuple[int], end: Tuple[int], steps: int) -> torch.Tensor:
    """
    Generates evenly spaced points along a linear space between start and end.

    Args:
        start (Tuple[int]): The starting coordinates for the linear space.
        end (Tuple[int]): The ending coordinates for the linear space.
        steps (int): The number of points to generate.

    Returns:
        torch.Tensor: A tensor containing evenly spaced points.
    """
    # Create a linear space for each dimension
    linspaces = [torch.linspace(start[i], end[i], steps) for i in range(len(start))]
    # Create a meshgrid from the linear spaces
    mesh = torch.meshgrid(*linspaces, indexing="ij")
    # Flatten the meshgrid and stack into a single tensor
    linspace_points = torch.stack([m.flatten() for m in mesh], dim=-1)
    return linspace_points

class DecisionSpaceEstimator:
    def __init__(self) -> None:
        self.inputs = None
        self.space_shape = None

    def estimate(self, model: torch.nn.Module, batch_size=1):
        dataloader = torch.utils.data.DataLoader(self.inputs, batch_size=batch_size)

        device = next(model.parameters()).device

        outputs = []
        for data in dataloader:
            data = data.to(device)
            # TODO: implement forward function for SAC
            outputs.append(model(data))
        outputs = torch.cat(outputs)
        outputs = outputs.reshape(self.space_shape)
        return outputs


class LinearDecisionSpaceEstimator(DecisionSpaceEstimator):
    def __init__(self, start: Tuple[int], end: Tuple[int], steps: int = 300, integer: bool = False):
        assert len(start) == len(end)
        self.start = start
        self.end = end
        self.steps = steps
        self.inputs = make_linspace_points(start, end, steps)
        if integer:
            self.inputs = self.inputs.to(torch.int64)
        self.space_shape = (steps,) * len(start)

if __name__ == "__main__":
    # Example usage
    X1, X2, Y1, Y2 = -1, 1, -1, 1
    POINTS_PER_AXIS = 4
    estimator = LinearDecisionSpaceEstimator(start=(X1, Y1), end=(X2, Y2), steps=POINTS_PER_AXIS)
    print(estimator.inputs)
