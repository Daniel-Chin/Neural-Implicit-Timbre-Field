from torchWork import BaseExperimentGroup

from hyper_params import HyperParams

class ExperimentGroup(BaseExperimentGroup):
    def __init__(self, hyperParams: HyperParams) -> None:
        self.hyperParams = hyperParams
