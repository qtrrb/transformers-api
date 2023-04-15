import torch
from transformers import StoppingCriteria


class TextStoppingCriteria(StoppingCriteria):
    def __init__(self, criteria: torch.LongTensor | None):
        self.critera = criteria

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if (
            self.critera is not None
            and len(input_ids[0]) > len(self.critera)
            and torch.equal(input_ids[0][-len(self.critera) :], self.critera)
        ):
            return True
        return False
