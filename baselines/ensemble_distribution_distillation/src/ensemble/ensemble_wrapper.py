import logging


class EnsembleWrapper:
    """This is a wrapper class that extracts saved data from a dataset structured as
    ((input, ensemble predictions, ensemble logits), labels),
    an ensemble where the ensemble member predictions are loaded from file"""

    def __init__(self, output_size, indices=None):
        self.members = list()
        self._log = logging.getLogger(self.__class__.__name__)
        self.output_size = output_size
        self.size = 0
        self.indices = indices

    def get_predictions(self, inputs):
        # Inputs should be a tuple (x, ensemble predictions, ensemble logits)

        predictions = inputs[1]

        if self.indices is not None:
            predictions = predictions[:, self.indices, :]

        return predictions

    def get_logits(self, inputs):
        # Inputs should be a tuple (x, ensemble predictions, ensemble logits)

        # original - logits were after predictions
            # logits = inputs[2]
        # New format, no predictions in saved DS
        logits = inputs[1]

        if self.indices is not None:
            logits = logits[:, self.indices, :]

        return logits








