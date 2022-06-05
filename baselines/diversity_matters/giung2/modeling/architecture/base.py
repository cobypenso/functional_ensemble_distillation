import torch
import torch.nn as nn
from typing import Dict, List, Tuple


__all__ = [
    "ClassificationModelBase",
]


class ClassificationModelBase(nn.Module):

    def __init__(
            self,
            backbone: nn.Module,
            classifier: nn.Module,
            pixel_mean: List[float],
            pixel_std: List[float],
        ) -> None:
        super(ClassificationModelBase, self).__init__()
        self.backbone   = backbone
        self.classifier = classifier
        self.register_buffer(
            name="pixel_mean",
            tensor=torch.Tensor(pixel_mean).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            name="pixel_std",
            tensor=torch.Tensor(pixel_std).view(-1, 1, 1),
            persistent=False,
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def _preprocess_images(
            self,
            images: torch.Tensor,
            labels: torch.Tensor,
        ) -> Tuple[torch.Tensor]:

        images = images.to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std
        if labels is not None:
            labels = labels.to(self.device)

        return images, labels

    def forward(
            self,
            images: torch.Tensor,
            labels: torch.Tensor = None,
            **kwargs,
        ) -> Dict[str, torch.Tensor]:

        outputs = dict()

        # preprocess images
        images, labels = self._preprocess_images(images, labels)
        outputs["images"] = images
        outputs["labels"] = labels

        # make predictions
        outputs.update(self.backbone(outputs["images"], **kwargs))
        outputs.update(self.classifier(outputs["features"], **kwargs))

        return outputs
