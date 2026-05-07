import torch
from torch import nn
from torchvision.models.inception import inception_v3, Inception_V3_Weights

class InceptionNetV3Backbone(nn.Module):
    """
    Leveraging the Pre-Trained inception net model with approximately the first "200" layers frozen using
    transfer learning from the pretrained model with the unfrozen layers being trained on local features of the PBS data.

    Input Resolution: (224x224x3), diverges from the standard 299x299x3 inceptionNet
    typically uses, the result is a smaller feature map (5x5x2048)

    """
    # Ordered layer names matching the Sequential indices
    LAYER_NAMES = [
        "Conv2d_1a", "Conv2d_2a", "Conv2d_2b", "MaxPool_1",
        "Conv2d_3b", "Conv2d_4a", "MaxPool_2",
        "Mixed_5b", "Mixed_5c", "Mixed_5d",
        "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
        "Mixed_7a", "Mixed_7b", "Mixed_7c",
    ]

    def __init__(self, frozen_layer_index: str = "Mixed_6e", logits_setting:bool = False):
        super().__init__()
        # "False" deactivates the auxiliary classifier which is not needed since we will be feeding the input to ViT
        # Weights argument set to default is what is allowing the use of the pretrained model.
        v3BackBone = inception_v3(aux_logits=logits_setting, weights=Inception_V3_Weights.DEFAULT)

        self.features = nn.Sequential(
            v3BackBone.Conv2d_1a_3x3,
            v3BackBone.Conv2d_2a_3x3,
            v3BackBone.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            v3BackBone.Conv2d_3b_1x1,
            v3BackBone.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            v3BackBone.Mixed_5b,
            v3BackBone.Mixed_5c,
            v3BackBone.Mixed_5d,
            v3BackBone.Mixed_6a,
            v3BackBone.Mixed_6b,
            v3BackBone.Mixed_6c,
            v3BackBone.Mixed_6d,
            v3BackBone.Mixed_6e,
            v3BackBone.Mixed_7a,
            v3BackBone.Mixed_7b,
            v3BackBone.Mixed_7c,
        )

        self._freeze_layer(frozen_layer_index)

    def _freeze_layer(self, frozen_layer_index):
        """
        Helper function which takes the string name of the layer finds the numerical index
        then freezes all layers up to that index. Required in order to leverage the pretrained segments of the model
        :param frozen_layer_index:
        :return:
        """
        if frozen_layer_index is None:
            return
        if frozen_layer_index not in self.LAYER_NAMES:
            choices = " ,".join(self.LAYER_NAMES)
            raise ValueError(f"Invalid frozen layer index: {frozen_layer_index}, "
                             f"valid options are: {choices}")
        layer_index = self.LAYER_NAMES.index(frozen_layer_index)
        for index, layer in enumerate(self.features):
            if index <= layer_index:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        tokens = features.flatten(2).transpose(1, 2)
        return tokens

