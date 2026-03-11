import torch

from ..utils.base_model import BaseModel


class XFeatLightGlue(BaseModel):
    required_inputs = [
        "image0",
        "keypoints0",
        "descriptors0",
        "image_size0",
        "image1",
        "keypoints1",
        "descriptors1",
        "image_size1",
    ]

    def _init(self, conf):
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        )

    def _forward(self, data):
        if self.model.lighterglue is None:
            output0 = {
                "keypoints": data["keypoints0"].squeeze(0),
                "descriptors": data["descriptors0"].squeeze(0).transpose(-1, -2),
                "scores": data["keypoint_scores0"].squeeze(0),
                "image_size": data["image_size0"].squeeze(0).detach(),
            }
            output1 = {
                "keypoints": data["keypoints1"].squeeze(0),
                "descriptors": data["descriptors1"].squeeze(0).transpose(-1, -2),
                "scores": data["keypoint_scores1"].squeeze(0),
                "image_size": data["image_size1"].squeeze(0).detach(),
            }
            mkpts0, mkpts1, matches = self.model.match_lighterglue(output0, output1)

        assert self.model.lighterglue is not None

        data["descriptors0"] = data["descriptors0"].transpose(-1, -2)
        data["descriptors1"] = data["descriptors1"].transpose(-1, -2)

        return self.model.lighterglue(data)
