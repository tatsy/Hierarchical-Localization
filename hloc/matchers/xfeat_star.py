import torch
import torch.nn.functional as F

from ..utils.base_model import BaseModel


class XFeatStar(BaseModel):
    default_conf = {
        "repo": "verlab/accelerated_features",
        "pretrained": True,
        "top_k": 8192,
        "multiscale": True,
        "fine_conf": 0.25,
        "min_cossim": -1.0,
        "detection_threshold": 0.05,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        self.net = torch.hub.load(
            conf["repo"],
            "XFeat",
            pretrained=conf["pretrained"],
            top_k=conf["top_k"],
            detection_threshold=conf["detection_threshold"],
        ).eval()

    @torch.inference_mode()
    def _forward(self, data):
        image0 = data["image0"]
        image1 = data["image1"]

        # XFeat は内部で self.dev を使うので、hloc 側の実デバイスに同期しておく
        self.net.dev = image0.device

        # coarse semi-dense features
        out0 = self.net.detectAndComputeDense(
            image0,
            top_k=self.conf["top_k"],
            multiscale=self.conf["multiscale"],
        )
        out1 = self.net.detectAndComputeDense(
            image1,
            top_k=self.conf["top_k"],
            multiscale=self.conf["multiscale"],
        )

        # mutual NN matching on coarse descriptors
        idxs_list = self.net.batch_match(
            out0["descriptors"],
            out1["descriptors"],
            min_cossim=self.conf["min_cossim"],
        )

        # hloc.match_dense は batch_size=1
        idx0, idx1 = idxs_list[0]

        if idx0.numel() == 0:
            dev = image0.device
            empty_k = torch.empty((0, 2), device=dev, dtype=torch.float32)
            empty_s = torch.empty((0,), device=dev, dtype=torch.float32)
            return {
                "keypoints0": empty_k,
                "keypoints1": empty_k.clone(),
                "scores": empty_s,
            }

        feats0 = out0["descriptors"][0][idx0]
        feats1 = out1["descriptors"][0][idx1]
        kpts0 = out0["keypoints"][0][idx0].clone()
        kpts1 = out1["keypoints"][0][idx1].clone()
        scales0 = out0["scales"][0][idx0]

        # refine_matches() 相当だが、confidence も返したいので明示的に展開
        offsets = self.net.net.fine_matcher(torch.cat([feats0, feats1], dim=-1))
        scores = F.softmax(offsets * 3.0, dim=-1).max(dim=-1)[0]
        offsets = self.net.subpix_softmax2d(offsets.view(-1, 8, 8))
        kpts0 = kpts0 + offsets * scales0[:, None]

        keep = scores > self.conf["fine_conf"]
        return {
            "keypoints0": kpts0[keep],
            "keypoints1": kpts1[keep],
            "scores": scores[keep],
        }
