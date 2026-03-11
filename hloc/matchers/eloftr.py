import sys
from copy import deepcopy
from pathlib import Path

import torch

from ..utils.base_model import BaseModel


class ELoFTR(BaseModel):
    default_conf = {
        # EfficientLoFTR repo root, e.g. third_party/EfficientLoFTR
        "repo_path": "third_party/EfficientLoFTR",
        # official checkpoint path, e.g. third_party/EfficientLoFTR/weights/eloftr_outdoor.ckpt
        "weights_path": None,
        # "full" = best quality, "opt" = best efficiency
        "model_type": "full",
        # "fp32", "mp", "fp16"
        "precision": "fp32",
        # if None, keep official default:
        #   full -> 0.2, opt -> 25
        "match_threshold": None,
        # optional manual NPE setting, e.g. [832, 832, 1024, 1024]
        "npe": None,
        "max_num_matches": None,
        # opt model's mconf is commonly rescaled for visualization in the official README.
        # for hloc this is optional, but turning it on usually makes the scores easier to interpret.
        "normalize_opt_scores": True,
        # strict loading of checkpoint
        "strict": True,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        repo_path = Path(conf["repo_path"]).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"EfficientLoFTR repo not found: {repo_path}\n"
                "Set model.repo_path to the cloned EfficientLoFTR repository root."
            )

        weights_path = conf["weights_path"]
        if weights_path is None:
            raise ValueError(
                "model.weights_path is required, e.g. 'third_party/EfficientLoFTR/weights/eloftr_outdoor.ckpt'"
            )
        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"EfficientLoFTR checkpoint not found: {weights_path}"
            )

        # Make `from src.loftr import ...` work exactly as in the official repo.
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from src.loftr import LoFTR as ELoFTR_
            from src.loftr import reparameter, opt_default_cfg, full_default_cfg
        except Exception as e:
            raise ImportError(
                "Failed to import EfficientLoFTR. Check repo_path and whether its dependencies are installed."
            ) from e

        model_type = conf["model_type"]
        if model_type not in ("full", "opt"):
            raise ValueError(f"Unknown model_type: {model_type}")

        precision = conf["precision"]
        if precision not in ("fp32", "mp", "fp16"):
            raise ValueError(f"Unknown precision: {precision}")

        cfg = deepcopy(full_default_cfg if model_type == "full" else opt_default_cfg)

        if conf["match_threshold"] is not None:
            cfg["match_coarse"]["thr"] = conf["match_threshold"]

        if conf["npe"] is not None:
            cfg["coarse"]["npe"] = list(conf["npe"])

        if precision == "mp":
            cfg["mp"] = True
        elif precision == "fp16":
            cfg["half"] = True

        self.net = ELoFTR_(config=cfg)

        ckpt = torch.load(str(weights_path), map_location="cpu")
        state_dict = (
            ckpt["state_dict"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt
            else ckpt
        )
        self.net.load_state_dict(state_dict, strict=conf["strict"])

        # Official repo says this is essential for good performance.
        self.net = reparameter(self.net)

        if precision == "fp16":
            self.net = self.net.half()

        self.model_type = model_type
        self.precision = precision

    def _normalize_opt_scores(self, scores: torch.Tensor) -> torch.Tensor:
        # Follows the official README post-processing for opt confidence visualization.
        if scores.numel() == 0:
            return scores
        lo = min(20.0, float(scores.min()))
        hi = max(30.0, float(scores.max()))
        denom = max(hi - lo, 1e-8)
        scores = (scores - lo) / denom
        return scores.clamp_(0.0, 1.0)

    def _forward(self, data):
        # Keep the same trick as hloc's LoFTR wrapper:
        # swap image0/image1 so that after swapping back, image0 becomes the refined side.
        rename = {
            "image0": "image1",
            "image1": "image0",
            "mask0": "mask1",
            "mask1": "mask0",
        }
        data_ = {rename.get(k, k): v for k, v in data.items()}

        if self.precision == "fp16":
            if data_["image0"].device.type != "cuda":
                raise RuntimeError("ELoFTR fp16 mode requires CUDA.")
            data_["image0"] = data_["image0"].half()
            data_["image1"] = data_["image1"].half()

        use_mp = self.precision == "mp" and data_["image0"].device.type == "cuda"

        with torch.inference_mode():
            if use_mp:
                with torch.autocast(device_type="cuda", enabled=True):
                    self.net(data_)
            else:
                self.net(data_)

        # data_ is in swapped order:
        #   data_["image0"] == original image1
        #   data_["image1"] == original image0
        #
        # EfficientLoFTR writes:
        #   mkpts0_f -> keypoints for swapped image0
        #   mkpts1_f -> keypoints for swapped image1
        #
        # So we switch back explicitly here.
        pred = {
            "keypoints0": data_["mkpts1_f"],
            "keypoints1": data_["mkpts0_f"],
            "scores": data_["mconf"],
        }

        if self.model_type == "opt" and self.conf["normalize_opt_scores"]:
            pred["scores"] = self._normalize_opt_scores(pred["scores"])

        top_k = self.conf["max_num_matches"]
        if top_k is not None and pred["scores"].numel() > top_k:
            keep = torch.argsort(pred["scores"], descending=True)[:top_k]
            pred["keypoints0"] = pred["keypoints0"][keep]
            pred["keypoints1"] = pred["keypoints1"][keep]
            pred["scores"] = pred["scores"][keep]

        return pred
