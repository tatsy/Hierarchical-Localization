# hloc/extractors/dedode.py
import torch
import kornia.feature as KF
import torch.nn.functional as F

from ..utils.base_model import BaseModel


def _parse_dtype(x: str):
    x = (x or '').lower()
    if x in ('fp16', 'float16', 'half'):
        return torch.float16
    if x in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if x in ('fp32', 'float32', 'float'):
        return torch.float32
    raise ValueError(f'Unknown amp_dtype: {x}')


class DeDoDe(BaseModel):
    """
    Kornia DeDoDe wrapper for hloc.

    Output format matches hloc expectations:
      - keypoints: (B, N, 2) in pixel coordinates of the *input image* (after hloc preprocessing resize)
      - keypoint_scores: (B, N)
      - descriptors: (B, D, N)  (IMPORTANT: D dimension first for hloc's NN matcher)
    """

    default_conf = {
        # weights
        'detector_weights': 'L-C4-v2',
        'descriptor_weights': 'B-upright',  # "G-upright" also possible
        # extraction behavior
        'max_keypoints': 10000,
        'pad_if_not_divisible': True,
        'apply_imagenet_normalization': True,
        # autocast / dtype
        'amp_dtype': 'float16',  # "float32" is safer on CPU/MPS
        # clamp keypoints into image bounds (recommended when padding is enabled)
        'clamp_keypoints': True,
    }

    required_inputs = ['image']

    def _init(self, conf):
        conf = dict(conf)
        conf.pop('name', None)

        amp_dtype = _parse_dtype(conf.get('amp_dtype', 'float16'))
        self.model = KF.DeDoDe.from_pretrained(
            detector_weights=conf['detector_weights'],
            descriptor_weights=conf['descriptor_weights'],
            amp_dtype=amp_dtype,
        )

    def _forward(self, data):
        # hloc's ImageDataset gives image in [0,1], shape (B,C,H,W)
        images = data['image']

        # DeDoDe.forward returns:
        #   keypoints: (B, N, 2) in image coordinate range (pixels),
        #   scores:    (B, N),
        #   desc:      (B, N, DIM) where DIM=256 (B) or 512 (G)
        kpts, scores, desc = self.model(
            images,
            n=self.conf['max_keypoints'],
            apply_imagenet_normalization=self.conf['apply_imagenet_normalization'],
            pad_if_not_divisible=self.conf['pad_if_not_divisible'],
        )

        # Convert descriptors to (B, D, N) and ensure L2 norm along D
        desc = desc.transpose(1, 2).contiguous()  # (B, D, N)
        desc = F.normalize(desc, p=2, dim=1)

        # Optionally clamp keypoints (useful if padding leads to boundary keypoints)
        if self.conf.get('clamp_keypoints', True):
            _, _, H, W = images.shape
            kpts_x = kpts[..., 0].clamp(0.0, float(W - 1))
            kpts_y = kpts[..., 1].clamp(0.0, float(H - 1))
            kpts = torch.stack([kpts_x, kpts_y], dim=-1)

        return {
            'keypoints': kpts,
            'keypoint_scores': scores,
            'descriptors': desc,
        }
