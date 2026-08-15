import torch
from torch import nn
from loguru import logger

from src.miftr.backbone import CNN_backbone
from src.miftr.miftr import MIFTr as MIFTr_model


class MIFTr(nn.Module):
    def __init__(self, pretrained, config, model_type='full') -> None:
        super().__init__()
        self.backbone = CNN_backbone(model_type=model_type)
        self.matcher = MIFTr_model(config=config, model_type=model_type)

        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state_dict["state_dict"], strict=True)
            logger.info(f"Load \'{pretrained}\' as pretrained checkpoint")

    def forward(self, data):
        self.backbone(data)
        self.matcher(data)
        return


cfg = {
    'coarse': {
        'd_model': 256,
    },
    'match_coarse': {
        # 'use_sm': True,
        'use_sm': False,
        'border_rm': 2,
        'dsmax_temperature': 0.1,
        'thr': 0.2,
        'inference': True
    },
    'fine': {
        'd_model': 64,
        'dsmax_temperature': 0.1,
        'thr': 0.1,
        'inference': True
    },
    'fine_window_size': 5,
    'resolution': [8, 2],
    'dense': False,
    # 'dense': True,
}
