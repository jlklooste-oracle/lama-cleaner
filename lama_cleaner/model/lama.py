import os

import cv2
import numpy as np
import torch
from loguru import logger

from helper import pad_img_to_modulo, download_model, norm_img, get_cache_path_by_url
from model.base import InpaintModel
from schema import Config

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)


class LaMa(InpaintModel):
    pad_mod = 8

    def init_model(self, device, **kwargs):
        logger.info("lama.py, init_model, start")
        logger.info(device)
        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        else:
            model_path = download_model(LAMA_MODEL_URL)
        logger.info(f"Load LaMa model from: {model_path}")
        model = torch.jit.load(model_path, map_location="cuda:0")
        #model = torch.jit.load(model_path, map_location="cpu")
        model = model.to(device)
        model.eval()
        self.model = model
        self.model_path = model_path
        logger.info("lama.py, init_model, end")

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """

        logger.info("lama.py, forward, start")
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        logger.info("lama.py, forward, a")
        inpainted_image = self.model(image, mask)
        logger.info("lama.py, forward, b")

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        logger.info("lama.py, forward, end")
        return cur_res
