import io
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .common import device_for
from .model import TextToImage, parameter_counts
from .text import tokenize


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


class Predictor:
    def __init__(self, checkpoint, device='auto', amp=True, threads=4):
        torch.set_num_threads(threads)
        self.device = device_for(device)
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f'Missing checkpoint {checkpoint}. Train with python -m doggen.train first.')
        # Only load trusted project checkpoints: optimizer/RNG state uses pickle.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('format_version') != 2 or state.get('step', 0) <= 0:
            raise ValueError('Expected a trained project checkpoint')
        self.model = TextToImage(base=state['base']).to(self.device).eval()
        self.model.load_state_dict(state['ema'])
        self.steps = state['step']
        self.parameters = parameter_counts(self.model)
        self.amp = amp and self.device.type == 'cuda'
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def generate(self, prompt, seed=650):
        if not prompt.strip():
            raise ValueError('Empty prompt')
        tokens = tokenize([prompt], self.device)
        # CPU RNG makes each seed reproducible without depending on MPS RNG support.
        z = torch.randn(1, self.model.generator.z_dim, generator=torch.Generator().manual_seed(seed)).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            tensor = self.model(tokens, z)
        array = tensor[0].float().clamp(-1,1).add(1).mul(127.5).round().to(torch.uint8).permute(1,2,0).cpu().numpy()
        return Image.fromarray(array)

    def warmup(self, count=5):
        for _ in range(count):
            self.generate('A photograph of a golden retriever dog in a bright scene.')
        synchronize(self.device)
