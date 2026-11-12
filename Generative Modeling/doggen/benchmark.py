"""Synchronized, batch-one latency including tokenization, transfer and PNG encoding."""
import argparse
import io
import platform
import time
import numpy as np
import torch
from .common import write_json
from .inference import Predictor, synchronize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/latest.pt')
    p.add_argument('--device', default='auto')
    p.add_argument('--runs', type=int, default=100)
    p.add_argument('--output', default='reports/latency.json')
    p.add_argument('--prompt', default='A photograph of a golden retriever dog in a bright green scene.')
    args = p.parse_args()
    if args.runs < 2:
        p.error('At least two measurements required')
    start = time.perf_counter()
    pred = Predictor(args.checkpoint, args.device)
    load = time.perf_counter()-start
    synchronize(pred.device)
    start = time.perf_counter()
    pred.generate(args.prompt).save(io.BytesIO(), format='PNG')
    synchronize(pred.device)
    cold = time.perf_counter()-start
    pred.warmup(10)
    if pred.device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(pred.device)
    values = []
    for i in range(args.runs):
        synchronize(pred.device)
        start = time.perf_counter()
        image = pred.generate(args.prompt, seed=650+i)
        image.save(io.BytesIO(), format='PNG')
        synchronize(pred.device)
        values.append(time.perf_counter()-start)
    result = dict(device=str(pred.device), hardware=torch.cuda.get_device_name(pred.device) if pred.device.type=='cuda' else platform.machine(),
                  os=platform.platform(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  checkpoint_step=pred.steps, runs=args.runs, batch_size=1, resolution=64,
                  load_seconds=load, cold_image_seconds=cold, mean_seconds=float(np.mean(values)),
                  p50_seconds=float(np.percentile(values,50)), p95_seconds=float(np.percentile(values,95)),
                  max_seconds=max(values), meets_warm_1s=max(values)<=1., meets_warm_target=max(values)<=.8,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(pred.device) if pred.device.type=='cuda' else None,
                  timing_scope='tokenization + batch-one forward + device transfer + PNG encoding; excludes checkpoint loading and disk write',
                  measurements_seconds=values, parameters=pred.parameters)
    write_json(args.output,result)
    print({k:v for k,v in result.items() if k != 'measurements_seconds'})


if __name__=='__main__':
    main()
