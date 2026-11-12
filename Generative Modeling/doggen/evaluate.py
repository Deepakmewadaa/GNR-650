"""Held-out FID, Inception Score and CLIP Score. Metric models are evaluation-only."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from .common import read_manifest, write_json, sha256
from .inference import Predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/curated/manifest.jsonl'))
    p.add_argument('--checkpoint',default='checkpoints/latest.pt')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--limit',type=int,default=0,help='0 means entire held-out split')
    p.add_argument('--device',default='auto')
    p.add_argument('--output',default='reports/metrics.json')
    p.add_argument('--seed',type=int,default=1650)
    args=p.parse_args()
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.multimodal.clip_score import CLIPScore
    predictor=Predictor(args.checkpoint,args.device)
    rows=[r for r in read_manifest(args.manifest) if r['split']==args.split]
    if args.limit:
        rows=rows[:args.limit]
    if len(rows)<10:
        raise ValueError('At least 10 held-out samples needed')
    # MPS lacks float64 covariance/eigendecomposition used by FID.
    device=torch.device('cpu') if predictor.device.type=='mps' else predictor.device
    # Pretrained feature extractors are used only here, never in training or generation.
    fid=FrechetInceptionDistance(feature=2048,normalize=False).to(device)
    inception=InceptionScore(normalize=False,splits=10).to(device)
    clip=CLIPScore(model_name_or_path='openai/clip-vit-base-patch32').to(device)
    with torch.inference_mode():
        for i,row in enumerate(tqdm(rows,desc='Held-out metrics')):
            generated=pil_to_tensor(predictor.generate(row['caption'],args.seed+i)).unsqueeze(0).to(device)
            with Image.open(args.manifest.parent/row['image_path']) as image:
                real=pil_to_tensor(image.convert('RGB')).unsqueeze(0).to(device)
            fid.update(real,real=True)
            fid.update(generated,real=False)
            inception.update(generated)
            clip.update(generated,[row['caption']])
        is_mean,is_std=inception.compute()
        result=dict(fid=float(fid.compute()),inception_score_mean=float(is_mean),
                    inception_score_std=float(is_std),clip_score=float(clip.compute()),
                    count=len(rows),split=args.split,seed=args.seed,checkpoint_step=predictor.steps,
                    checkpoint_sha256=sha256(args.checkpoint),manifest_sha256=sha256(args.manifest),
                    resolution=64,device=str(device),
                    fid_feature_dim=2048,low_sample_fid=len(rows)<2049,
                    metric_models='torch-fidelity InceptionV3; openai/clip-vit-base-patch32; evaluation only')
    write_json(args.output,result)
    print(result)


if __name__=='__main__':
    main()
