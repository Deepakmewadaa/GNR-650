import argparse
import json
from .model import TextToImage, Discriminator, parameter_counts
from .common import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=int,default=32)
    p.add_argument('--output',default='reports/parameters.json')
    a=p.parse_args()
    counts=parameter_counts(TextToImage(a.base),Discriminator(base=a.base))
    write_json(a.output,counts)
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    main()
