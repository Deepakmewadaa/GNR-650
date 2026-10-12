import torch.nn as nn

from models.flops import count_macs, count_params
from models.resnet import ResNetEncoder


def test_hand_computed_conv_linear():
    # conv: 3->4, 3x3, stride 1, pad 1 on 8x8 -> out 4x8x8 = 256 outputs, each 3*3*3 = 27 MACs -> 6912
    # linear: 256 -> 10 -> 2560 MACs.  total 9472 MACs, 18944 FLOPs
    net = nn.Sequential(nn.Conv2d(3, 4, 3, 1, 1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 10))
    r = count_macs(net, (1, 3, 8, 8))
    assert r["macs_conv"] == 6912
    assert r["macs_linear"] == 2560
    assert r["macs"] == 9472 and r["flops"] == 2 * 9472
    assert r["other_detail"]["act"] == 256


def test_strided_grouped_conv():
    # depthwise 4->4 groups=4, 3x3 stride 2 on 8x8 -> 4x4x4 = 64 outputs * (1*9) = 576
    net = nn.Sequential(nn.Conv2d(4, 4, 3, 2, 1, groups=4))
    assert count_macs(net, (1, 4, 8, 8))["macs"] == 576


def test_params_and_mode_restored():
    enc = ResNetEncoder(3, 8, 32, stem="small").eval()
    head = nn.Linear(32, 5)
    seq = nn.Sequential(enc, head)          # fresh Sequential defaults to train mode
    count_macs(seq, (1, 3, 16, 16))
    assert enc.training is False            # regression: counting must not flip BN into train mode
    assert count_params(head) == 32 * 5 + 5
