import csv
import json
from pathlib import Path
import subprocess
import sys
import torch
from doggen.curate import HashTree, phash, caption
from doggen.text import tokenize
from doggen.model import TextToImage, Discriminator, parameter_counts
from doggen.common import seed_all
from PIL import Image


def test_hash_tree_matches_bruteforce():
    rng=__import__('random').Random(65)
    tree=HashTree()
    values=[rng.getrandbits(63) for _ in range(100)]
    for i,v in enumerate(values):
        tree.add(v,str(i))
    for v in values[:20]:
        query=v ^ 0b111
        found=tree.match(query,4)
        assert found is not None
        assert (query ^ values[int(found)]).bit_count() <= 4
    for _ in range(50):
        q=rng.getrandbits(63)
        assert (tree.match(q,4) is not None)==any((q^v).bit_count()<=4 for v in values)


def test_trainable_conditioning_and_parameter_budget():
    torch.set_num_threads(2)
    seed_all(650)
    model=TextToImage(base=8)
    disc=Discriminator(breeds=2,base=8)
    model.eval()
    tokens=tokenize(['a black poodle dog','a white samoyed dog'])
    z=torch.randn(1,96).repeat(2,1)
    images=model(tokens,z)
    assert images.shape==(2,3,64,64)
    assert not torch.allclose(images[0],images[1])
    score,classes,_=disc(images,model.text_encoder(tokens))
    loss=score.mean()+torch.nn.functional.cross_entropy(classes,torch.tensor([0,1]))
    loss.backward()
    assert model.text_encoder.embedding.weight.grad.abs().sum()>0
    assert model.generator.fc.weight.grad.abs().sum()>0
    counts=parameter_counts(TextToImage(),Discriminator())
    assert counts['deployed_total']==1847523
    assert counts['training_total']<50_000_000


def test_arbitrary_utf8_and_padding():
    tokens=tokenize(['','a dog','犬'*200])
    assert tokens.shape==(3,192)
    assert tokens.min()>=0 and tokens.max()<258
    assert tokens[0,0]==257
    assert tokens[2,-1]==257


def test_checkpoint_inference_cli(tmp_path):
    seed_all(1)
    model=TextToImage(base=8)
    # This is explicitly a fixture, never a deliverable checkpoint.
    checkpoint=tmp_path/'fixture.pt'
    torch.save(dict(format_version=2,step=1,base=8,ema=model.state_dict()),checkpoint)
    prompts=tmp_path/'prompts.csv'
    prompts.write_text('id,prompt\n5,"a brown dog, outside"\n9,a white dog\n',encoding='utf-8')
    out=tmp_path/'output'
    run=subprocess.run([sys.executable,'test.py','--test-prompts',str(prompts),'--output-dir',str(out),
                        '--checkpoint',str(checkpoint),'--device','cpu'],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert sorted(p.name for p in out.iterdir())==['image_5.png','image_9.png']
    assert Image.open(out/'image_5.png').size==(64,64)
    prompts.write_text('id,prompt\n../escape,a dog\n')
    run=subprocess.run([sys.executable,'test.py','--test-prompts',str(prompts),'--output-dir',str(out)],capture_output=True)
    assert run.returncode!=0


def test_curation_duplicate_filter_and_restart(tmp_path, monkeypatch):
    import io
    import tarfile
    import hashlib
    import numpy as np
    import doggen.curate as curation
    archive=tmp_path/'images.tar.gz'
    entries=[]
    rng=np.random.default_rng(42)
    for i in range(12):
        image=Image.fromarray(rng.integers(0,256,(80,90,3),dtype=np.uint8))
        stream=io.BytesIO()
        image.save(stream,format='PNG')
        entries.append((f'images/dog{i}.png',stream.getvalue()))
    entries.append(('images/duplicate.png',entries[0][1]))
    with tarfile.open(archive,'w:gz') as tar:
        for name,content in entries:
            member=tarfile.TarInfo(name)
            member.size=len(content)
            tar.addfile(member,io.BytesIO(content))
    meta=tmp_path/'metadata.csv'
    with meta.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['name','annotations','target'])
        for name,_ in entries:
            writer.writerow([Path(name).name,'[[5, 10, 70, 75]]','Poodle'])
    monkeypatch.setattr(curation,'METADATA_SHA',hashlib.sha256(meta.read_bytes()).hexdigest())
    output=tmp_path/'curated'
    curation.curate(archive,output,minimum=10)
    original=(output/'manifest.jsonl').read_bytes()
    rows=[json.loads(line) for line in original.splitlines()]
    assert len(rows)==12
    assert {r['split'] for r in rows}=={'train','validation','test'}
    (output/rows[0]['image_path']).write_bytes(b'broken')
    curation.curate(archive,output,minimum=10)
    assert (output/'manifest.jsonl').read_bytes()==original
    assert Image.open(output/rows[0]['image_path']).size==(64,64)
    rejected=[json.loads(line) for line in (output/'rejections.jsonl').read_text().splitlines()]
    assert len(rejected)==1 and 'duplicate' in rejected[0]['reason']
