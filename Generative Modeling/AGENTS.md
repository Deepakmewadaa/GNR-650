# Project instructions

Use Python 3.11+ inside a freshly created platform-local `.venv`.
Read README.md and docs/STATUS.md before changing this project.
Keep the assignment's two mandatory components central. Preserve the Linux
`bash setup.bash` / `python test.py --test-prompts ... --output-dir ...` contract.
Never initialize generator, discriminator or text encoder with pretrained weights.
Pretrained metric/caption models belong only to isolated evaluation/curation tools.
Include text and all learned conditioning in parameter reports; enforce <50M.
Keep batch-one latency measurements synchronized, with the scope explicit.
Do not invent dataset counts, convergence, quality scores or target-GPU results.
Data paths in manifests are relative; avoid OS-specific absolute paths in code.
Rerun `python -m pytest -q` after model or inference changes. After data changes,
rerun curation and `python -m doggen.verify`, preserving >=10k accepted images.
Do not commit `.venv`, raw archives or regenerated temporary output.
