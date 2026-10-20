# proprioception

This repo replicates the training and evaluation for the preprint "A Sense of Prioprioception in Language Models",
in which open-weight LLMs are trained through multiple stages to endow them with a sense of (digital) proprioception.
You can also find the paper at http://arxiv.org/abs/2610.17495.

Feel free to reach out in case of any questions, or raise an issue.

## Install

```bash
pip install "uv>=0.12"
uv venv --python 3.12.11
uv pip install --python .venv/bin/python --prerelease=allow -e ".[eval]" # use -c constraints.txt if you really want the exact env
source .venv/bin/activate
python -m nltk.downloader punkt_tab
```

```bash
mkdir -p external && cd external
git clone https://github.com/PrimeIntellect-ai/prime-rl.git && cd prime-rl
git checkout 961c11f4cf8ebb6e23b074d34b81df80edbcfa9f
git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false submodule update --init
git apply ../../prime-rl.patch
uv sync --extra flash-attn
uv pip install --no-deps --python .venv/bin/python -e ../..
uv pip install --no-config --python .venv/bin/python --index-url https://flashinfer.ai/whl/cu128/ "flashinfer-jit-cache==0.6.12+cu128"
cd ../..
```

## Environment

```bash
source configs/env/local.sh
```

## Build Prompt bank and Dataset

```bash
python scripts/build_prompt_bank.py $PROPRIOCEPTION_DATA/prompt_bank.json --num-prompts 100000 --seed 0
proprio @ configs/build.toml --model lfm2_prime --num_train 240000 --data_dir $PROPRIOCEPTION_DATA/lfm_routing # repeat this per model
```

## SFT Stage

```bash
external/prime-rl/.venv/bin/torchrun --nproc_per_node=2 -m proprioception.cli @ configs/sft.toml --data_dir $PROPRIOCEPTION_DATA/lfm_routing --output $PROPRIOCEPTION_OUTPUTS/sft-lfm
```

## DAgger

```bash
external/prime-rl/.venv/bin/proprio launch @ configs/on_policy_sft.toml --server configs/inference.toml --output $PROPRIOCEPTION_OUTPUTS/dagger-lfm
```

## RL

```bash
cd external/prime-rl && VIRTUAL_ENV= uv run --no-sync rl @ ../../configs/rl.toml --output-dir $PROPRIOCEPTION_OUTPUTS/rl-lfm
```


## Quick Capabilities Eval

```bash
proprio @ configs/capabilities.toml --checkpoint $PROPRIOCEPTION_OUTPUTS/sft-lfm/hf --output $PROPRIOCEPTION_OUTPUTS/capabilities-sft-lfm
```

## Full Eval Suite

```bash
evals/runners/sad/setup.sh
evals/runners/ciaware/setup.sh
python -m evals prefetch
printf 'OPENROUTER_KEY=...\n' > .env
python -m evals run @evals/configs/lfm_routing.toml @evals/configs/profiles/full.toml --checkpoint $PROPRIOCEPTION_OUTPUTS/sft-lfm/hf
python -m evals report $PROPRIOCEPTION_OUTPUTS/evals/lfm_routing
```

## Citation

```
@article{geiping_sense_2026,
  title = {A {{Sense}} of {{Proprioception}} in {{Language Models}}},
  author = {Geiping, Jonas},
  year = 2026,
  month = oct,
  eprint = {2610.17495},
  primaryclass = {cs.LG},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2610.17495},
  url = {http://arxiv.org/abs/2610.17495},
  urldate = {2026-10-14},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Computation and Language,Computer Science - Machine Learning},
  journal = {arxiv:2610.17495[cs.LG]}
}
```

## Paper Abstract

In humans, proprioception corresponds to the sense of self-movement or position, e.g. understanding the position of one's arm. This sense is often less well-understood and in popular understanding sometimes conflated with introspection. But, notably, proprioception is a separate sense, with dedicated receptors and verifiable outcomes. On the other hand though, language models cannot observe their own internal states during training (and for example answer the question which experts activated in a certain layer of an MoE) and so cannot accurately report on them. Yet, such information is, by virtue of being internal to the model, both accessible, and readily constructed into verifiable signals. We argue that this provides the basis for a digital analog of proprioception. As such, we train models through SFT and on-policy training to equip them with this ability. We show that modern models can be trained to understand a range of subtypes of proprioception, such as routing decision awareness, token entropy, decision depth or logit lens information, without affecting model capabilities. We then run these models through a broad evaluation of situational awareness, legibility, and epiphenomenality, to evaluate in what ways access to proprioception changes model behavior.
