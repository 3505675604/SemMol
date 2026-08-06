# Semantic-level multimodal molecular learning inspired by biological concept formation via soft matching

Fig. 1 | From instance-level to semantic-level alignment in multimodal molecular representation learning. 
a, Instance-level (left) and semantic-level (right) alignment paradigm. b, An example of human perception, where the simultaneous presence of bread, meat, and vegetables naturally leads to the recognition of a hamburger (left); an illustration of the proposed semantic-level soft matching mechanism for molecular understanding (right). c, Classification performance comparison across multiple benchmark datasets for representative molecular representation models.
![WPS图片(1)](https://github.com/user-attachments/assets/eec1ebf2-230a-4315-9b4d-dcf797eb9de6)
Fig. 2 | Conceptual framework of the semantic-level multimodal molecular representation learning of SemMol. 
a, DCL construction. Initial semantic centers for 2D and 3D modalities are constructed from 1D, 2D, and 3D representations via intra-batch clustering and refined using mini-batch K-means with Exponential Moving Average (EMA) updates in the DCL. b, ACSM mechanism. b1, Positive sample generation: retrieve the nearest semantic centers from 2D/3D center libraries relative to the anchor and construct positive samples using similarity-weighted fusion. b2, Negative sample generation and debiasing: filter out easily distinguishable negatives using a similarity threshold while retaining hard negatives to improve discriminative ability. c, Semantic alignment training by using the constructed positive and negative samples.
<img width="1493" height="1310" alt="图片1" src="https://github.com/user-attachments/assets/1152bc6a-7c20-4239-8d26-ef0139749594" />


##
The most important supplementary file is provided at the following link: https://github.com/3505675604/SemMol/blob/main/Supplementary_Materials/Supplementary%20Materials.pdf

## 🚀 Features
- Semantic-level learning: By introducing the anchor-center soft matching (ACSM) mechanism and the dynamic center library (DCL), cross-modal semantic alignment is achieved, avoiding the risk of overfitting caused by one-to-one instance matching. This method can simultaneously capture the global skeleton structure and local functional group characteristics of the molecule, improving the model's generalization ability and interpretability.
- **Multi-Modal Learning**: Integrates chemical structure, text, and other modalities for improved prediction.
- **Flexible Task Support**: Handles both classification and regression tasks with dynamic configuration.
- **Advanced Pseudo-Pair Generation**: Supports hard negative mining, adaptive temperature, and memory bank for contrastive learning.
- **Streaming & Incremental Clustering**: Online center library with streaming K-means and FAISS acceleration.
- **Configurable & Reproducible**: All settings managed via a single JSON config; supports experiment reproducibility.
- **Extensible Architecture**: Modular codebase for easy extension of models, data pipelines, and loss functions.
- **Pretrained Model Integration**: Easy download and usage of state-of-the-art pretrained models.

---
## Environment Setup

```bash
conda env create -f environment.yml
conda activate semmol
pip install -e .
```
```bash
pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### PCQM4Mv2: Pretraining Data

After preparing the OGB PCQM4Mv2 dataset, execute the following steps in order. Default paths are `data/raw/PCQM4Mv2` and `data/processed/pcqm`.

1. Export a source manifest containing `source_index`, `official_split`, SMILES, and gap:

```bash
python scripts/preprocess/prepare_pcqm_manifest.py \
  --dataset-root data/raw/PCQM4Mv2 \
  --output data/processed/pcqm/source_manifest.parquet

cd ~/autodl-tmp/SemMol/data/raw/PCQM4Mv2

wget -c \
  http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz

echo "fd72bce606e7ddf36c2a832badeec6ab  pcqm4m-v2-train.sdf.tar.gz" \
  | md5sum -c -
tar -xf pcqm4m-v2-train.sdf.tar.gz
test -s pcqm4m-v2-train.sdf &&
  echo "PCQM4Mv2 official 3D SDF: OK"
```

2. Filter 1M/3M selections:

```bash
python scripts/preprocess/filter_pcqm.py \
  --source-manifest data/processed/pcqm/source_manifest.parquet \
  --output-dir data/processed/pcqm/manifests \
  --target-sizes 1000000 3000000 \
  --smiles-col smiles --gap-col homolumogap \
  --source-index-col source_index --official-split-col official_split \
  --selection-mode approximate
```

3. Train ESPF vocabulary from the selection:

```bash
python scripts/preprocess/build_espf_vocab.py \
  --input <selection_manifest.parquet> \
  --output-dir data/processed/pcqm/tokenizer \
  --smiles-col canonical_smiles
```

4. Generate 3D conformer shards:

```bash
python scripts/preprocess/generate_3d_conformer.py \
  --input data/processed/pcqm/source_manifest.parquet \
  --manifest <selection_manifest.parquet> \
  --source-index-col source_index \
  --output-dir data/processed/pcqm/geometry
```

Without `--official-sdf`, this command uses the deterministic ETKDGv2/MMFF path. The source manifest already exports `train_ordinal`; if you have the SDF locally, replace the above with:

```bash
python scripts/preprocess/generate_3d_conformer.py \
  --input data/processed/pcqm/source_manifest.parquet \
  --manifest <selection_manifest.parquet> \
  --source-index-col source_index --sdf-ordinal-col train_ordinal \
  --official-sdf data/raw/PCQM4Mv2/pcqm4m-v2-train.sdf \
  --output-dir data/processed/pcqm/geometry
```

5. Build the pretraining store:

```bash
python scripts/preprocess/build_pcqm_store.py \
  --selection-manifest data/processed/pcqm/manifests/pcqm_selection_CURRENT.json \
  --geometry-dir data/processed/pcqm/geometry \
  --tokenizer-dir data/processed/pcqm/tokenizer \
  --output-dir data/processed/pcqm/store \
  --target-sizes 1000000,3000000
```

6. Validate the store:

```bash
python scripts/preprocess/validate_processed_store.py \
  --store-dir data/processed/pcqm/store --full
```

`generate_qm_density.py` is an optional step for independent NPZ generation. It is not part of the formal pretraining objective, nor is it a required input consumer for the final store. However, the current PCQM and MoleculeNet store builders still unconditionally compute and store promolecular density, which adds to preprocessing compute and storage overhead.

### MoleculeNet: Downstream Data

Place the raw CSV files for the nine registered datasets under `<moleculenet_raw_root>`. The filenames and required label columns are determined by the dataset registry; the repository cannot uniquely infer your raw data layout from the README. Then build splits, then the store, then validate:

```bash
python scripts/preprocess/build_scaffold_splits.py \
  --raw-root <moleculenet_raw_root> \
  --output-dir data/splits \
  --datasets bace bbbp clintox tox21 toxcast sider freesolv esol lipophilicity \
  --frac 0.8 0.1 0.1

python scripts/preprocess/build_moleculenet_store.py \
  --raw-root <moleculenet_raw_root> \
  --split-root data/splits \
  --tokenizer-dir data/processed/pcqm/tokenizer \
  --output-root data/processed/moleculenet \
  --datasets bace bbbp clintox tox21 toxcast sider freesolv esol lipophilicity

for dataset in bace bbbp clintox tox21 toxcast sider freesolv esol lipophilicity; do
  python scripts/preprocess/validate_processed_store.py \
    --store-dir "data/processed/moleculenet/${dataset}" --full
done
```

## Training Configuration & Launch
Single-node 8-GPU (the script defaults to `NNODES=1` and derives 8 processes per node from the YAML `world_size=8`):

```bash
bash scripts/pretrain/run_pretrain.sh configs/pretrain/pretrain_1m.yaml
bash scripts/pretrain/run_pretrain.sh configs/pretrain/pretrain_3m.yaml
```

For multi-node training, set the same `NNODES`, `MASTER_ADDR`, `MASTER_PORT`, and `NPROC_PER_NODE` on every machine, and set a different `NODE_RANK` for each. For example, two nodes with four GPUs each:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<master_ip> MASTER_PORT=29500 NPROC_PER_NODE=4 \
  bash scripts/pretrain/run_pretrain.sh configs/pretrain/pretrain_3m.yaml

NNODES=2 NODE_RANK=1 MASTER_ADDR=<master_ip> MASTER_PORT=29500 NPROC_PER_NODE=4 \
  bash scripts/pretrain/run_pretrain.sh configs/pretrain/pretrain_3m.yaml
```

When `NPROC_PER_NODE` is not set, both scripts derive it as `world_size / NNODES`. For multi-node setups, `MASTER_ADDR` and `NODE_RANK` must be explicitly set. The finetuning scripts follow the same relationship, but the current nine YAMLs have `world_size` set to 1. `--device` is only recommended for single-process debugging; do not pass a fixed `cuda` or `cuda:0` when using torchrun with multiple GPUs — omit it to let `LOCAL_RANK` map automatically. The pretraining script also supports `--resume CHECKPOINT`.

## Ten-Seed Finetuning

Each finetuning YAML mandates exactly ten different seeds. The entry point runs them sequentially and writes each seed's checkpoint to:
```text
checkpoints/finetune/<experiment>/seed_<seed>/{best.pt,latest.pt}
```

Aggregated results are written to `outputs/logs/<experiment>/ten_seed_results.json`. For example:
```bash
bash scripts/finetune/run_finetune.sh configs/finetune/bace.yaml
```

## Hyperparameter Grid Search

The entry point is `scripts/hyperparam/run_grid_search.py`, which supports both grid and random search strategies across pretraining and finetuning stages. 

### Predefined Grid Definitions (`configs/hyperparam/`)

| File | Search Space | Strategy |
|---|---|---|
| `dcl_sensitivity.yaml` | DCL cluster count K, EMA momentum β | grid |
| `acsm_sensitivity.yaml` | ACSM retrieval count M, temperature τ, denoise threshold δ | grid |
| `training_sensitivity.yaml` | lr, weight_decay, loss weights | grid |
| `mask_encoder_sensitivity.yaml` | Per-modality mask ratios, geo noise, dropout | grid |
| `finetune_sensitivity.yaml` | Finetuning lr, weight_decay, batch_size, warmup | grid |


Core fields of a grid definition: `axes` (`path` is a dotted YAML key, `values` is the list of candidate values), `constraints` (expressions like `"M < K"` to filter invalid combinations), and `evaluation` (`mode`, `fast_epochs`, `metrics`, `direction`).

### Running

```bash
python scripts/hyperparam/run_grid_search.py \
  --base-config configs/pretrain/pretrain_3m.yaml \
  --grid configs/hyperparam/dcl_sensitivity.yaml \
  --output-dir outputs/hyperparam/dcl_sensitivity \
  --mode pretrain \
  --epochs 10 \
  --device cuda:0
```

`--max-trials` overrides the grid definition's upper limit. `--dry-run` generates configs without executing. `--timeout-per-trial` sets the per-trial timeout (default: 24 h).

### Outputs

Each trial generates an isolated directory under `outputs/hyperparam/<name>/trials/trial_NNNN/` containing the synthesized `config.yaml` and `trial_output.log`. Aggregated results:

| File | Contents |
|---|---|
| `results.json` | Status, metrics, and wall time for all trials |
| `report.md` | Ranked table + sensitivity scores + best config |
| `sensitivity.csv` | mean/std/sensitivity_score per parameter value |
| `best_config.yaml` | YAML snippet of the best trial, diffable against the base config |


📈 Reproducibility
-Fixed random seeds for all experiments
-Centralized configuration management
-Modular and extensible codebase for easy customization

📚 Citation
If you use SemMol in your research, please cite the original authors:
```bash
@article{semmol,
  title={SemMol:Semantic-level multimodal molecular learning inspired by biological concept formation via soft matching},
  author={Anonymous},
  journal={arXiv preprint},
  year={202X}
}
```

##License
'''
This project is licensed under the MIT License - see the LICENSE file for details.
'''
