<h1 align="center">
  Time-to-Goal: Learned Temporal Metrics for<br>
  Long-Horizon Planning with JEPA World Models
</h1>

<div align="center">
  <a href="https://arxiv.org/abs/2611.05847"><img alt="arXiv:2611.05847" src="https://img.shields.io/badge/arXiv-2611.05847-b31b1b?logo=arxiv"></a>
  <a href="2611.05847v1.pdf"><img alt="Local paper" src="https://img.shields.io/badge/Paper-PDF-blue"></a>
</div>

<p align="center">
  <b>Ruilong Yu · Fei Ye</b><br>
  University of Electronic Science and Technology of China
</p>

---

**Time-to-Goal (T2G)** learns a temporal distance on frozen JEPA representations and uses it as the goal cost for latent-space planning. A margin ranking loss teaches a small MLP to assign smaller distances to temporally nearer states, using offline trajectories without rewards or policy labels. The world-model encoder and predictor remain frozen during metric training.

This repository provides training and planning evaluation for **Visual AntMaze, FourRoom Distractors, OGBench Cube, and Push-T**, built on [H-JEPA](https://github.com/kevinghst/H-JEPA), [LeWorldModel](https://github.com/lucas-maes/le-wm), and [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

In the [paper](https://arxiv.org/abs/2611.05847), last-n T2G improves the unweighted average planning success across the four environments from **32.7% to 63.0%**, while retaining a flat world model.

## 1) Installation

The commands below target a Linux/CUDA environment. Python 3.11 and the pinned environment versions in `pyproject.toml` are used by the code.

```bash
git clone https://github.com/yrluestc/T2G-JEPA.git
cd T2G-JEPA
conda create -n t2g python=3.11.13
conda activate t2g
pip install -e ".[train,env]" --extra-index-url https://download.pytorch.org/whl/cu128
pip install huggingface_hub

export HJEPA_HOME=/absolute/path/to/t2g-data
mkdir -p "$HJEPA_HOME"
```

`HJEPA_HOME` stores datasets, training checkpoints, normalization statistics, and evaluation outputs. The package retains the upstream name `h-jepa` and Python module `stable_worldmodel`.

**Working directories:** data preparation commands run from the repository root. Training, planning, and diagnostic commands run from **`hjepa/`**. Planning evaluation currently requires CUDA. Select a GPU with `CUDA_VISIBLE_DEVICES=0`; metric training also accepts `device=cpu` or `device=cuda:0`.

## 2) Quick Start

First prepare the datasets and fixed evaluation tasks in [Section 3](#3-datasets-and-evaluation-tasks). From the repository root:

```bash
cd hjepa

# 1. Train a flat LeWM, or use an upstream checkpoint as described in Section 3.3.
# These overrides disable the optional end-of-training planning/probing callbacks.
python main_hjepa.py --config-name ant_lewm seed=42 \
  planning_eval.enabled=false final_probing_decoding_eval.enabled=false

# 2. Train T2G on the frozen LeWM representations.
python train_temporal_metric.py --config-name metric_ant seed=42

# 3. Plan using last-2 T2G, as in the paper.
RUN="$HJEPA_HOME/ckpts/ant/ant_lewm/seed42"
python eval.py --config-name ant_flat_metric seed=42 \
  policy="$RUN/ant_lewm_object.ckpt" \
  metric_ckpt="$RUN/temporal_metric.pt" \
  cost=metric_dense ++metric_last_n=2 output.dir=eval_t2g_lastn
```

Planning results are written beside the world-model checkpoint, here to `eval_t2g_lastn/metrics.yaml`. `success_rate` is a percentage. Use a new `output.dir` when changing costs, metric checkpoints, or planner settings so that saved results from different experiments are not mixed.

## 3) Datasets and Evaluation Tasks

### 3.1) Offline training datasets

Place the HDF5 datasets directly under `$HJEPA_HOME`:

| Environment | Training file | Validation file |
|---|---|---|
| Visual AntMaze | `visual_antmaze_medium_explore_stitch_train.h5` | `visual_antmaze_medium_stitch_val.h5` |
| FourRoom Distractors | `fourroom_tp35_d1.h5` | `fourroom_tp35_d1_val.h5` |
| OGBench Cube | `cube_single_expert_train.h5` | `cube_single_expert_val.h5` |
| Push-T | `pusht_expert_train.h5` | `pusht_expert_val.h5` |

**Push-T and Cube.** Use the upstream LeWM data releases. The commands require `zstd`/`unzstd` and `tar`:

```bash
# Run from the repository root.
hf download quentinll/lewm-pusht --repo-type dataset --local-dir "$HJEPA_HOME" --include "*.zst"
hf download quentinll/lewm-cube --repo-type dataset --local-dir "$HJEPA_HOME" --include "*.zst"
hf download jepa-world-models/h-jepa pusht_expert_val.h5 --repo-type dataset --local-dir "$HJEPA_HOME"
unzstd "$HJEPA_HOME/pusht_expert_train.h5.zst"
tar -I unzstd -xf "$HJEPA_HOME/cube_single_expert.tar.zst" -C "$HJEPA_HOME"
python scripts/data/add_pusht_block_ori.py "$HJEPA_HOME/pusht_expert_train.h5"
python scripts/data/split_cube.py
```

The Cube split uses the first 9,800 episodes for training and the final 200 for validation. The split script creates new files and should be run once.

**AntMaze and FourRoom.** Download the H-JEPA datasets:

```bash
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir "$HJEPA_HOME" \
  --include "visual_antmaze_medium_*" --include "fourroom_tp35_d1*" --include SHA256SUMS
(cd "$HJEPA_HOME" && sha256sum -c --ignore-missing SHA256SUMS)
```

These patterns also fetch the probing files. Frozen-model probing is optional for T2G. If you keep the upstream end-of-training probing callback enabled, prepare the corresponding probing datasets as well. Dataset provenance and licensing are described in the upstream dataset cards.

### 3.2) Fixed planning tasks

Each environment uses 50 fixed start/goal tasks. From the repository root:

```bash
cd hjepa
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir assets --include "eval_trajs/*"
(cd assets/eval_trajs && sha256sum -c SHA256SUMS)
```

The evaluation configs expect:

```text
hjepa/assets/eval_trajs/
├── ant/expert_grid_d3_n50.pt
├── fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt
├── ogbench/goal_offset_20_pickup_val.pt
└── pusht/goal_offset_75_val.pt
```

AntMaze tasks connect cells at grid distance three; FourRoom tasks cross two rooms; Cube uses 20-step grasp-centered windows; Push-T uses 75-step held-out frame pairs. All compared methods use the same task files.

### 3.3) Pretrained flat world models

To skip LeWM training, download the upstream flat checkpoints and their `normalizer.pt` files:

```bash
# Run from the repository root.
hf download jepa-world-models/h-jepa --local-dir "$HJEPA_HOME/ckpts" \
  --include "ant/ant_lewm/*" --include "fourroom/fourroom_lewm/*" \
  --include "cube/cube_lewm/*" --include "pusht/pusht_lewm/*"
```

The expected layout is `$HJEPA_HOME/ckpts/<env>/<env>_lewm/seed<seed>/`. The upstream release supplies **world models**, not the learned T2G weights; train the temporal metric in the next section. With fixed task files and the saved normalizer, planning evaluation does not need to read the training HDF5 data. T2G training does.

## 4) Train the Temporal Metric

All commands in this and the following sections run from `hjepa/`.

### 4.1) Train or reuse flat LeWM

```bash
for env in ant fourroom cube pusht; do
  for seed in 42 43 44; do
    python main_hjepa.py --config-name "${env}_lewm" seed="$seed" \
      planning_eval.enabled=false final_probing_decoding_eval.enabled=false
  done
done
```

Skip this loop if you downloaded the upstream checkpoints. A run stores `<env>_lewm_object.ckpt`, training configuration, and normalization statistics under its checkpoint directory.

### 4.2) Train T2G

```bash
for env in ant fourroom cube pusht; do
  for seed in 42 43 44; do
    python train_temporal_metric.py --config-name "metric_${env}" seed="$seed"
  done
done
```

The default `policy` points to the flat LeWM checkpoint with the same environment and seed. Override `policy=/absolute/path/to/model_object.ckpt` to use another checkpoint. T2G outputs are saved beside that checkpoint.

| Environment | Training config | Latent dimension | Clip length | Maximum offset |
|---|---|---:|---:|---:|
| Visual AntMaze | `metric_ant` | 384 | 24 | 23 |
| FourRoom | `metric_fourroom` | 192 | 24 | 23 |
| Cube | `metric_cube` | 192 | 16 | 15 |
| Push-T | `metric_pusht` | 192 | 16 | 15 |

Offsets are in model steps; one model step covers five environment steps. The maximum training offset is distinct from the planning horizon.

The simulation defaults match the paper's training recipe: **5 epochs**, batch size **128**, AdamW learning rate **5e-5**, weight decay **1e-3**, margin **1**, and gradient clipping at **1.0**. The MLP uses two hidden blocks of width 256 with LayerNorm/GELU, followed by a 256-dimensional output layer and a learned positive distance scale. The encoder and predictor stay frozen.

Each clip supplies one triple `(z_t, z_{t+k}, z_{t+k+delta})`. The simulation sampler uses minimum near offset 1 and minimum near-far gap 2. It samples the near offset and an additional gap, clips the far offset to the maximum, then samples a valid anchor, as described in the paper's appendix. Additional negative-sample loss is disabled by default.

Outputs include:

```text
seed42/
├── ant_lewm_object.ckpt
├── normalizer.pt
├── temporal_metric.pt
├── temporal_metric_state.pt
├── temporal_metric_epoch_01.pt
├── ...
└── temporal_metric_epoch_05.pt
```

The metric is saved every epoch by default. The state file includes optimizer state; the entry point does not automatically resume metric training. A fresh run overwrites the metric output names. For a separate experiment, choose another name with `metric.output_name=temporal_metric_experiment.pt`.

T2G learns **relative temporal ordering**; its distance is not calibrated in remaining steps.

## 5) Planning Evaluation

### 5.1) Cost variants and configuration defaults

For predicted states `pred_t` and a fixed encoded goal, the available flat costs are:

| Cost | Configuration | Scoring |
|---|---|---|
| Euclidean baseline | `<env>_flat` | Sum of squared latent distances over the last `plan_config.cost_last_n` states |
| Dense T2G | `cost=metric_dense`, `metric_last_n=null` | Mean learned distance over all predicted future states |
| Last-n T2G | `cost=metric_dense`, `metric_last_n=n` | Mean learned distance over the last n predicted states |
| Terminal T2G | `cost=metric` | Learned distance at the final predicted state |

The effective last-n window is capped by the current rollout length. Terminal T2G and last-1 T2G give the same score. Use Hydra's `++metric_last_n=...` syntax to add or override the key consistently across all environment configs.

**The shipped defaults are mixed:** `ant_flat_metric`, `fourroom_flat_metric`, and `pusht_flat_metric` use dense T2G; `cube_flat_metric` uses terminal T2G. The explicit overrides below select the paper variants consistently.

| Environment | Episode budget | Initial T2G horizon | Paper last-n | Candidates | Planner LR |
|---|---:|---:|---:|---:|---:|
| AntMaze | 350 | 23 | 2 | 100 | 0.3 |
| FourRoom | 210 | 23 | 2 | 200 | 0.1 |
| Cube | 30 | 2 | 1 | 800 | 0.05 |
| Push-T | 90 | 15 | 2 | 800 | 0.05 |

Episode budgets count environment steps; horizons count model predictions. Cube T2G uses a constant horizon of 2, while the native Cube baseline starts at 4 with a decreasing schedule. The other environments retain the flat baseline's planning settings. Thus the Cube comparison includes a horizon change as well as a cost change.

### 5.2) Evaluate baseline, dense, and last-n T2G

After preparing tasks, world-model checkpoints, metrics, and normalizers:

```bash
for env in ant fourroom cube pusht; do
  n=2
  [[ "$env" == cube ]] && n=1
  for seed in 42 43 44; do
    RUN="$HJEPA_HOME/ckpts/$env/${env}_lewm/seed$seed"
    POLICY="$RUN/${env}_lewm_object.ckpt"

    python eval.py --config-name "${env}_flat" seed="$seed" \
      policy="$POLICY" output.dir=eval_flat

    python eval.py --config-name "${env}_flat_metric" seed="$seed" \
      policy="$POLICY" metric_ckpt="$RUN/temporal_metric.pt" \
      cost=metric_dense ++metric_last_n=null output.dir=eval_t2g_dense

    python eval.py --config-name "${env}_flat_metric" seed="$seed" \
      policy="$POLICY" metric_ckpt="$RUN/temporal_metric.pt" \
      cost=metric_dense ++metric_last_n="$n" output.dir=eval_t2g_lastn
  done
done
```

Choose an earlier metric with `metric_ckpt=/path/to/temporal_metric_epoch_03.pt`. Explicit metric paths are useful when the planner seed differs from the world-model training seed: the shipped simulation paths otherwise interpolate `seed` into the metric checkpoint directory.

Results are stored in the corresponding `eval_*/metrics.yaml` folders beside each checkpoint. Report mean and standard error over seeds 42/43/44, keeping task files and planner budgets fixed within each comparison.

### 5.3) Single-dataset, single-seed example (FourRoom, seed 42)

From `hjepa/`, train T2G for FourRoom with the default 5 epochs:

```bash
python train_temporal_metric.py --config-name metric_fourroom seed=42
```

Then evaluate the paper's last-2 T2G with planning seed 42:

```bash
RUN="$HJEPA_HOME/ckpts/fourroom/fourroom_lewm/seed42"

python eval.py --config-name fourroom_flat_metric seed=42 \
  policy="$RUN/fourroom_lewm_object.ckpt" \
  metric_ckpt="$RUN/temporal_metric.pt" \
  cost=metric_dense ++metric_last_n=2 \
  output.dir=eval_t2g_last2_seed42
```

### 5.4) Results reported in the paper

Success rate (%), mean ± SE across three seeds, with 50 tasks per environment and seed:

| Method | AntMaze | Cube | FourRoom | Push-T |
|---|---:|---:|---:|---:|
| H-JEPA, 2 levels | 39.3 ± 3.7 | 47.3 ± 2.7 | 81.3 ± 6.7 | 45.3 ± 1.3 |
| H-JEPA, 3 levels | **73.3 ± 3.5** | 60.0 ± 5.0 | **96.0 ± 1.1** | 17.3 ± 3.5 |
| H-JEPA, 4 levels | 63.3 ± 7.7 | 55.3 ± 4.4 | 80.0 ± 4.0 | 0.7 ± 0.7 |
| Flat LeWM | 18.0 ± 3.5 | 32.0 ± 6.1 | 40.7 ± 6.8 | 40.0 ± 2.3 |
| Flat LeWM + T2G dense | 48.3 ± 2.4 | 4.0 ± 1.2 | 81.3 ± 2.9 | 42.0 ± 1.2 |
| Flat LeWM + T2G last-n | 58.7 ± 2.4 | **63.3 ± 0.7** | 78.7 ± 2.9 | **51.3 ± 5.9** |

These are the results in Table 2 of the accompanying paper, not measurements produced by running this README. Dense and last-n aggregation can behave differently when intermediate actions are needed before approaching the final goal.

## 6) Temporal-Cost Diagnostics

Compare Euclidean and T2G costs on the **same encoded execution states and future goals**:

```bash
RUN="$HJEPA_HOME/ckpts/ant/ant_lewm/seed42"
python scripts/diagnose_planning_geometry.py \
  --config-name ant_flat_metric --policy "$RUN/ant_lewm_object.ckpt" \
  --metric-ckpt "$RUN/temporal_metric.pt" --seed 42 \
  --override metric_last_n=2 --plan-max-tasks 50 \
  --pair-mode fixed --geometry-details --output-dir "$RUN/diagnostics_last2"
```

The script records executions, caches encoded states, and reports temporal distances, near-far ranking, and temporal contrast. It includes successful and failed executions. Use `--help` for lag, comparison-gap, sampling-stride, and plotting options. These diagnostics measure cost behavior on execution states; they do not replace planning success evaluation.

To regenerate reports from a branch's saved `executed_latents.npz` without repeating simulation:

```bash
python scripts/diagnose_planning_geometry.py \
  --replay /path/to/executed_latents.npz --output-dir /path/to/replayed_diagnostics
```

Replay can use cached T2G embeddings without Torch; older caches without those embeddings require a metric checkpoint and Torch.

## 7) Code Reference

| File / directory | Purpose |
|---|---|
| [hjepa/temporal_metric.py](hjepa/temporal_metric.py) | Learned distance, triple sampling, and ranking loss |
| [hjepa/train_temporal_metric.py](hjepa/train_temporal_metric.py) | Train a metric on frozen LeWM latents |
| [hjepa/config/train](hjepa/config/train) | Flat world-model and `metric_<env>` training configs |
| [hjepa/config/eval](hjepa/config/eval) | Flat baseline and T2G planning configs |
| [hjepa/models/jepa.py](hjepa/models/jepa.py) | Latent rollout and planning-cost implementations |
| [hjepa/planning_eval.py](hjepa/planning_eval.py) | Checkpoint loading, metric attachment, planning evaluation |
| [hjepa/scripts/diagnose_planning_geometry.py](hjepa/scripts/diagnose_planning_geometry.py) | Execution-based cost diagnostics and replay |
| [hjepa/cost_diagnostics.py](hjepa/cost_diagnostics.py) | Temporal-cost statistics and reports |
| [scripts/data](scripts/data) | Dataset preparation and collection |

## 8) Citation and Acknowledgments

If you use T2G in your research, please cite our paper:

```bibtex
@misc{yu2026timetogoallearnedtemporalmetrics,
      title={Time-to-Goal: Learned Temporal Metrics for Long-Horizon Planning with JEPA World Models}, 
      author={Ruilong Yu and Fei Ye},
      year={2026},
      eprint={2611.05847},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.05847}, 
}
```

We thank the authors of [H-JEPA](https://github.com/kevinghst/H-JEPA), [LeWorldModel](https://github.com/lucas-maes/le-wm), and [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for their implementations and released resources. Please also cite the upstream world models and benchmarks when using their code, checkpoints, or datasets, and retain their applicable licenses and copyright notices.
