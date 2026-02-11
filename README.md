# T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation

<a href='https://t2v-compbench-2025.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2407.14505'><img src='https://img.shields.io/badge/T2V--CompBench-Arxiv-red'></a> 
<a href='https://huggingface.co/spaces/Kaiyue/T2V-CompBench_Leaderboard'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue'></a> 
[![Dataset Download](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-red)](https://huggingface.co/datasets/Kaiyue/T2V-CompBench-Videos)
[![Video](https://img.shields.io/badge/T2V--CompBench-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=td0wWN-5PbY)


This repository is the official implementation of the following paper:
> **T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation**<br>
> [Kaiyue Sun](https://scholar.google.com/citations?user=mieuBzUAAAAJ&hl=en)<sup>1</sup>, [Kaiyi Huang](https://github.com/Karine-Huang)<sup>1</sup>, [Xian Liu](https://alvinliu0.github.io/)<sup>2</sup>, [Yue Wu](https://yuewuhkust.github.io/)<sup>3</sup>, Zihan Xu<sup>1</sup>, [Zhenguo Li](https://scholar.google.com/citations?hl=en&user=XboZC1AAAAAJ&view_op=list_works&sortby=pubdate)<sup>3</sup>, [Xihui Liu](https://xh-liu.github.io/)<sup>1</sup><br>
> ***<sup>1</sup>The University of Hong Kong, <sup>2</sup>The Chinese University of Hong Kong, <sup>3</sup>Huawei Noah’s Ark Lab***<br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025

### Table of Contents
- [Updates](#updates)
- [Overview](#overview)
- [Evaluation Results](#evaluation_results)
- [Leaderboard](#leaderboard)
- [Prompt Suite](#prompt_suite)
- [Quick Start](#quick_start)
- [Prepare Evaluation Videos](#prepare_videos)
- [Run Individual Categories](#individual_eval)
- [Score Aggregation](#score_aggregation)
- [Weight File Reference](#weight_reference)
- [Citation](#citation)

<a name="updates"></a>
## 🚩 Updates
- ✅ [10/2025] Release the generated videos for T2V-CompBench evaluation.
- :boom: [02/2025] Paper accepted to CVPR 2025.
- ✅ [01/2025] T2V-CompBench Leaderboard
- ✅ [01/2025] Release the evaluation scripts for the 7 categories.
- ✅ [01/2025] Release the prompt dataset and metadata.
  
<a name="overview"></a>
## :mega: Overview
![teaser](./asset/teaser.png)
We propose **T2V-CompBench**, the first benchmark tailored for **compositional text-to-video generation**. T2V-CompBench encompasses diverse aspects of compositionality, including **consistent attribute binding**, **dynamic attribute binding**, **spatial relationships**, **motion binding**, **action binding**, **object interactions**, and **generative numeracy**. We further carefully design evaluation metrics of **MLLM-based metrics**, **detection-based metrics**, and **tracking-based metrics**, which can better reflect the compositional text-to-video generation quality of seven proposed categories with 1400 text prompts. The effectiveness of the proposed metrics is verified by correlation with human evaluations. We also **benchmark various text-to-video generative models** and conduct in-depth analysis across different models and different compositional categories. We find that compositional text-to-video generation is highly challenging for current models, and we hope that our attempt will shed light on future research in this direction.

<a name="evaluation_results"></a>
## :mortar_board: Evaluation Results
We benchmark 17 publicly available text-to-video generation models and 6 commercial models including Kling, Gen-3, Gen-2, Pika, Dreamina and PixVerse. We normalize the results for clearer comparisons. 
Please see our leaderboard for the most updated ranking and numerical results. <a href='https://huggingface.co/spaces/Kaiyue/T2V-CompBench_Leaderboard'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue'></a> 
![ranking](./asset/ranking.png)

<a name="leaderboard"></a>
## :mortar_board: How to join T2V-CompBench Leaderboard
If you have already evaluated all or any categories of T2V-CompBench in your report/paper, submit your eval_results.zip to the [T2V-CompBench Leaderboard](https://huggingface.co/spaces/Kaiyue/T2V-CompBench_Leaderboard) using the Submit here! form. The evaluation results will be automatically updated to the leaderboard. Also, share your model information for our records for any field in the form.

### Instructions:
The `.zip` file requires at most eight csv files if you have evaluated all the seven categories, please follow the evaluation steps to generate each of them, they are:
```
mymodel_consistent_attr_score.csv,
mymodel_dynamic_attr_score.csv,
mymodel_spatial_score.csv,
mymodel_motion_score.csv,
mymodel_motion_back_fore.csv,
mymodel_action_binding_score.csv,
mymodel_object_interactions_score.csv,
mymodel_numeracy_video.csv,
```

1. All of the files listed above are final CSV files that record the model's score for their respective categories, except for "mymodel_motion_back_fore.csv," which contains the intermediate results for motion binding.
Please replace "mymodel" with your model name.
2. If your model is unable to generate one or more videos for certain categories due to safety reasons or other technical issues, the evaluation scripts will automatically skip these cases. As a result, they will not be recorded in the CSV file, and the final average score will exclude them.
3. The backend script of our leaderboard will also exclude those ungenerated videos if any of the submitted final CSV files contain fewer than 200 videos.
4. To successfully showcase your model's performance on our leaderboard, please ensure that the last line of each final CSV file, which records the video-level scores, includes the model's score for that category. This line must begin with "score: " or "Score: ".

Put the CSV files in a folder and compress it, then submit the `.zip` to [T2V-CompBench Leaderboard](https://huggingface.co/spaces/Kaiyue/T2V-CompBench_Leaderboard)

<a name="prompt_suite"></a>
## :blue_book: T2V-CompBench Prompt Suite
The T2V-CompBench prompt suite includes 1400 prompts covering 7 categories, each with 200 prompts.

For each category, the text prompts used to generate the videos are saved in a text file under the `playground/prompts/` directory.
The metadata used to assist the evaluation are saved in a json file under the `playground/meta_data/` directory.

<a name="quick_start"></a>
## :rocket: Quick Start

Three steps to evaluate a model:

### 1. Install

```bash
bash install.sh
```

This creates the `t2v` conda environment, installs dependencies (Qwen3-VL, GroundingSAM, DOT), and downloads required detection/tracking weights (GroundingDINO, SAM, DOT checkpoints). Qwen3-VL weights are downloaded automatically on first inference from HuggingFace.

### 2. Prepare Videos

Generate 200 videos per category using the prompts in `playground/prompts/`, then organize them as:

```
playground/model_output/{model_name}/
├── consistent_attr_1/    # 0001.mp4 ~ 0200.mp4
├── dynamic_attr_2/       # 0001.mp4 ~ 0200.mp4
├── spatial_3/            # 0001.mp4 ~ 0200.mp4
├── motion_4/             # 0001.mp4 ~ 0200.mp4
├── action_5/             # 0001.mp4 ~ 0200.mp4
├── interaction_6/        # 0001.mp4 ~ 0200.mp4
└── numeracy_7/           # 0001.mp4 ~ 0200.mp4
```

See [Prepare Evaluation Videos](#prepare_videos) for details.

### 3. Run Evaluation

```bash
bash T2V-CompBench/eval_all.sh <model_name>
```

This runs all 7 category evaluations and prints the final aggregated score. Results are written to `playground/results/csv_*/`.

<a name="prepare_videos"></a>
## :clapper: Prepare Evaluation Videos

Generate 200 videos per category using the prompts in `playground/prompts/`. Videos must be named `0001.mp4` through `0200.mp4`, where each number corresponds to the line number in the prompt file (and the index in the metadata JSON).

- **Format**: MP4 (or any format supported by OpenCV `cv2.VideoCapture`)
- **Naming**: `0001.mp4` ~ `0200.mp4` (4-digit zero-padded, 1-indexed)
- **Count**: 200 per category, 1400 total
- **Resolution/FPS**: No strict requirement; scripts extract frames automatically

The evaluation scripts will automatically:
- Convert videos to 3x2 image grids (6 frames) for MLLM-based evaluation
- Convert videos to 16 individual frames for detection/tracking-based evaluation
- Intermediate files are saved under the video directory (e.g., `frames/`, `image_grid/`)

<a name="individual_eval"></a>
## :wrench: Run Individual Categories

All scripts are run from the **project root** directory. Activate the environment first:

```bash
conda activate t2v
```

### MLLM-based Evaluation (Qwen3-VL, frame-based)

| Category | Command |
|----------|---------|
| Consistent Attribute | `python T2V-CompBench/compbench_eval_consistent_attr.py --video-path playground/model_output/mymodel/consistent_attr_1 --t2v-model mymodel` |
| Dynamic Attribute | `python T2V-CompBench/compbench_eval_dynamic_attr.py --video-path playground/model_output/mymodel/dynamic_attr_2 --t2v-model mymodel` |
| Action Binding | `python T2V-CompBench/compbench_eval_action_binding.py --video-path playground/model_output/mymodel/action_5 --t2v-model mymodel` |
| Interaction | `python T2V-CompBench/compbench_eval_interaction.py --video-path playground/model_output/mymodel/interaction_6 --t2v-model mymodel` |

Optional: `--model-path` (default: `Qwen/Qwen3-VL-32B-Instruct`), `--output-path`, `--read-prompt-file`

### Detection-based Evaluation (GroundingDINO + SAM + Depth Anything)

| Category | Command |
|----------|---------|
| Spatial Relationships | `python T2V-CompBench/compbench_eval_spatial_relationships.py --video-path playground/model_output/mymodel/spatial_3 --t2v-model mymodel` |
| Numeracy | `python T2V-CompBench/compbench_eval_numeracy.py --video-path playground/model_output/mymodel/numeracy_7 --t2v-model mymodel` |

### Tracking-based Evaluation (GroundingSAM + DOT)

Motion binding requires two steps run sequentially:

```bash
# Step 1: Foreground/background segmentation
python T2V-CompBench/compbench_motion_binding_seg.py \
  --video-path playground/model_output/mymodel/motion_4 --t2v-model mymodel

# Step 2: Point tracking and motion scoring
python T2V-CompBench/compbench_eval_motion_binding.py \
  --video-path playground/model_output/mymodel/motion_4 --t2v-model mymodel
```

<a name="score_aggregation"></a>
## :bar_chart: Score Aggregation

After all 7 categories are evaluated, compute the final normalized scores:

```bash
python T2V-CompBench/compbench_eval_get_result.py --t2v-model mymodel
```

Normalization:
| Category | Formula |
|----------|---------|
| Consistent Attribute | `(score - 1) / 14` |
| Action Binding | `(score - 1) / 9` |
| Interaction | `(score - 1) / 9` |
| Dynamic Attribute | raw score |
| Numeracy | avg frame score per video |
| Spatial | avg of 2D and 3D video scores |
| Motion | negative scores clamped to 0, positive scaled by `0.8x + 0.2` |
| **Final** | **average of all 7 normalized scores** |

<a name="weight_reference"></a>
## :package: Weight File Reference

All weights are downloaded automatically by `install.sh`. For manual setup:

| Weight | Expected Path | Used By |
|--------|--------------|---------|
| GroundingDINO SwinT | `GSA/groundingdino_swint_ogc.pth` | spatial, numeracy, motion_seg |
| SAM ViT-H | `GSA/sam_vit_h_4b8939.pth` | spatial, motion_seg |
| DOT estimator | `dot/checkpoints/cvo_raft_patch_8.pth` | motion_binding |
| DOT refiner | `dot/checkpoints/movi_f_raft_patch_4_alpha.pth` | motion_binding |
| DOT tracker | `dot/checkpoints/movi_f_cotracker2_patch_4_wind_8.pth` | motion_binding |
| Qwen3-VL-32B-Instruct | HF cache (`$HF_HOME` / `~/.cache/huggingface`) | consistent_attr, dynamic_attr, action, interaction |
| Depth Anything | HF cache (`$HF_HOME`) | spatial (3D), auto-downloaded on first run |

<a name="citation"></a>
## :black_nib: Citation
If you find T2V-CompBench useful for your research, please cite our paper. :)
```
@article{sun2024t2v,
  title={T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation},
  author={Sun, Kaiyue and Huang, Kaiyi and Liu, Xian and Wu, Yue and Xu, Zihan and Li, Zhenguo and Liu, Xihui},
  journal={arXiv preprint arXiv:2407.14505},
  year={2024}
}
```
