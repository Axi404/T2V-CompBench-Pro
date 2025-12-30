#!/bin/zsh

export model_name=$1
echo "Model Name: ${model_name}"

source ~/.zshrc

export HF_HOME=/root/T2V-CompBench-Pro/weights
export TRANSFORMERS_CACHE=/root/T2V-CompBench-Pro/weights
export HF_ENDPOINT=https://hf-mirror.com

conda activate t2v

python T2V-CompBench/compbench_eval_consistent_attr.py --video-path playground/model_output/${model_name}/consistent_attr_1 --t2v-model ${model_name}

python T2V-CompBench/compbench_eval_interaction.py --video-path playground/model_output/${model_name}/interaction_6 --t2v-model ${model_name}

python T2V-CompBench/compbench_eval_dynamic_attr.py --video-path playground/model_output/${model_name}/dynamic_attr_2 --t2v-model ${model_name}

python T2V-CompBench/compbench_eval_action_binding.py --video-path playground/model_output/${model_name}/action_5 --t2v-model ${model_name}

python T2V-CompBench/compbench_eval_numeracy.py --video-path playground/model_output/${model_name}/numeracy_7 --t2v-model ${model_name}

python T2V-CompBench/compbench_eval_spatial_relationships.py --video-path playground/model_output/${model_name}/spatial_3 --t2v-model ${model_name}

python T2V-CompBench/compbench_motion_binding_seg.py  --video-path playground/model_output/${model_name}/motion_4 --t2v-model ${model_name}
python T2V-CompBench/compbench_eval_motion_binding.py --video-path playground/model_output/${model_name}/motion_4 --t2v-model ${model_name}

python T2V-CompBench/compbench_eval_get_result.py --t2v-model ${model_name}