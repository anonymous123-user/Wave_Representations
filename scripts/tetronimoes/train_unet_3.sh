# Useful variables
project_name="object-shape"
base_folder="experiments"
dataset="new_tetronimoes"
model_type="unet"
batch_size=64
num_classes=6
seed=702
extension=3
run_name="${dataset}/${model_type}/${extension}"
run_dir="${base_folder}/${run_name}"

python main.py wandb.project="${project_name}" hydra.run.dir="${run_dir}" \
params.run_name="${run_name}" \
params.model_type="${model_type}" params.dataset="${dataset}" \
params.batch_size=${batch_size} params.seed=${seed} \
params.num_classes=${num_classes}

# DEACTIVATE ENVIRONMENT
mamba deactivate