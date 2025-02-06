# Useful variables
project_name="object-shape"
base_folder="experiments"
dataset="mnist"
model_type="baseline3"
N=56
batch_size=64
seed=781
max_iters=100
num_classes=11
hidden_channels=16
cell_type="lstm"
extension=2

run_name="${dataset}/${model_type}/${extension}/${cell_type}_${max_iters}"
run_dir="${base_folder}/${run_name}"

# RUN EXPERIMENTS
python main.py wandb.project="${project_name}" hydra.run.dir="${run_dir}" \
params.run_name="${run_name}" \
params.model_type="${model_type}" params.dataset="${dataset}" params.N=${N} \
params.batch_size=${batch_size} params.seed=${seed} params.max_iters=${max_iters} \
params.num_classes=${num_classes} \
params.hidden_channels=${hidden_channels} params.cell_type=${cell_type}

# DEACTIVATE ENVIRONMENT
mamba deactivate