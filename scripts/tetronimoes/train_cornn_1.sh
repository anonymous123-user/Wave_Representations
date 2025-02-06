# Useful variables
project_name="object-shape"
base_folder="experiments"
dataset="new_tetronimoes"
model_type="cornn_model"
N=64
batch_size=64
seed=200
num_classes=6
hidden_channels=16
dt=0.1
max_iters=300

extension=1
run_name="${dataset}/${model_type}/${extension}/${max_iters}iters"
run_dir="${base_folder}/${run_name}"

# RUN EXPERIMENTS
python main.py wandb.project="${project_name}" hydra.run.dir="${run_dir}" \
params.run_name="${run_name}" \
params.model_type="${model_type}" params.dataset="${dataset}" params.N=${N} \
params.batch_size=${batch_size} params.seed=${seed} params.max_iters=${max_iters} \
params.num_classes=${num_classes} params.dt=${dt} \
params.hidden_channels=${hidden_channels}

# DEACTIVATE ENVIRONMENT
mamba deactivate