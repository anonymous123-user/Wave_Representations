# Useful variables
project_name="object-shape"
base_folder="experiments"
dataset="mnist"
model_type="baseline1_flexible"
N=56
batch_size=64
max_iters=300
num_classes=11
hidden_channels=16
extension=3

# 2 Layers
num_layers=2
seed=420
run_name="${dataset}/${model_type}/${extension}/${num_layers}"
run_dir="${base_folder}/${run_name}"
python main.py wandb.project="${project_name}" hydra.run.dir="${run_dir}" \
params.run_name="${run_name}" \
params.model_type="${model_type}" params.dataset="${dataset}" params.N=${N} \
params.batch_size=${batch_size} params.seed=${seed} params.max_iters=${max_iters} \
params.num_classes=${num_classes} \
params.hidden_channels=${hidden_channels} params.cell_type=${cell_type} params.num_layers=${num_layers}

# 4 Layers
num_layers=4
seed=421
run_name="${dataset}/${model_type}/${extension}/${num_layers}"
run_dir="${base_folder}/${run_name}"
python main.py wandb.project="${project_name}" hydra.run.dir="${run_dir}" \
params.run_name="${run_name}" \
params.model_type="${model_type}" params.dataset="${dataset}" params.N=${N} \
params.batch_size=${batch_size} params.seed=${seed} params.max_iters=${max_iters} \
params.num_classes=${num_classes} \
params.hidden_channels=${hidden_channels} params.cell_type=${cell_type} params.num_layers=${num_layers}

# 8 Layers
num_layers=8
seed=422
run_name="${dataset}/${model_type}/${extension}/${num_layers}"
run_dir="${base_folder}/${run_name}"
python main.py wandb.project="${project_name}" hydra.run.dir="${run_dir}" \
params.run_name="${run_name}" \
params.model_type="${model_type}" params.dataset="${dataset}" params.N=${N} \
params.batch_size=${batch_size} params.seed=${seed} params.max_iters=${max_iters} \
params.num_classes=${num_classes} \
params.hidden_channels=${hidden_channels} params.cell_type=${cell_type} params.num_layers=${num_layers}

# 16 Layers
num_layers=16
seed=423
run_name="${dataset}/${model_type}/${extension}/${num_layers}"
run_dir="${base_folder}/${run_name}"
python main.py wandb.project="${project_name}" hydra.run.dir="${run_dir}" \
params.run_name="${run_name}" \
params.model_type="${model_type}" params.dataset="${dataset}" params.N=${N} \
params.batch_size=${batch_size} params.seed=${seed} params.max_iters=${max_iters} \
params.num_classes=${num_classes} \
params.hidden_channels=${hidden_channels} params.cell_type=${cell_type} params.num_layers=${num_layers}

# DEACTIVATE ENVIRONMENT
mamba deactivate