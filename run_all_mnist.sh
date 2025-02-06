#!/bin/bash

script="scripts"

sbatch ${script}/mnist/train_baseline_1.sh
sbatch ${script}/mnist/train_baseline_2.sh
sbatch ${script}/mnist/train_baseline_3.sh

sbatch ${script}/mnist/train_cornn_1.sh
sbatch ${script}/mnist/train_cornn_2.sh
sbatch ${script}/mnist/train_cornn_3.sh

sbatch ${script}/mnist/train_lstm_1.sh
sbatch ${script}/mnist/train_lstm_2.sh
sbatch ${script}/mnist/train_lstm_3.sh

sbatch ${script}/mnist/train_rnn_1.sh
sbatch ${script}/mnist/train_rnn_2.sh
sbatch ${script}/mnist/train_rnn_3.sh

sbatch ${script}/mnist/train_unet_1.sh
sbatch ${script}/mnist/train_unet_2.sh
sbatch ${script}/mnist/train_unet_3.sh

sbatch ${script}/mnist/train_hidden_lstm_1.sh
sbatch ${script}/mnist/train_hidden_lstm_2.sh
sbatch ${script}/mnist/train_hidden_lstm_3.sh

sbatch ${script}/mnist/train_hidden_rnn_1.sh
sbatch ${script}/mnist/train_hidden_rnn_2.sh
sbatch ${script}/mnist/train_hidden_rnn_3.sh