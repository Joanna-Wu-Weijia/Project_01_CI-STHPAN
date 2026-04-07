#!/usr/bin/env bash
# Run from anywhere: CI-STHPAN_self_supervised (recommended) or via scripts/finetune/ft_ashare.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/StockForecasting" ]; then
    mkdir ./logs/StockForecasting
fi
seq_len=256
pred_len=1
model_name=Finetune_ashare

model_id_name=AShare

data_name=stock

random_seed=2023

# Alphas run one after another (not in parallel). a=8.log stays empty until a=1,2,4,6 finish.
export PYTHONUNBUFFERED=1

for alpha in 1 2 4 6 8 10
do
  LOGFILE="logs/StockForecasting/${model_name}_${model_id_name}_${seq_len}_a=${alpha}.log"
  echo ""
  echo "========== Finetune alpha=${alpha} (logging to ${LOGFILE}) =========="
  echo "First loads ~462 stocks x3 splits — can be quiet 10–30+ min before epoch lines appear."
  # tee: same output on terminal and in log; 2>&1 captures errors in the log too
  python -u ./patchtst_finetune.py \
      --random_seed $random_seed \
      --market $model_id_name \
      --is_finetune 1 \
      --is_linear_probe 0 \
      --context_points $seq_len \
      --target_points $pred_len \
      --graph 0 \
      --revin 1 \
      --ci 1 \
      --n_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --n_epochs_finetune 20 \
      --alpha $alpha \
      --batch_size 1 \
      --lr 0.0001 2>&1 | tee "$LOGFILE"
  echo "========== Finished alpha=${alpha} =========="
done
