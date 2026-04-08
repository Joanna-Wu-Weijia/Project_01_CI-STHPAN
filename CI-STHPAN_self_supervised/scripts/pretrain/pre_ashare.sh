#!/usr/bin/env bash
# A 股预训练：--market AShare，数据来自 scripts/step1_qlib_to_csv.py
# step1 从 Qlib 的 instruments/csi300.txt 或 csi500.txt 读入成分并集，
# 导出 SH/SZ/BJ 全市场标的（与指数文件一致），不写死交易所。
# 切换指数时：改 step1 里的 INSTRUMENT_FILE，重新跑 step1，再跑本脚本。

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/StockForecasting" ]; then
    mkdir ./logs/StockForecasting
fi
seq_len=256
pred_len=1
model_name=Pretrain_ashare

model_id_name=AShare

data_name=stock

random_seed=2023


LOG=logs/StockForecasting/${model_name}_${model_id_name}_${seq_len}.log
echo "Logging to $LOG (also shown below). First run loads every stock CSV — can take many minutes."
python -u ./patchtst_pretrain.py \
      --random_seed $random_seed \
      --market $model_id_name \
      --context_points $seq_len \
      --target_points $pred_len \
      --graph 0 \
      --revin 1 \
      --ci 1 \
      --rel_type 0\
      --n_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --n_epochs_pretrain 100 \
      --batch_size 1 \
      --lr 0.0001 2>&1 | tee "$LOG"
