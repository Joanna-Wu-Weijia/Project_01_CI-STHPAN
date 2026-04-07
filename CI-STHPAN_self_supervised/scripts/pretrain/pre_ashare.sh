

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
