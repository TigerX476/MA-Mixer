export CUDA_VISIBLE_DEVICES=1

model_name=MA_Mixer

python -u run.py \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --learning_rate 0.0002 \
  --d_state 2 \
  --d_ff 128 \
  --itr 1 \
  --dropout 0.2


python -u run.py \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --d_model 128 \
  --d_state 2 \
  --d_ff 128 \
  --itr 1 \
  --dropout 0.2

python -u run.py \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --d_model 128 \
  --d_state 2 \
  --d_ff 128 \
  --itr 1 \
  --dropout 0.2


python -u run.py \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --d_model 128 \
  --d_state 2 \
  --d_ff 128 \
  --itr 1 \
  --dropout 0.2