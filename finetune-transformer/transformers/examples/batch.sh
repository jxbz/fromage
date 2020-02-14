export SQUAD_DIR=../data/squad1

for seed in 100 200 300
do
  python run_squad.py \
--model_type bert \
--model_name_or_path bert-base-cased \
--do_train \
--do_eval \
--do_lower_case \
--train_file $SQUAD_DIR/train-v1.1.json \
--predict_file $SQUAD_DIR/dev-v1.1.json \
--per_gpu_train_batch_size 12 \
--optim adam \
--learning_rate 0.0001 \
--num_train_epochs 2.0 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir ./logs/adam0.0001seed$seed \
--logging_steps 1000 \
--save_steps 1000000 \
--evaluate_during_training \
--seed $seed
  python run_squad.py \
--model_type bert \
--model_name_or_path bert-base-cased \
--do_train \
--do_eval \
--do_lower_case \
--train_file $SQUAD_DIR/train-v1.1.json \
--predict_file $SQUAD_DIR/dev-v1.1.json \
--per_gpu_train_batch_size 12 \
--optim fromage \
--learning_rate 0.001 \
--num_train_epochs 2.0 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir ./logs/fromage0.001seed$seed \
--logging_steps 1000 \
--save_steps 1000000 \
--evaluate_during_training \
--seed $seed
  python run_squad.py \
--model_type bert \
--model_name_or_path bert-base-cased \
--do_train \
--do_eval \
--do_lower_case \
--train_file $SQUAD_DIR/train-v1.1.json \
--predict_file $SQUAD_DIR/dev-v1.1.json \
--per_gpu_train_batch_size 12 \
--optim SGD \
--learning_rate 0.1 \
--num_train_epochs 2.0 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir ./logs/SGD0.1seed$seed \
--logging_steps 1000 \
--save_steps 1000000 \
--evaluate_during_training \
--seed $seed
done