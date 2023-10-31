python scripts/retrospective/retro_exp.py \
--seed 0 \
--dataset ml-1m \
--model gru4rec \
--max_seq 100 \
--future_num 5 \
--K 10 \
--device cuda \
--model_path logs_base/gru4rec/ml-1m/model_500.model \
--step 500 \
--lr 0.01 \
--lam 10 \
--gam 1 \
--alp_1 1 \
--alp_2 1 \
--mask_thresh 0.5 \
--test_user_num 1000 \
--pool_size -1