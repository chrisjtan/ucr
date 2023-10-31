python scripts/train_base/train_base.py \
  --seed 0 \
  --l2=0.00001 \
  --num_workers=3 \
  --dataset=yelp \
  --model=gru4rec \
  --max_seq=100 \
  --num_epoch 1000 \
  --dropout_rate=0.2 \
  --device=cuda