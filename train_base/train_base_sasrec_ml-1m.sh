python scripts/train_base/train_base.py \
 --seed 0 \
 --dataset=ml-1m \
 --num_workers=3 \
 --model=sasrec \
 --max_seq=100 \
 --num_epoch 1000 \
 --dropout_rate=0.2 \
 --l2=0.00001 \
 --device=cuda