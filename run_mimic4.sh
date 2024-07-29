CUDA_VISIBLE_DEVICES=0 python train.py --dataset=mimic3 --task=h --epochs=10 --lr=5e-4 --eval_steps=200 --batch_size=4
CUDA_VISIBLE_DEVICES=0 python train.py --dataset=mimic3 --task=m --epochs=10 --lr=5e-4 --eval_steps=200 --batch_size=4
