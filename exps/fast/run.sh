export CUDA_VISIBLE_DEVICES=$1
python train.py --data_path=~/datasets/cifar-10 --dataset=CIFAR10 --model=ResNet18LP --batch_size=128 --wd=1e-4 --lr_init=0.1 --epochs=200 \
--weight-exp=4 --weight-man=3 \
--activate-exp=4 --activate-man=3 \
--error-exp=4 --error-man=3