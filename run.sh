# CUDA_VISIBLE_DEVICES=0 python main.py --cuda --dataset=alibaba-fashion -pre -pre_i -train -test
###########
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --dataset=yelp2018 -pre -pre_i -train -test
###########
# CUDA_VISIBLE_DEVICES=0 python main.py --cuda --dataset=last-fm -pre -pre_i -train -test
###########
# CUDA_VISIBLE_DEVICES=0 python main.py --cuda --dataset=amazon-book -pre -pre_i -train -test -pre_epoch 8