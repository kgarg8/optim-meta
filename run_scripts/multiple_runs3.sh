python main.py --device=1 --dataset=miniimagenet --alg=iMAML --imaml_reg=True --lamb=100 --n_inner=5 --lr_sched=True --num_epoch=150 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --num_way=5 --num_shot=5 --exp=imaml_mini_shot_5_way_5_lamb100 

python main.py --device=1 --dataset=miniimagenet --alg=iMAML --imaml_reg=True --lamb=2 --n_inner=5 --lr_sched=True --num_epoch=150 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --num_way=5 --num_shot=1 --exp=imaml_mini_shot_1_way_5

python main.py --device=1 --dataset=omniglot --in_channels=1 --alg=iMAML --imaml_reg=True --lamb=2 --n_inner=5 --lr_sched=True --num_epoch=150 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --num_way=5 --num_shot=1 --exp=imaml_omni_shot_1_way_5

python main.py --device=1 --dataset=omniglot --in_channels=1 --alg=iMAML --imaml_reg=True --lamb=2 --n_inner=5 --lr_sched=True --num_epoch=150 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --num_way=5 --num_shot=5 --exp=imaml_omni_shot_5_way_5 

python main.py --device=1 --dataset=omniglot --in_channels=1 --alg=iMAML --imaml_reg=True --lamb=2 --n_inner=5 --lr_sched=True --num_epoch=150 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --num_way=10 --num_shot=1 --exp=imaml_omni_shot_1_way_10

python main.py --device=1 --dataset=omniglot --in_channels=1 --alg=iMAML --imaml_reg=True --lamb=2 --n_inner=5 --lr_sched=True --num_epoch=150 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --num_way=10 --num_shot=5 --exp=imaml_omni_shot_5_way_10