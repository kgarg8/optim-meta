# python main.py --device=2 --dataset=miniimagenet --alg=MAML --n_inner=5 --lr_sched=True --num_epoch=400 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=5 --num_shot=1 --exp=maml_mini_shot_1_way_5

# python main.py --device=2 --dataset=miniimagenet --alg=MAML --n_inner=5 --lr_sched=True --num_epoch=400 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=5 --num_shot=5 --exp=maml_mini_shot_5_way_5 

# python main.py --device=2 --dataset=omniglot --in_channels=1 --alg=MAML --n_inner=5 --lr_sched=True --num_epoch=100 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=5 --num_shot=1 --exp=maml_omni_shot_1_way_5

#python main.py --device=2 --dataset=omniglot --in_channels=1 --alg=MAML --n_inner=5 --lr_sched=True --num_epoch=100 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=5 --num_shot=5 --exp=maml_omni_shot_5_way_5 

python main.py --device=2 --dataset=omniglot --in_channels=1 --alg=MAML --n_inner=5 --lr_sched=True --num_epoch=100 --batch_size=1 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=10 --num_shot=1 --exp=maml_omni_shot_1_way_10 

python main.py --device=2 --dataset=omniglot --in_channels=1 --alg=MAML --n_inner=5 --lr_sched=True --num_epoch=100 --batch_size=1 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=10 --num_shot=5 --exp=maml_omni_shot_5_way_10
