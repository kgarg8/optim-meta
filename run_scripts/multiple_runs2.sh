# python main.py --device=1 --dataset=miniimagenet --alg=FOMAML --n_inner=5 --lr_sched=True --num_epoch=400 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=5 --num_shot=1 --exp=fomaml_mini_shot_1_way_5

# python main.py --device=1 --dataset=miniimagenet --alg=FOMAML --n_inner=5 --lr_sched=True --num_epoch=400 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=5 --num_shot=5 --exp=fomaml_mini_shot_5_way_5 

# python main.py --device=1 --dataset=omniglot --in_channels=1 --alg=FOMAML --n_inner=5 --lr_sched=True --num_epoch=400 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=5 --num_shot=1 --exp=fomaml_omni_shot_1_way_5

# python main.py --device=1 --dataset=omniglot --in_channels=1 --alg=FOMAML --n_inner=5 --lr_sched=True --num_epoch=400 --batch_size=2 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=5 --num_shot=5 --exp=fomaml_omni_shot_5_way_5 

# python main.py --device=1 --dataset=omniglot --in_channels=1 --alg=FOMAML --n_inner=5 --lr_sched=True --num_epoch=400 --batch_size=1 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=10 --num_shot=1 --exp=fomaml_omni_shot_1_way_10

# python main.py --device=1 --dataset=omniglot --in_channels=1 --alg=FOMAML --n_inner=5 --lr_sched=True --num_epoch=400 --batch_size=1 --inner_lr=1e-2 --outer_lr=1e-3 --net=functional_net --num_way=10 --num_shot=5 --exp=fomaml_omni_shot_5_way_10
