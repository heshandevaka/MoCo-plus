folder="logs"
mkdir -p $folder
dataset_path=office-home
dataset=office-home

weighting=MoCoPlus
seed=0
gpu_id=0
bs=128
weight_decay="1e-5"

k="1e-4"
w="1e-3"
sigma="0.1"
c="10"
gamma="100"
optim="adam"

echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --bs $bs --weighting $weighting --k_mocoplus $k --w_mocoplus $w --sigma_mocoplus $sigma --c_mocoplus $c  --gamma_mocoplus $gamma --weight_decay $weight_decay --optim $optim --momentum 0. > $folder/$dataset-$weighting-seed-$seed-$optim-k-$k-w-$w-sigma-$sigma-c-$c-gamma-$gamma-weight_decay-$weight_decay-bs-$bs.out"
python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --bs $bs --weighting $weighting --k_mocoplus $k --w_mocoplus $w --sigma_mocoplus $sigma --c_mocoplus $c  --gamma_mocoplus $gamma --weight_decay $weight_decay --optim $optim --momentum 0. > $folder/$dataset-$weighting-seed-$seed-$optim-k-$k-w-$w-sigma-$sigma-c-$c-gamma-$gamma-weight_decay-$weight_decay-bs-$bs.out