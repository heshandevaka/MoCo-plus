folder="icassp_logs"
mkdir -p $folder
dataset_path=office-31
dataset=office-31


weighting=EW_STORM
seed=0
gpu_id=0
bs=128

LR_SET="1e-5"
WEIGHT_DECAY_SET="1e-5"
GAMMA_SET="100"
C_SET="10"
K_SET="1e-4"
W_SET="1e-3"
SIGMA_SET="0.1"
optim="adam"
# k="1e-2"
# --sigma_mo_storm $sigma --k_mo_storm $k

for lr in $LR_SET; do
    for weight_decay in $WEIGHT_DECAY_SET; do
        for gamma in $GAMMA_SET; do
            for c in $C_SET; do
                for sigma in $SIGMA_SET; do
                    for w in $W_SET; do
                        for k in $K_SET; do
                            echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --bs $bs --weighting $weighting  --lr $lr --c_ew_storm $c --w_ew_storm $w --k_ew_storm $k --sigma_ew_storm $sigma --weight_decay $weight_decay --optim $optim --momentum 0. > $folder/$dataset-$weighting-seed-$seed-$optim-lr-$lr-k-$k-w-$w-sigma-$sigma-c-$c-weight_decay-$weight_decay-bs-$bs.out"
                            python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --bs $bs --weighting $weighting  --lr $lr --c_ew_storm $c --w_ew_storm $w --k_ew_storm $k --sigma_ew_storm $sigma --weight_decay $weight_decay --optim $optim --momentum 0. > $folder/$dataset-$weighting-seed-$seed-$optim-lr-$lr-k-$k-w-$w-sigma-$sigma-c-$c-weight_decay-$weight_decay-bs-$bs.out
                        done
                    done
                done
            done
        done
    done
done


# folder="icassp_logs"
# mkdir -p $folder
# dataset_path=office-home
# dataset=office-home


# weighting=MO_STORM
# seed=0
# gpu_id=1
# bs=64 #128

# LR_SET="1e-5"
# WEIGHT_DECAY_SET="1e-5"
# GAMMA_SET="100"
# C_SET="10"
# K_SET="1e-4"
# W_SET="1e-2"
# SIGMA_SET="0.1"
# optim="adam"
# # k="1e-2"
# # --sigma_mo_storm $sigma --k_mo_storm $k

# for lr in $LR_SET; do
#     for weight_decay in $WEIGHT_DECAY_SET; do
#         for gamma in $GAMMA_SET; do
#             for c in $C_SET; do
#                 for sigma in $SIGMA_SET; do
#                     for w in $W_SET; do
#                         for k in $K_SET; do
#                             echo "python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --bs $bs --weighting $weighting  --lr $lr --gamma_mo_storm $gamma --c_mo_storm $c --w_mo_storm $w --k_mo_storm $k --sigma_mo_storm $sigma --weight_decay $weight_decay --optim $optim --momentum 0. > $folder/$dataset-$weighting-seed-$seed-$optim-lr-$lr-k-$k-w-$w-sigma-$sigma-c-$c-gamma-$gamma-weight_decay-$weight_decay-bs-$bs.out"
#                             python -u train_office.py --multi_input --seed $seed --gpu_id $gpu_id --dataset_path $dataset_path --dataset $dataset --bs $bs --weighting $weighting  --lr $lr --gamma_mo_storm $gamma --c_mo_storm $c --w_mo_storm $w --k_mo_storm $k --sigma_mo_storm $sigma --weight_decay $weight_decay --optim $optim --momentum 0. > $folder/$dataset-$weighting-seed-$seed-$optim-lr-$lr-k-$k-w-$w-sigma-$sigma-c-$c-gamma-$gamma-weight_decay-$weight_decay-bs-$bs.out
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done