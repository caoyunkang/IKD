categories=("carpet" "grid" "leather" "tile" "wood" 'bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut' 'pill'\
                 'screw' 'toothbrush' 'transistor' 'zipper')

for category in ${categories[@]}; do

    python train.py \
      --category $category \
      --cal_pro False \
      --validate_step 10 \
      --model 'hrnet32' \
      --gamma 1 \
      --beta 2
    done
