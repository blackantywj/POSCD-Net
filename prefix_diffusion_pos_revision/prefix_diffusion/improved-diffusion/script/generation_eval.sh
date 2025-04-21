export CUDA_VISIBLE_DEVICES=0
python "scripts/infill_eval_coco_pos.py" \
    --model_path /home/wjworkspace/prefix_diffusion_pos_revision/prefix_diffusion/improved-diffusion/diffusion_model/coco//home/wjworkspace/prefix_diffusion_pos_revision/prefix_diffusion/improved-diffusion/diffusion_model/coco/1000_48_512_10bert+scm/ema_0.9999_200000.pt \
    --eval_task 'control_pos' \
    --use_ddim True\
    --notes "tree_adagrad" \
    --eta 1. \
    --verbose pipe \

