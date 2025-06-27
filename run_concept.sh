python train_sdd.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --removing_concepts \
        "nudity" \
    --validation_prompts \
        "japan body" \
    --num_images_per_prompt 10 \
    --seed 42 \
    --devices 0 1 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --use_fp16 \
    --enable_xformers \
    --gradient_checkpointing \
    --num_train_steps 1000 &

python train_esd.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --removing_concepts \
        "nudity" \
    --validation_prompts \
        "japan body" \
    --num_images_per_prompt 10 \
    --seed 42 \
    --devices 2 3 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --use_fp16 \
    --enable_xformers \
    --gradient_checkpointing \
    --num_train_steps 1000 &

wait 