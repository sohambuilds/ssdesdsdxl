export MAX_INFER_BATCH_SIZE=2

python generate.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pipeline_type auto \
    --image_dir "images/sdxl_test" \
    --prompt_path "prompts/country_body.txt" \
    --num_images_per_prompt 1 \
    --use_fp16 \
    --device "cuda:0" 