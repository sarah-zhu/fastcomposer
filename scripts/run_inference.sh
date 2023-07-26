CAPTION="an APPL-Emoji of a man <|image|> wearing hat"
DEMO_NAME="marco1"

accelerate launch \
    --mixed_precision=no \
    fastcomposer/inference.py \
    --pretrained_model_name_or_path ./model/fastcomposer \
    --finetuned_model_path model/fastcomposer \
    --test_reference_folder data/${DEMO_NAME} \
    --test_caption "${CAPTION}" \
    --output_dir outputs/${DEMO_NAME} \
    --mixed_precision no \
    --image_encoder_type clip \
    --image_encoder_name_or_path openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --max_num_objects 2 \
    --object_resolution 224 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 7.5 \
    --inference_steps 50 \
    --start_merge_step 10 \
    --no_object_augmentation \
    --use_dreamtorch_unet
