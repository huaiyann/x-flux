AE=/data/flux.1-dev/ae.safetensors \ 
    FLUX_DEV=/data/flux.1-dev/flux1-dev.safetensors \ 
    T5=/data/models/xlabs-ai-xflux_text_encoders \ 
    CLIP=/data/models/openai-clip-vit-large-patch14 \ 
    python3 ./main.py --width=256 --height=256 --seed=10 --model_type=flux-dev --num_steps=30 --prompt='a dog' --device=cuda --num_images_per_prompt=2 --custom_offload=True
