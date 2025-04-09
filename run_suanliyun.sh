AE=/data/models/flux1dev/ae.safetensors \
    FLUX_DEV=/data/models/flux1dev/flux1-dev.safetensors \
    T5=/data/models/xlabs-ai-xflux_text_encoders \
    CLIP=/data/models/openai-clip-vit-large-patch14 \
    python3 ./main.py --width=816 --height=1449 --seed=10 --model_type=flux-dev --num_steps=25 \
    --prompt='1man, adult male, exquisitely beautiful facial features, the highest image quality, close - up portrait, exquisite painting style, delicate face, waist-up pose in standing position: 1.1, occupying 50% of the frame, look at camera, upper body, full clothes, normal fingers,' \
    --device=cuda --num_images_per_prompt=2 --custom_offload=True --guidance=8 --timestep_to_start_cfg=999
