KL=0.0001
CL=0.001
D=4

python train_chirpy3d.py \
     --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
     --train_data_dir=data/dogs \
     --resolution=256 --random_flip --train_batch_size=4 --gradient_accumulation_steps=4 \
     --num_train_epochs=100 --checkpointing_steps=750 --learning_rate=0.00001 --mapper_lr_scale=10 \
     --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 --output_dir="mvsd15-dogs-lr1e5-d${D}-kl${KL}-cl${CL}" \
     --validation_prompt="a photo of a <0:0> <2:0> <3:0> <4:0> <5:0> <6:0> <7:0>, 3d asset" \
     --num_validation_images 8 --num_parts 8 --num_k_per_part 256 --filename="train.txt" \
     --code_filename="train_caps_better_m8_k256.txt" --projection_nlayers=3 \
     --use_templates --vector_shuffle \
     --attn_loss=0.01 --use_gt_label --bg_code=1 \
     --partids="0,2,3,4,5,6,7" \
     --resume_from_checkpoint="latest" --mixed_precision="fp16" --attn_size="8,16" \
     --masked_training --vae_dims=${D} --recaption --concat_pe --learnable_dino_embeddings --exp_attn --attn_scale=10. --skip_background \
     --drop_tokens --drop_rate=0.5 --kl=${KL} --mixed_latent_reg=${CL} --skip_sa --add_reg_attn --mode="spatial"