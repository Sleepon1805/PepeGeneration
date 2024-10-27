Used papers:
 - https://arxiv.org/abs/2006.11239
 - https://arxiv.org/abs/2011.13456
 - https://arxiv.org/abs/2206.00364

Used repos:
 - https://github.com/openai/improved-diffusion
 - https://github.com/awjuliani/pytorch-diffusion
 - https://github.com/dome272/Diffusion-Models-pytorch
 - https://github.com/yang-song/score_sde_pytorch

TODOs:
 - find appropriate quality metric for generated images (used for early stopping and checkpointing),
 - collect a new Pepe dataset, train a Pepe generation model,
 - try bigger image size or train an upscaling model,
 - (?) use flash attention (https://github.com/Dao-AILab/flash-attention),
 - (?) use deformable attention
 - visualise influence of condition on celeba generation,
 - is it possible to train an SDE-invariant model? (https://arxiv.org/abs/2206.00364)


Last results:

Celeba v11: unconditional, 64x64

![last_results_celeba](docs/celeba/final_pred.png)