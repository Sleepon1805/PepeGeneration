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
 - (?) fix hparams logging in tensorboard,
 - fix early stopping,
 - collect a new Pepe dataset, train a Pepe generation model,
 - try bigger image size or train an upscaling model,
 - (?) use flash attention (https://github.com/Dao-AILab/flash-attention),
 - visualise influence of condition on celeba generation,
 - implement conditional SDE Sampling,
 - implement generalized SDE Sampling,
 - try to use direct solution of SDE (i.e. Ornstein-Uhlenbeck)

Last results:

Celeba v11: unconditional

![last_results_celeba](docs/celeba/final_pred.png)

Pepe v6: unconditional

![last_results_pepe](docs/pepe/final_pred.png)