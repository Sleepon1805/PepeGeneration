Used repos:
 - https://github.com/openai/improved-diffusion
 - https://github.com/awjuliani/pytorch-diffusion
 - https://github.com/dome272/Diffusion-Models-pytorch

TODOs:
 - log FID,
 - (?) setup hparams logging,
 - try out pepe emoticons from bttv and 7tv,
 - (?) find a way to train on pepe dataset without celeba pretraining,
 - try bigger image size or interpolate (upscale) images to bigger size,
 - try different number of diffusion steps
 - (?) use flash attention,
 - visualise influence of condition on celeba generation

Modifications:
 - (?) fix distribution during training,
 - conditioning,
 - super resolution model,
 - SDE in diffusion (https://arxiv.org/pdf/2011.13456.pdf, https://arxiv.org/pdf/2206.00364.pdf),
 - use conditional embedding to apply features from celeba dataset on generated Pepe,

Last results:

Celeba v1: unconditional

![last_results_celeba_v1](docs/celeba/final_pred.png)

Pepe v6: unconditional

![last_results_pepe_v6](docs/pepe/final_pred.png)