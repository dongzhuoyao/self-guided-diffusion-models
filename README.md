# Self-Guided Diffusion Models




### Image-level Guidance

Baseline

```
python main.py data=in64_pickle dynamic=unet_fast sg.params.condition_method=label sg.params.cond_drop_prob=0.1 \
sg.params.cond_scale=2 dynamic.params.model_channels=128 sg.params.cond_dim=1000 \
name=label_in64 data.trainer.max_epochs=100 data.fid_every_n_epoch=10 \
pl.trainer.strategy=ddp devices=4 debug=0 data.params.batch_size=40 data.params.num_workers=4
```

Self-Labeled Guidance
 
```
python main.py data=in64_pickle dynamic=unet_fast sg.params.condition_method=cluster sg.params.cond_drop_prob=0.1 \
sg.params.cond_scale=2 dynamic.params.model_channels=128 sg.params.cond_dim=5000 \
name=self_labeled_in64 data.trainer.max_epochs=100 data.fid_every_n_epoch=10 \
pl.trainer.strategy=ddp devices=4 debug=0 data.params.batch_size=40 data.params.num_workers=4
```

### Box-level Guidance


Baseline
 
```
python main.py data=voc64 sg.params.condition_method=attr dynamic=unetca_fast dynamic.params.cond_token_num=1 
dynamic.params.context_dim=32 sg.params.cond_dim=21 sg.params.cond_drop_prob=0.1
sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=80
name=self_boxed_baseline_voc data.params.batch_size=80 debug=0
```


Self-Boxed Guidance
 
```
python main.py data=voc64 sg.params.condition_method=clusterlayout condition.clusterlayout.how=lost 
condition.clusterlayout.layout_dim=1 dynamic=unetca_fast dynamic.params.cond_token_num=1 
dynamic.params.context_dim=32 data.params.batch_size=80 condition.cluster.feat_cluster_k=100
sg.params.cond_dim=100 
data.h5_file=data/sg_data/cluster/v3_voc64_cluster100_iter30minp200_nns-1_dino_vits16_2022-08-11T20_311135d.h5 
sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 
data.fid_every_n_epoch=80 name=self_boxed_voc debug=0

```


### Pixel-level Guidance
 

Baseline
 
```
python main.py data=voc64 sg.params.condition_method=attr dynamic=unetca_fast dynamic.params.cond_token_num=1 \
dynamic.params.context_dim=32 sg.params.cond_dim=21 sg.params.cond_drop_prob=0.1 \
sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=80 \
name=self_segmented_baseline_voc data.params.batch_size=80 debug=0
```


Self-Segmented Guidance, VOC
 
```
python main.py data=voc64 sg.params.condition_method=stegoclusterlayout condition.stegoclusterlayout.how=stego \
condition.stegoclusterlayout.layout_dim=21 dynamic=unetca_fast dynamic.params.cond_token_num=1 \
sg.params.cond_dim=21 dynamic.params.context_dim=32 data.params.batch_size=80 \
sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 \
data.fid_every_n_epoch=80 name=self_segmented_voc debug=0

```

Self-Segmented Guidance, COCO


```
python main.py data=cocostuff64 sg.params.condition_method=stegoclusterlayout condition.stegoclusterlayout.how=stego condition.stegoclusterlayout.layout_dim=27 dynamic=unetca_fast dynamic.params.cond_token_num=1 sg.params.cond_dim=27 dynamic.params.context_dim=32 data.params.batch_size=80 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=80 name=self_segmented_cocostuff64 debug=0
```

multi-gpu:

```
pl.trainer.strategy=ddp devices=4
```

multi-gpu, with torch2.0

```
pl.trainer.strategy=ddp_find_unused_parameters_true devices=4
```



### Params


- dynamic: UNet structure
- sg.params.condition_method: condition method, could be label or cluster
- sg.params.cond_drop_prob: guidance drop probability, default 0.1
- sg.params.cond_scale: guidance strength scale, default 2
- dynamic.params.model_channels: the basic channel number of UNet, default 128 
- sg.params.cond_dim: the dimension of guidance signal 
- data.trainer.max_epochs: max_epochs
- data.fid_every_n_epoch: fid_every_n_epoch
- pl.trainer.strategy: multi-gpu training mode, default "ddp"
- devices: gpu number
- data.params.batch_size: batch size 
- data.params.num_workers: num_workers




### Prepare clustering(including extracting feature)



##### Extracting feature

```
python feat_extractor.py  --feat dino_vitb16 --ds in32p --bs 32 --image_size 32 --debug 0
```

it will generate a path FEAT_H5_PATH that can be used in clustering


##### do clustering

```
python cluster_on_feat.py --feat_h5 'FEAT_H5_PATH' --k 5000 --debug 0
```


## Environment Preparation


```
conda create -n sgdm python=3.9
pip install opencv-python pudb==2019.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.6.3 omegaconf==2.1.1 einops==0.3.0 torch-fidelity==0.3.0  transformers==4.3.1
conda install pytorch cudatoolkit=11.3 -c pytorch -c nvidia
conda install -c conda-forge pytorch=*=*cuda*  cudatoolkit=11.3 -c nvidia
pip install timm==0.4.5 torchmetrics==v0.7.0 hydra-core loguru sklearn clean-fid matplotlib lightning-bolts rich torchvision
pip install einops torch_fidelity albumentations
pip install --upgrade wandb
pip install POT h5py einops_exts blobfile GitPython
conda install -c conda-forge mpi4py 
pip install pycocotools seaborn cyanure distinctipy 
pip install pytorch-fid
pip install  -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install  -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install  -e .
    
pip install git+https://github.com/dongzhuoyao/pytorch-fid-with-sfid

conda install -n sgdm ipykernel --update-deps --force-reinstall
pip install notebook
```



```
conda create -n sgdm_p310t20c118 python=3.9
pip install opencv-python pudb==2019.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.6.3 omegaconf==2.1.1 einops==0.3.0 torch-fidelity==0.3.0  transformers
conda install pytorch cudatoolkit=11.3 -c pytorch -c nvidia
conda install -c conda-forge pytorch=*=*cuda*  cudatoolkit=11.3 -c nvidia
pip install timm==0.4.5 torchmetrics==v0.7.0 hydra-core loguru sklearn clean-fid matplotlib lightning-bolts rich torchvision
pip install einops torch_fidelity albumentations
pip install --upgrade wandb
pip install pytorch-lightning==1.8.0 #cannot be 2.0
pip install POT h5py einops_exts blobfile GitPython
conda install -c conda-forge mpi4py 
pip install pycocotools seaborn cyanure distinctipy 
pip install pytorch-fid
pip install  -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install  -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install  -e .
    
pip install git+https://github.com/dongzhuoyao/pytorch-fid-with-sfid

conda install -n sgdm_p310t20c118 ipykernel --update-deps --force-reinstall
pip install notebook
```



```
conda env create -f environment.yaml
conda activate sgdm
```


## Dataset Preparation

### Imagenet

Download and prepare imagenet following [Imagenet32_Scripts](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts)

For large-resolution Imagenet, please follow [https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4)


### COCOstuff

-  Download COCO stuff from [STEGO repo](https://github.com/mhamilton723/STEGO)
-  Run dataset/ds_utils/extract_cocostuff_from_coco17.py


Prepare box(optional):
- 'python LOST/sg_main_lost.py sample=cocostuff27'

Prepare mask(optional):
- 'python STEGO/src/sg_generate_segmask.py sample=cocostuff27'

### Pascal VOC

- Download VOC12 dataset from [official website](http://host.robots.ox.ac.uk/pascal/VOC/)
- Prepare the dataset following [official practice](http://host.robots.ox.ac.uk/pascal/VOC/)

Prepare box(optional):
- 'python LOST/sg_main_lost.py sample=voc12'

Prepare mask(optional):
- 'python STEGO/src/sg_generate_segmask.py sample=voc12'


