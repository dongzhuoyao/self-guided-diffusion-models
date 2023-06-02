# test_my_unittest
from hydra import initialize, compose

# 1. initialize will add config_path the config search path within the context
# 2. The module with your configs should be importable.
#    it needs to have a __init__.py (can be empty).
# 3. THe config path is relative to the file calling initialize (this file)
from main import run_without_decorator


def test_with_initialize() -> None:
    common_command = ["hydra.runtime.output_dir='./'", "debug=1"]

    with initialize(version_base=None, config_path="config"):
        if False:
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in32_pickle sg.params.condition_method=cluster_lookup dynamic=unet_fast dynamic.params.model_channels=128 data.params.batch_size=128 sg.params.cond_dim=100 data.h5_file=sg_data/cluster/v3_in32p_cluster10000_iter30minp200_nns-1_dino_vits16_2022-08-12T20_7383c8d.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=20 data.fid_every_n_epoch=4 name=v1.6.2_dino_vits16_cluster10k_in32p_unetfast_ep20_rerun debug=0".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )
        elif False:#5k
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in32_pickle sg.params.condition_method=cluster dynamic=unet_fast dynamic.params.model_channels=128 data.params.batch_size=128  sg.params.cond_dim=50000 data.h5_file=sg_data/cluster/v4_in32p_cluster50000_iter30minp200_nns-1_dino_vits16_grey0_2022-11-09T18_b4b1257.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=20 data.fid_every_n_epoch=4 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:#corruption
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in32_pickle sg.params.condition_method=cluster dynamic=unet_fast dynamic.params.model_channels=128 data.params.batch_size=128 data.corruption=0.5 sg.params.cond_dim=5000 data.h5_file=sg_data/cluster/v3_in32p_cluster5000_iter30minp200_nns-1_dino_vits16_2022-08-20T19_2a4fe12.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=20 data.fid_every_n_epoch=4 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif True:#subgroup
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in32_pickle  sg.params.condition_method=label dynamic=unet_fast dynamic.params.model_channels=128 data.params.batch_size=128 data.subgroup=5  sg.params.cond_dim=5000  sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=20 data.fid_every_n_epoch=4 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )
             


        elif False:
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=coco64 sg.params.condition_method=clusterlayout condition.clusterlayout.how=lost condition.clusterlayout.layout_dim=1 dynamic=unetca_fast_s64 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 dynamic.params.model_channels=128 data.params.batch_size=80 condition.cluster.feat_cluster_k=100 sg.params.cond_dim=100 data.h5_file='~/data/sg_data/cluster/v3_coco64_cluster100_iter30minp200_nns-1_dino_vits16_2022-08-11T20_311135d.h5' data.lost_file=dataset/data_files/lost/LOST-vit_small16_k_COCO20k_train/lost_k100.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=400 data.fid_every_n_epoch=40 name=v1.6.2_dino_vits16_cluster100layout_coco64_unetca_fast_s64_ep400 debug=0 train=0 resume_from='outputs/v1.6.2_dino_vits16_cluster100layout_coco64_unetca_fast_s64_ep400/16-08-2022/21-06-46/ckpts/last.ckpt' train=0 name='analysis_coco'  +vis.voc_vis_samelayout=1 ".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )
        elif False:
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides=" data=voc64 sg.params.condition_method=attr  dynamic=unetca_fast   dynamic.params.cond_token_num=1  dynamic.params.context_dim=32  sg.params.cond_dim=21 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=40 name=v1.6.2_label_voc64_unetca_fast data.params.batch_size=80 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )
        elif False:  # vis_voc_lost
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides=" data=voc64 sg.params.condition_method=clusterlayout condition.clusterlayout.how=lost condition.clusterlayout.layout_dim=1 dynamic=unetca_fast_s64 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 dynamic.params.model_channels=128  condition.cluster.feat_cluster_k=100 sg.params.cond_dim=100 data.h5_file='data/sg_data/cluster/v3_voc64_cluster100_iter30minp200_nns-1_dino_vits16_2022-08-11T20_311135d.h5' data.lost_file=dataset/data_files/lost/LOST-vit_small16_k_voc12aug/lost_k100.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=40 name=vis_voc_lost debug=0 resume_from='outputs/v1.6.2_dino_vits16_cluster100layout_voc64_unetca_fast_s64_ep800/16-08-2022/21-05-43/ckpts/epoch\=000640.ckpt' train=0 data.params.batch_size=10  +vis.voc_lost=1 ".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # vis_voc_stego
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides=" data=voc64 sg.params.condition_method=clusterlayout condition.clusterlayout.how=stego condition.clusterlayout.layout_dim=21 dynamic=unetca_fast_s64 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 dynamic.params.model_channels=128 data.params.batch_size=80 condition.cluster.feat_cluster_k=100 sg.params.cond_dim=100 data.h5_file=data/sg_data/cluster/v3_voc64_cluster100_iter30minp200_nns-1_dino_vits16_2022-08-11T20_311135d.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=40 name=vis_voc_stego resume_from='outputs/v1.6.2_dino_vits16_cluster100layout_stego_voc64_unetca_fast_s64_ep800/23-08-2022/16-54-40/ckpts/epoch\=000720.ckpt' train=0 data.params.batch_size=10  +vis.voc_stego=1 ".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # vis_voc_stego_layout
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides=" data=voc64 sg.params.condition_method=layout condition.layout.how=stego condition.layout.layout_dim=21 dynamic=unetca_fast_s64 dynamic.params.cond_token_num=0 dynamic.params.context_dim=32  data.params.batch_size=80  sg.params.cond_dim=0  sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=40".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # stegoclusterlayout
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cocostuff64 sg.params.condition_method=stegoclusterlayout condition.stegoclusterlayout.how=stego condition.stegoclusterlayout.layout_dim=27 dynamic=unetca_fast dynamic.params.cond_token_num=1 sg.params.cond_dim=27 dynamic.params.context_dim=32 data.params.batch_size=80  sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=400 data.fid_every_n_epoch=40 name=v1.6.2_dino_vits16_stegoclusterlayout_cocostuff64_unetca_fast_ep400".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # in32, label
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in32_pickle dynamic=unet_fast sg.params.condition_method=label sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 dynamic.params.model_channels=128 sg.params.cond_dim=1000 data.params.batch_size=128 name=v1.6.2_label_in32p_unet_fast_ep100 debug=0 data.trainer.max_epochs=100 data.fid_every_n_epoch=10  debug=0 data.params.batch_size=200 resume_from=outputs/v1.6.2_label_in32p_unet_fast_ep100/15-09-2022/19-05-29/ckpts/last.ckpt train=0 +vis.samecondition=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # in64,label
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in64_pickle dynamic=unet_fast sg.params.condition_method=label sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 dynamic.params.model_channels=128 sg.params.cond_dim=1000 name=v1.6.2_label_in64p_unet_fast_ep100_vis data.trainer.max_epochs=100 data.fid_every_n_epoch=10 debug=0 data.params.batch_size=100 resume_from=outputs/v1.6.2_label_in64p_unet_fast_ep100/15-09-2022/19-05-29/ckpts/last.ckpt +vis.samecondition=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # VOC,lost
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=voc64 sg.params.condition_method=clusterlayout condition.clusterlayout.how=lost condition.clusterlayout.layout_dim=1 dynamic=unetca_fast dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 data.params.batch_size=80  sg.params.cond_dim=100 data.h5_file=sg_data/cluster/v3_voc64_cluster100_iter30minp200_nns-1_dino_vits16_2022-08-11T20_311135d.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=80 name=v1.6.2_dino_vits16_cluster100layout_voc64_unetca_fast_ep800 debug=0 resume_from='outputs/v1.6.2_dino_vits16_cluster100layout_voc64_unetca_fast_ep800/10-09-2022/00-26-28/ckpts/last.ckpt' train=0   +vis.lost_chainvis=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # VOC,attr
            # +exp.ablate_scale=1  +exp.ablate_scale_list=[4,0.0,0.5,1,1.5,2,2.5,3,3.5]
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=voc64 sg.params.condition_method=attr dynamic=unetca_fast dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_dim=21 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=80 name=v1.6.2_label_voc64_unetca_fast_ep800_vis data.params.batch_size=80 debug=0 resume_from='outputs/v1.6.2_label_voc64_unetca_fast_ep800/10-09-2022/00-15-33/ckpts/last.ckpt' train=0   +vis.samecondition=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # cocosstuff64, attr-guidance, vis.diff_z_same_stego=1
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cocostuff64 sg.params.condition_method=attr dynamic=unetca_fast dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_dim=27 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=4000 data.fid_every_n_epoch=40 name=v1.6.2_attr27_cocostuff64_unetca_fast_vis data.params.batch_size=128 debug=0 train=0 resume_from='outputs/v1.6.2_attr27_cocostuff64_unetca_fast/05-09-2022/01-00-46/ckpts/last.ckpt' train=0 +vis.samecondition=1 ".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # cocosstuff64, stego, vis.stego_chainvis=1
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cocostuff64 sg.params.condition_method=layout condition.layout.how=stego condition.layout.layout_dim=27 dynamic=unetca_fast dynamic.params.cond_token_num=0 dynamic.params.context_dim=32 data.params.batch_size=80 sg.params.cond_dim=0 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=400 data.fid_every_n_epoch=40 name=v1.6.2_dino_vits16_stego_cocostuff64_unetca_fast_ep400_vis resume_from=outputs/v1.6.2_dino_vits16_stego_cocostuff64_unetca_fast_ep400/03-09-2022/21-59-52/ckpts/last.ckpt train=0 +vis.stego_chainvis=1 ".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # cocosstuff64, stego, vis.diff_z_same_stego=1
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cocostuff64 sg.params.condition_method=layout condition.layout.how=stego condition.layout.layout_dim=27 dynamic=unetca_fast dynamic.params.cond_token_num=0 dynamic.params.context_dim=32 data.params.batch_size=80 sg.params.cond_dim=0 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=400 data.fid_every_n_epoch=40 name=v1.6.2_dino_vits16_stego_cocostuff64_unetca_fast_ep400_vis resume_from=outputs/v1.6.2_dino_vits16_stego_cocostuff64_unetca_fast_ep400/03-09-2022/21-59-52/ckpts/last.ckpt train=0 +vis.diff_z_same_stego=1 ".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # vis, in32, vits10k,
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in32_pickle sg.params.condition_method=cluster dynamic=unet_fast dynamic.params.model_channels=128 data.params.batch_size=320  sg.params.cond_dim=10000 data.h5_file=sg_data/cluster/v3_in32p_cluster10000_iter30minp200_nns-1_dino_vitb16_2022-08-17T21_7b919c8.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=100 data.fid_every_n_epoch=10 name=v1.6.2_dino_vitb16_cluster10k_in32p_unetfast_ep100_4gpu_vis  debug=0 train=0 resume_from='outputs/v1.6.2_dino_vitb16_cluster10k_in32p_unetfast_ep100_4gpu/21-08-2022/01-16-31/ckpts/last.ckpt'  seed=66 +vis.samecondition=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # vis, in64, vitb10k,
            # +vis.interp=1 +vis.interp_c.samples=16 +vis.interp_c.n=9
            # +vis.chainvis=1 +vis.chainvis_c.samples=7 +vis.chainvis_c.timestep=7
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in64_pickle sg.params.condition_method=cluster dynamic=unet_fast dynamic.params.model_channels=128  sg.params.cond_dim=10000 data.h5_file=sg_data/cluster/v3_in64p_cluster10000_iter30minp200_nns-1_dino_vitb16_2022-08-20T23_4e15934.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2  data.trainer.max_epochs=100 data.fid_every_n_epoch=10 debug=0 name=v1.6.2_dino_vitb16_cluster10k_in64p_unetfast_ep100_4gpu_inferencelast_vis resume_from=outputs/v1.6.2_dino_vitb16_cluster10k_in64p_unetfast_ep100_4gpu/04-09-2022/10-57-50/ckpts/last.ckpt   data.params.batch_size=128 train=0 seed=66  +vis.interp=1 +vis.interp_c.samples=16 +vis.interp_c.n=9".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # VIS, VOC, LOST
            #  exp.cond_scale=0 exp.ablate_scale=1  exp.ablate_scale_list=[0,1,2,3,0.5,1.5,2.5,3.5,4] data.test_fid_num=500  name=voc_condscale_lost
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=voc64 sg.params.condition_method=clusterlayout condition.clusterlayout.how=lost condition.clusterlayout.layout_dim=1 dynamic=unetca_fast dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 data.params.batch_size=80  sg.params.cond_dim=100 data.h5_file=sg_data/cluster/v3_voc64_cluster100_iter30minp200_nns-1_dino_vits16_2022-08-11T20_311135d.h5 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=80 name=v1.6.2_dino_vits16_cluster100layout_voc64_unetca_fast_ep800 debug=0 resume_from='outputs/v1.6.2_dino_vits16_cluster100layout_voc64_unetca_fast_ep800/10-09-2022/00-26-28/ckpts/last.ckpt' train=0  exp.cond_scale=0 exp.ablate_scale=1  exp.ablate_scale_list=[0,1,2,3,0.5,1.5,2.5,3.5,4] data.test_fid_num=500  name=voc_condscale_lost_f".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # VIS, VOC, stego
            # +vis.diff_z_same_stego=1
            # +vis.condscale=1 +vis.condscale_c.samples=8
            # exp.cond_scale=0 exp.ablate_scale=1  exp.ablate_scale_list=[0,1,2,3,0.5,1.5,2.5,3.5,4] data.test_fid_num=500 name=voc_condscale_lost
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=voc64 sg.params.condition_method=stegoclusterlayout condition.stegoclusterlayout.how=stego condition.stegoclusterlayout.layout_dim=21 dynamic=unetca_fast dynamic.params.cond_token_num=1 sg.params.cond_dim=21 dynamic.params.context_dim=32 data.params.batch_size=80 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=800 data.fid_every_n_epoch=80 name=v1.6.2_dino_vits16_stegoclusterlayout_voc64_unetca_fast_ep800 debug=0 resume_from='outputs/v1.6.2_dino_vits16_stegoclusterlayout_voc64_unetca_fast_ep800/19-09-2022/18-49-07/ckpts/last.ckpt' exp.cond_scale=0 exp.ablate_scale=1 exp.ablate_scale_list=[0,1,2,3,0.5,1.5,2.5,3.5,4] name=voc_condscale_stegoclusterlayout train=0 data.test_fid_num=500".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif True:  # VIS, cocostuff, stego
            # +vis.diff_z_same_stego=1
            # +vis.condscale=1 +vis.condscale_c.samples=8
            # exp.cond_scale=0 exp.ablate_scale=1  exp.ablate_scale_list=[0,1,2,3,0.5,1.5,2.5,3.5,4] data.test_fid_num=500 name=voc_condscale_lost
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cocostuff64 sg.params.condition_method=layout condition.layout.how=stego condition.layout.layout_dim=27 dynamic=unetca_fast dynamic.params.cond_token_num=0 dynamic.params.context_dim=32 data.params.batch_size=80 sg.params.cond_dim=0 sg.params.cond_drop_prob=0.1 sg.params.cond_scale=2 data.trainer.max_epochs=400 data.fid_every_n_epoch=40 name=v1.6.2_dino_vits16_stego_cocostuff64_unetca_fast_ep400_vis resume_from=outputs/v1.6.2_dino_vits16_stego_cocostuff64_unetca_fast_ep400/03-09-2022/21-59-52/ckpts/last.ckpt train=0 +cond_scale=0 exp.ablate_scale=1  exp.ablate_scale_list=[0,1,2,3,0.5,1.5,2.5,3.5,4]".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        print(cfg)
        run_without_decorator(cfg, run_unittest=True)


test_with_initialize()
