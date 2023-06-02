
from dataset.ds_utils.dataset_common_utils import ds_has_label_info
from diffusion_utils.util import clip_unnormalize_to_zero_to_255
from tqdm import tqdm
import wandb 

def attr_vis(pl_module, _dl, cluster_ids=None):
    if pl_module.hparams.condition_method == "attr" and pl_module.trainer.global_step==0:
        # assert is voc,coco
        assert not ds_has_label_info(_dl.dataset.dataset_name)
        ATTR_NUM_VIS = 16
        MAX_IMG_PER_ATTR = 16
        attr_dict = {k: list() for k in range(ATTR_NUM_VIS)}
        if cluster_ids is not None:  # vis all
            cluster_dict_4papervis = {k_id: list() for k_id in cluster_ids}
        for _, batch in tqdm(
            enumerate(_dl),
            total=len(_dl),
            desc=f"Visualization Only, KNN-attr",
        ):
            batch_attrs = batch["attr"].cpu().numpy()  # [B,multi-hot attr ]
            batch_image = clip_unnormalize_to_zero_to_255(batch["image"])
            for batch_id, multihot_attrs in enumerate(batch_attrs):
                multihot_attrs = multihot_attrs.nonzero()[0]
                for attr_id in multihot_attrs:
                    if (
                        attr_id < ATTR_NUM_VIS
                        and len(attr_dict[attr_id]) < MAX_IMG_PER_ATTR
                    ):
                        attr_dict[attr_id].append(batch_image[batch_id])
                    if cluster_ids is not None and attr_id in cluster_ids:
                        cluster_dict_4papervis[attr_id].append(batch_image[batch_id])

        if cluster_ids is None:  # vis all
            wandb_dict = dict()
            for i in range(ATTR_NUM_VIS):
                attr_dict[i] = [
                    wandb.Image(_img.float())
                    for _img in attr_dict[i]
                ]
                wandb_dict.update({f"attr_vis/attr{i}": attr_dict[i]})
            pl_module.logger.experiment.log(wandb_dict, commit=True)
        else:  # used for papervis
            return cluster_dict_4papervis


def prepare_image(pl_module, batch_data):
    attr_vis(pl_module, _dl=pl_module.trainer.datamodule.train_dataloader())
    return batch_data
