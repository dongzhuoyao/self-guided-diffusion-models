

def has_attention_map(feat):
    if feat in ['dino_vits16']:
        return True
    elif feat in ['dino_vits8', 'dino_vitb16', 'dino_xcit_m24_p8', 'dino_vitb8', 'dino_resnet50']:
        return False
    elif feat in ['simclr', 'mae_vitbase', 'mae_vitlarge']:
        return False
    elif feat.startswith('msn'):
        return False
    elif feat.startswith('timm_'):
        return False
    else:
        raise ValueError(f'feat {feat} not supported')
