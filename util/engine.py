import torch


@torch.inference_mode
def generate_image(model, B, label_B, cfg, seed, top_k=900, top_p=0.95):
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):  # using bfloat16 can be faster
        recon_B3HW = model.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=seed)
    return recon_B3HW