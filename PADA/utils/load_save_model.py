import torch
import os
from clip import clip

def load_clip(args):
    backbone_name = args.backbone + '-' + args.pretrained_dataset
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    # model_path = os.path.join(os.path.expanduser("~/.cache/clip"), "vit_b_16_plus_240-laion400m_e32-699c4b84.pt")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device).eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": args.n_ctx}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    model.to(device)

    return model


def save_check_point(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)
