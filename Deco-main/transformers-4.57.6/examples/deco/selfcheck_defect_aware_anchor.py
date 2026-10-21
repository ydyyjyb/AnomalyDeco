#!/usr/bin/env python
import torch

from transformers.generation.utils import GenerationMixin


def build_fake_layer_logits(num_layers: int, vocab_size: int) -> dict[int, torch.Tensor]:
    layer_logits = {i: torch.zeros(vocab_size, dtype=torch.float32) for i in range(num_layers)}

    defect_ids = torch.tensor([7, 11, 13], dtype=torch.long)
    norm_ids = torch.tensor([2, 3, 5], dtype=torch.long)
    hedge_ids = torch.tensor([17, 19], dtype=torch.long)

    # Layer 4 should be the intended anomaly anchor under hedge penalty.
    layer_logits[4][defect_ids] = 8.0
    layer_logits[4][norm_ids] = 1.0

    # Layer 2 looks more normal.
    layer_logits[2][defect_ids] = 2.0
    layer_logits[2][norm_ids] = 7.0

    # Layer 5 has strongest defect evidence but also strongest hedge evidence.
    layer_logits[5][defect_ids] = 9.0
    layer_logits[5][norm_ids] = 1.0
    layer_logits[5][hedge_ids] = 10.0

    return layer_logits


def run_case(use_logsumexp: bool) -> None:
    layer_logits = build_fake_layer_logits(num_layers=6, vocab_size=128)
    def_ids = torch.tensor([7, 11, 13], dtype=torch.long)
    norm_ids = torch.tensor([2, 3, 5], dtype=torch.long)
    hedge_ids = torch.tensor([17, 19], dtype=torch.long)

    # Hedge disabled: layer 5 should win because defect evidence is strongest.
    anchor_no_hedge = GenerationMixin.select_anchor_layer_defect_aware(
        layer_logits_list=layer_logits,
        a=1,
        b=5,
        def_ids=def_ids,
        norm_ids=norm_ids,
        hedge_ids=hedge_ids,
        lambda_norm=1.0,
        eta_hedge=1.5,
        use_logsumexp=use_logsumexp,
        gamma=8.0,
        enable_hedge=False,
        deco_debug=True,
        decoding_step=0,
    )
    assert anchor_no_hedge == 5, f"Expected layer 5 without hedge, got {anchor_no_hedge}"

    # Hedge enabled: layer 5 is penalized, layer 4 should be selected.
    anchor_with_hedge = GenerationMixin.select_anchor_layer_defect_aware(
        layer_logits_list=layer_logits,
        a=1,
        b=5,
        def_ids=def_ids,
        norm_ids=norm_ids,
        hedge_ids=hedge_ids,
        lambda_norm=1.0,
        eta_hedge=1.5,
        use_logsumexp=use_logsumexp,
        gamma=8.0,
        enable_hedge=True,
        deco_debug=True,
        decoding_step=0,
    )
    assert anchor_with_hedge == 4, f"Expected layer 4 with hedge, got {anchor_with_hedge}"

    print(
        f"[OK] use_logsumexp={use_logsumexp} anchor_no_hedge={anchor_no_hedge} "
        f"anchor_with_hedge={anchor_with_hedge}"
    )


def main() -> None:
    run_case(use_logsumexp=False)
    run_case(use_logsumexp=True)
    print("Defect-aware anchor selection self-check passed.")


if __name__ == "__main__":
    main()
