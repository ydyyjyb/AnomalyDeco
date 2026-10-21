import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='tile')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Dataset root. For mvtec, pass the directory containing category folders or mvtec_anomaly_detection.')
    parser.add_argument('--mvtec-root', type=str, default=None,
                        help='MVTecAD root. Overrides --data-root when --dataset mvtec.')
    parser.add_argument('--visa-root', type=str, default=None,
                        help='VisA root. Overrides --data-root when --dataset visa.')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--experiment-name", type=str, default="baseline")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--selected", type=int, default=0)
    parser.add_argument("--support-sampling", type=str, default="seed_file",
                        choices=["seed_file", "random", "first"],
                        help="How to choose few-shot normal support images.")
    parser.add_argument("--support-seed", type=int, default=None,
                        help="Seed for random support sampling. Defaults to --seed.")
    parser.add_argument("--prompt-variant", type=str, default="",
                        help="Name of the prompt ablation variant. Used for result grouping.")
    parser.add_argument("--use-generic-anomaly-prompt", type=str2bool, choices=[True, False], default=True,
                        help="Use dataset-level hand-crafted anomaly prompts.")
    parser.add_argument("--use-category-anomaly-prompt", type=str2bool, choices=[True, False], default=True,
                        help="Use category-specific hand-crafted anomaly prompts.")
    parser.add_argument("--use-learnable-anomaly-prompt", type=str2bool, choices=[True, False], default=True,
                        help="Use learnable abnormal context prompts.")
    parser.add_argument("--fusion-variant", type=str, default="",
                        help="Name of the branch/fusion ablation variant. Used for result grouping.")
    parser.add_argument("--use-text-branch", type=str2bool, choices=[True, False], default=True,
                        help="Use text-guided anomaly scores during evaluation.")
    parser.add_argument("--use-visual-memory", type=str2bool, choices=[True, False], default=True,
                        help="Use visual memory-bank anomaly scores during evaluation.")
    parser.add_argument("--fusion-type", type=str, default="harmonic",
                        choices=["none", "mean", "max", "harmonic", "weighted"],
                        help="How to fuse text and visual anomaly scores when both branches are enabled.")
    parser.add_argument("--fusion-alpha", type=float, default=0.5,
                        help="Text-branch weight for weighted fusion.")
    parser.add_argument("--model-variant", type=str, default="",
                        help="Name of the model/module ablation variant. Used for result grouping.")
    parser.add_argument("--prompt-module", type=str, default="kan",
                        choices=["none", "mlp", "transformer", "kan"],
                        help="Visual prompt projection module for KAN-PAM ablation.")
    parser.add_argument("--kan-variant", type=str, default="full",
                        help="Reserved name for KAN-PAM internal variant. Current safe default is full.")
    parser.add_argument("--dfa-variant", type=str, default="",
                        help="Name of the DFA ablation variant. Used for result grouping.")
    parser.add_argument("--use-dfa", type=str2bool, choices=[True, False], default=True,
                        help="Use distance-based feature anchor weighting in BCE loss.")
    parser.add_argument("--save-anchors", type=str2bool, choices=[True, False], default=False,
                        help="Save text anchor and prompt feature diagnostics.")
    parser.add_argument("--sensitivity-variant", type=str, default="",
                        help="Name of the hyper-parameter sensitivity variant. Used for result grouping.")
    parser.add_argument("--lambda-dfa", type=float, default=1.0,
                        help="Strength of DFA distance weighting. 0 makes BCE weights uniform.")
    parser.add_argument("--memory-variant", type=str, default="",
                        help="Name of the memory-layer ablation variant. Used for result grouping.")
    parser.add_argument("--memory-layers", type=str, default="layer3,layer8",
                        help="Comma-separated visual memory features: layer3,layer8,final.")
    parser.add_argument("--save-cases", type=str2bool, choices=[True, False], default=False,
                        help="Save per-sample scores for failure-case analysis.")
    parser.add_argument("--support-perturbation", type=str, default="none",
                        choices=["none", "brightness_low", "brightness_high", "noise", "blur", "rotate"],
                        help="Perturb only support-set training images for robustness analysis.")
    parser.add_argument("--robustness-variant", type=str, default="",
                        help="Name of the support perturbation variant. Used for result grouping.")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="Limit the number of test samples for smoke tests. Does not affect support samples.")
    parser.add_argument("--gpu-id", type=int, default="5")

    # method related parameters
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    # optimizer
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--Epoch", type=int, default=100)
    parser.add_argument("--lambda1", type=float, default=1e-6)
    parser.add_argument("--lambda2", type=float, default=1e-1)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=6)
    parser.add_argument("--n_ctx_ab", type=int, default=5)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)

    parser.add_argument('--k_shot', type=int, default=1)

    parser.add_argument("--input_size0",type=int, default=240)  # 输入图像的行数
    parser.add_argument("--prompt_depth", type=int, default=12, help="depth of prompt")

    args = parser.parse_args()

    return args
