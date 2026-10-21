import numpy as np
import os
import torch.nn as nn
from PIL import Image

from datasets import *
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *
from utils.load_save_model import *
from utils.diagnostics import save_anchor_diagnostics
from MaPLe.model import MaPLe
from MaPLe.params_init import *

TASK = 'SEG'

def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        img_dir: str,
        check_path: str,
        train_data: DataLoader,
        anchor_dir: str = None,
        ):

    params = model.prompt_learner.parameters()
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    # tranparams = [name for name, param in model.named_parameters() if param.requires_grad]
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion_cross = nn.CrossEntropyLoss().to(device)

    train_losses = []
    low_count = 0
    best_result_dict = None
    best_score_maps = None
    best_gt_mask_list = None
    for epoch in range(args.Epoch):
        features1 = []
        features2 = []
        features3 = []
        start_time = time.time()
        # 以下训练
        model.train()
        for (data, mask, label, name, img_type) in train_data:
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)

            data = data.to(device)

            [normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned], shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = model.prompt_learner()

            optimizer.zero_grad()
            _, feature_map, feature1, feature2 = model.encode_image(data, shared_ctx, deep_compound_prompts_vision)

            features1.append(feature1)
            features2.append(feature2)
            features3.append(feature_map)

            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts, deep_compound_prompts_text)

            abnormal_text_features = model.encode_abnormal_text_embedding(
                abnormal_text_prompt_handle,
                abnormal_text_prompt_learned,
                deep_compound_prompts_text,
            )

            # compute v2t loss and triplet loss
            normal_text_features_anchor = normal_text_features.mean(dim=0).unsqueeze(0).half()
            normal_text_features_anchor = normal_text_features_anchor / normal_text_features_anchor.norm(dim=-1, keepdim=True)

            abnormal_text_features_anchor = abnormal_text_features.mean(dim=0).unsqueeze(0).half()
            abnormal_text_features_anchor = abnormal_text_features_anchor / abnormal_text_features_anchor.norm(dim=-1, keepdim=True)

            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features.half()

            l_pos = torch.einsum('nic,cj->nij', feature_map, normal_text_features_anchor.transpose(0, 1))   # 正样本
            l_pos_bce = torch.einsum('nic,cm->nim', feature_map, normal_text_features.transpose(0, 1))
            l_neg_v2t = torch.einsum('nic,cj->nij', feature_map, abnormal_text_features.transpose(0, 1))   # 负样本
            l_neg_bce = torch.einsum('nic,cm->nim', feature_map, abnormal_text_features_anchor.transpose(0, 1))

            if model.precision == 'fp16':
                logit_scale = model.custom_clip.logit_scale.half()
            else:
                logit_scale = model.custom_clip.logit_scale

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
            target_v2t_cross = torch.zeros([logits_v2t.shape[0], logits_v2t.shape[1]], dtype=torch.long).to(device)
            loss_v2t_cross = criterion_cross(logits_v2t.transpose(1, 2), target_v2t_cross)
            
            distance = torch.sum((abnormal_text_features_anchor - normal_text_features) ** 2, dim=-1, keepdim=True)   # 每个正常文本与异常文本锚点的距离
            if args.use_dfa:
                weight = torch.sqrt(distance)
                weight = weight.T
                weight = (weight - weight.min()) / torch.clamp(weight.max() - weight.min(), min=1e-8) * 2
                weight = 1.0 + args.lambda_dfa * (weight - 1.0)
                weight = torch.cat([weight, torch.tensor([[1]]).to(device)], dim=-1)   # 异常文本的权重值
            else:
                weight = torch.ones((1, normal_text_features.shape[0] + 1), device=device)

            criterion_bce = nn.BCEWithLogitsLoss(weight.detach()).to(device)
            logits_v2t_bce = torch.cat([l_pos_bce, l_neg_bce], dim=-1) * logit_scale
            target_v2t_bce = torch.zeros(logits_v2t_bce.shape, dtype=torch.float).to(device)
            target_v2t_bce[:, :, :normal_text_features.shape[0]] = 1
            loss_v2t_bce = criterion_bce(logits_v2t_bce, target_v2t_bce)
            
            loss = loss_v2t_cross + loss_v2t_bce

            # print(f'loss_v2t_cross: {loss_v2t_cross.item()}, loss_v2t_bce: {loss_v2t_bce.item()}, trip_loss: {trip_loss.item()}, loss_match_abnormal: {loss_match_abnormal.item()}, loss: {loss.item()}')

            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())
        scheduler.step()
        features1 = torch.cat(features1, dim=0)
        features2 = torch.cat(features2, dim=0)
        features3 = torch.cat(features3, dim=0)
        model.build_image_feature_gallery(features1, features2, features3)

        model.build_text_feature_gallery()

        # 以下是评估
        model.eval()
        score_maps_seg = []
        test_imgs = []
        gt_mask_list = []
        names = []
        case_rows = []

        for (data, mask, label, name, img_type) in dataloader:

            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0)

            for d, n, l, m in zip(data, name, label, mask):
                test_imgs += [denormalization(d.cpu().numpy())]
                l = l.numpy()
                m = m.numpy()
                m[m > 0] = 1

                names += [n]
                gt_mask_list += [m]

            data = data.to(device)
            score_map_seg, _ = model(data, 'seg', is_eval=True)
            score_maps_seg += score_map_seg
            for n, l, score_map in zip(name, label, score_map_seg):
                case_rows.append({'name': n, 'label': int(l), 'score': float(np.max(score_map))})

        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps_seg, gt_mask_list, resolution=(args.resolution, args.resolution))

        result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list, cal_pro=False)

        if best_result_dict is None:
            best_result_dict = result_dict
            best_score_maps = np.array(score_maps)
            best_gt_mask_list = list(gt_mask_list)
            save_check_point(model, check_path)
            if args.vis:
                plot_sample_cv2(names, test_imgs, {'MPAD': score_maps}, gt_mask_list, save_folder=img_dir)

        elif best_result_dict['p_roc'] < result_dict['p_roc']:
            best_result_dict.update(result_dict)
            best_score_maps = np.array(score_maps)
            best_gt_mask_list = list(gt_mask_list)
            save_check_point(model, check_path)
            low_count = 0
            if args.vis:
                plot_sample_cv2(names, test_imgs, {'MPAD': score_maps}, gt_mask_list, save_folder=img_dir)
        
        print(f"{args.k_shot}-shot {args.class_name} 第 {epoch + 1} 个 epoch 花了 {time.time() - start_time:.2f} 秒。\
              当前epoch Pixel AUROC 是 {result_dict['p_roc']:.4f}, 当前最佳Pixel AUROC是 {best_result_dict['p_roc']:.4f}。")

        if(best_result_dict['p_roc'] >= 99.99):
            print(f"Image AUROC 达到了 0.9999, 提前停止训练。")
            break
        elif(best_result_dict['p_roc'] > result_dict['p_roc']):
            low_count += 1
            if low_count >= 30:
                print(f"Image AUROC 连续30个epoch没有提升, 提前停止训练。")
                break

    # # Loss曲线
    # plt.figure()
    # plt.plot(train_losses, 'b-o', linewidth=2, label='Training Loss')
    # plt.scatter((best_epoch - 1), train_losses[best_epoch - 1], c='red', s=100, zorder=5,  # 标出最佳点
    #             label=f'Best Epoch ({best_epoch+1})')
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Loss', fontsize=12)
    # plt.title('Training Loss Curve', fontsize=14)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.ylim(0, 10)
    # plt.legend()

    # # 保存图表
    # plt.tight_layout()
    # save_path = os.path.join(os.path.dirname(img_dir), f"TrainingLoss_curves_epoch{args.Epoch}.png")
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()  # 释放内存

    if args.cal_pro and best_score_maps is not None and best_gt_mask_list is not None:
        pro_result = metric_cal_pix(best_score_maps, best_gt_mask_list, cal_pro=True)
        if 'aupro' in pro_result:
            best_result_dict['aupro'] = pro_result['aupro']

    if args.save_anchors and anchor_dir is not None:
        best_result_dict.update(save_anchor_diagnostics(model, args, anchor_dir))

    if args.save_cases:
        import pandas as pd
        os.makedirs(os.path.dirname(img_dir), exist_ok=True)
        pd.DataFrame(case_rows).to_csv(os.path.join(os.path.dirname(img_dir), 'cases_seg.csv'), index=False)
    
    return best_result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    kwargs['device'] = device

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)
    artifact_paths = get_run_artifact_paths(TASK, **kwargs)
    img_dir = artifact_paths['heatmap_dir']
    check_path = os.path.join(artifact_paths['checkpoint_dir'], f"{TASK}-Seed_{kwargs['seed']}-{kwargs['class_name']}-check_point.pt")
    save_config(kwargs, artifact_paths['config_path'])

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = MaPLe(args, **kwargs)
    kwargs['trainable_params_m'] = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    metrics = fit(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path,
                  train_data=train_dataloader, anchor_dir=artifact_paths['anchor_dir'])

    p_roc = round(metrics['p_roc'], 2)
    aupro = metrics.get('aupro')
    print('\n{:=^50}'.format(' Pixel AUROC Evaluation Result '))
    print(f"{'K-shot:':<25} {kwargs['k_shot']}")
    print(f"{'Dataset:':<25} {kwargs['dataset']}")
    print(f"{'Class:':<25} {kwargs['class_name']}")
    print(f"{'Pixel-AUROC:':<25} {p_roc} %")
    if aupro is not None:
        print(f"{'AUPRO:':<25} {round(aupro, 2)} %")
    print('='*50)

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)
    append_run_summary(metrics, kwargs, TASK, artifact_paths['summary_path'],
                       checkpoint_path=check_path,
                       heatmap_path=img_dir,
                       config_path=artifact_paths['config_path'])


if __name__ == '__main__':
    args = get_args()
    main(args)
