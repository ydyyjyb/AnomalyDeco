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

TASK = 'CLS'

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
    record_for_tsne = None
    
    for epoch in range(args.Epoch):
        features1 = []
        features2 = []
        features3 = []
        start_time = time.time()
        
        temp = []
        # ================= 训练阶段 =================
        model.train_mode()
        for (data, mask, label, name, img_type) in train_data:
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)

            # data = data[0:1, :, :, :].to(device)
            data = data.to(device)

            [normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned], shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = model.prompt_learner()

            optimizer.zero_grad()
            cls_feature, patch_tokens, feature1, feature2 = model.encode_image(data, shared_ctx, deep_compound_prompts_vision)

            features1.append(feature1)
            features2.append(feature2)
            features3.append(patch_tokens)

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

            # temp.append(abnormal_text_features_handle)
            # temp.append(abnormal_text_features_learned)

            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features.half()

            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_anchor.transpose(0, 1))   # 正样本, l_pos = logits positive
            l_pos_bce = torch.einsum('nc,cm->nm', cls_feature, normal_text_features.transpose(0, 1))
            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))   # 负样本
            l_neg_bce = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features_anchor.transpose(0, 1))

            if model.precision == 'fp16':
                logit_scale = model.custom_clip.logit_scale.half()
            else:
                logit_scale = model.custom_clip.logit_scale

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
            logits_v2t_bce = torch.cat([l_pos_bce, l_neg_bce], dim=-1) * logit_scale

            target_v2t_cross = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

            loss_v2t_cross = criterion_cross(logits_v2t, target_v2t_cross)
            
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
            target_v2t_bce = torch.zeros(logits_v2t_bce.shape, dtype=torch.float).to(device)
            target_v2t_bce[:, :normal_text_features.shape[0]] = 1

            loss_v2t_bce = criterion_bce(logits_v2t_bce, target_v2t_bce)

            loss = loss_v2t_cross + loss_v2t_bce

            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())
        scheduler.step()
        features1 = torch.cat(features1, dim=0)
        features2 = torch.cat(features2, dim=0)
        features3 = torch.cat(features3, dim=0)
        model.build_image_feature_gallery(features1, features2, features3)

        normal = model.build_text_feature_gallery()
        # temp.append(F.normalize(normal, dim=-1))

        # ================= 评估阶段 =================
        model.eval_mode()
        scores_img = []
        score_maps = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        names = []
        case_rows = []
        tsne_data = []

        for (data, mask, label, name, img_type) in dataloader:

            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0)

            for d, n, l, m, t in zip(data, name, label, mask, img_type):
                test_imgs += [denormalization(d.cpu().numpy())]
                l = l.numpy()
                m = m.numpy()
                m[m > 0] = 1

                names += [n]
                gt_list += [l]
                gt_mask_list += [m]

                # if t == 'good':
                #     tsne_data.append(d)

            data = data.to(device)

            score_img, score_map = model(data, 'cls', is_eval=True)
            score_maps += score_map
            scores_img += score_img
            for n, l, s in zip(name, label, score_img):
                case_rows.append({'name': n, 'label': int(l), 'score': float(s)})

        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
        result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))
        gt_array = np.array(gt_list)
        score_array = np.array(scores_img)
        normal_scores = score_array[gt_array == 0]
        abnormal_scores = score_array[gt_array == 1]
        if len(normal_scores) > 0 and len(abnormal_scores) > 0:
            result_dict['score_margin'] = float(abnormal_scores.mean() - normal_scores.mean())

        if best_result_dict is None:
            save_check_point(model, check_path)
            best_result_dict = result_dict
            best_epoch = epoch + 1

            # tsne_data = torch.stack(tsne_data, dim=0)
            # tsne_data = tsne_data.to(device)
            # cls_feature_, _, _, _ = model.encode_image(tsne_data)
            # temp.append(cls_feature_)
            # labels = np.concatenate([np.full(f.shape[0], i) for i, f in enumerate(temp)])
            # temp = torch.cat(temp, dim=0)
            # temp = temp.cpu().detach().numpy()
            # record_for_tsne = [temp, labels]

        elif best_result_dict['i_roc'] < result_dict['i_roc']:
            save_check_point(model, check_path)
            best_result_dict.update(result_dict)
            best_epoch = epoch + 1
            low_count = 0

            # tsne_data = torch.stack(tsne_data, dim=0)
            # tsne_data = tsne_data.to(device)
            # cls_feature_, _, _, _ = model.encode_image(tsne_data)
            # temp.append(cls_feature_)
            # labels = np.concatenate([np.full(f.shape[0], i) for i, f in enumerate(temp)])
            # temp = torch.cat(temp, dim=0)
            # temp = temp.cpu().detach().numpy()
            # record_for_tsne = [temp, labels]
        
        print(f"{args.k_shot}-shot {args.class_name} 第 {epoch + 1} 个 epoch 花了 {time.time() - start_time:.2f} 秒。\
              当前epoch Image AUROC 是 {result_dict['i_roc']:.4f}, 当前最佳 Image AUROC 是 {best_result_dict['i_roc']:.4f} 。")

        if(best_result_dict['i_roc'] >= 99.99):
            print(f"Image AUROC 达到了 0.9999, 提前停止训练。")
            break
        elif(best_result_dict['i_roc'] >= result_dict['i_roc']):
            low_count += 1
            if low_count >= 30:
                print(f"Image AUROC 连续30个epoch没有提升, 提前停止训练。")
                break

    # visualize_feature(record_for_tsne[0], record_for_tsne[1], \
    #                    ['Abnormal text feature handle', 'Abnormal text feature learned', 'Normal text feature', 'Normal visual feature'], \
    #                     os.path.join(os.path.dirname(img_dir), f"{args.class_name}_cls_tsne.png"), 2)
    
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
    
    if args.save_anchors and anchor_dir is not None:
        best_result_dict.update(save_anchor_diagnostics(model, args, anchor_dir))

    if args.save_cases:
        import pandas as pd
        os.makedirs(os.path.dirname(img_dir), exist_ok=True)
        pd.DataFrame(case_rows).to_csv(os.path.join(os.path.dirname(img_dir), 'cases_cls.csv'), index=False)

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

    i_roc = round(metrics['i_roc'], 2)
    i_ap = round(metrics.get('i_ap', 0.0), 2)
    print('\n{:=^50}'.format(' Image AUROC Evaluation Result '))
    print(f"{'K-shot:':<25} {kwargs['k_shot']}")
    print(f"{'Dataset:':<25} {kwargs['dataset']}")
    print(f"{'Class:':<25} {kwargs['class_name']}")
    print(f"{'Image-AUROC:':<25} {i_roc} %")
    print(f"{'Image-AP:':<25} {i_ap} %")
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
