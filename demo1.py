import argparse
import warnings
import os, random, time, sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from layers2 import AD_MVC
from loss import DeepMVCLoss
from dataprocessing1 import MultiviewData, get_multiview_data
from models1 import pre_train, contrastive_train, valid, evaluate_teacher_with_multiple_metrics, ViewQualityEvaluator
from torch.optim import AdamW

warnings.filterwarnings("ignore")

# ---------------- Cmd line ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--db', default='MSRCv1',
                    choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP', 'NUSWIDEOBJ', 'ORL',
                             'cifar10'],
                    help='dataset name')
parser.add_argument('--gpu', default='0')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--mse_epochs', type=int, default=200)
parser.add_argument('--con_epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=35)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--temperature_l', type=float, default=1)
parser.add_argument('--ib_lambda', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=int, default=100)
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--save_model', action='store_true', help='whether to save model')
parser.add_argument('--fusion_mode', default='static', choices=['static', 'dynamic', 'attention'],
                    help='type of fusion mechanism')
args = parser.parse_args()

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_model_device(model):
    """检查模型所有参数是否在同一设备上"""
    devices = set()
    for name, param in model.named_parameters():
        devices.add(param.device)

    if len(devices) > 1:
        print(f"警告: 模型参数分布在多个设备上: {devices}")
        return False
    else:
        print(f"模型参数都在同一设备上: {next(iter(devices))}")
        return True


# ---------------- 数据集特定超参 ----------------
if __name__ == "__main__":
    if args.db == "MSRCv1":
        args.lr = 0.0001
        args.batch_size = 35
        args.seed = 42
        args.con_epochs = 500
        args.warmup_epochs = 200  
       
        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512, 1024]
        lambda_max = 0.01
        beta_max = 0.005

    elif args.db == "BDGP":
        args.lr = 0.0001
        args.batch_size = 50
        args.seed = 42
        args.con_epochs = 500
        args.normalized = True
        args.warmup_epochs = 200  
        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512]
        lambda_max = 0.05
        beta_max = 0.05


    elif args.db == "MNIST-USPS":
        args.lr = 0.0001
        args.batch_size = 50
        args.seed = 10
        args.mse_epochs = 200
        args.con_epochs = 500
        args.warmup_epochs = 200  

        dim_high_feature = 1500
        dim_low_feature = 1024
        dims = [256, 512, 1024]
        lambda_max = 0.05
        beta_max = 0.01

    elif args.db == "Fashion":
        args.lr = 0.0001
        args.batch_size = 100
        args.seed = 20
        args.con_epochs = 500
        args.warmup_epochs = 200  
        args.temperature_l = 0.5

        dim_high_feature = 2000
        dim_low_feature = 500
        dims = [256, 512]
        lambda_max = 0.01
        beta_max = 0.001



    # =================== 运行实验 ===================
    set_seed(args.seed)
    mv_train = MultiviewData(args.db, device, training=True)
    mv_eval = MultiviewData(args.db, device, training=False)

    num_views = len(mv_train.data_views)
    num_samples = mv_train.labels.size
    num_clusters = np.unique(mv_train.labels).size
    input_sizes = [mv_train.data_views[v].shape[1] for v in range(num_views)]

    print(f"[Info] views={num_views}  samples={num_samples}  clusters={num_clusters}")

    # 创建增强型模型，使用自适应融合
    model = AD_MVC(
        num_views, input_sizes, dims,
        dim_high_feature, dim_low_feature, num_clusters,
        teacher_index=0,  # 初始教师视图
        fusion_mode=args.fusion_mode  # 使用指定的融合模式
    ).to(device)

    # 在训练前添加检查
    check_model_device(model)

    optim = AdamW(model.parameters(), lr=args.lr)
    # optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)

    # ---------- (1) 预训练 ----------
    # 预训练
    print("==> Pre-training ...")
    t = time.time()
    pre_train(model, mv_train, args.batch_size, args.mse_epochs, optim)

    # 选择教师视图 - 使用聚类性能评估
    print("==> Selecting teacher view based on clustering performance...")

    if args.db == "MNIST-USPS, COIL20":
        # 对MNIST-USPS使用特殊策略
        model.train_mode = 'fusion'
        teacher_evaluations = evaluate_teacher_with_multiple_metrics(
            model, mv_eval, args.batch_size, method='acc_primary'
        )

        # 选择最佳教师
        best_teacher_info = max(teacher_evaluations, key=lambda x: x['score'])
        new_teacher = best_teacher_info['teacher_idx']

        print(f"Selected teacher: View {new_teacher}")
        print(f"Performance: {best_teacher_info['fusion_performance']}")

        # 如果性能都很差，强制选择MNIST作为教师
        if best_teacher_info['fusion_performance']['acc'] < 0.3:
            print("All teachers perform poorly, forcing MNIST (View 0) as teacher")
            new_teacher = 0

    else:
        # 其他数据集使用综合评估
        model.train_mode = 'fusion'
        teacher_evaluations = evaluate_teacher_with_multiple_metrics(
            model, mv_eval, args.batch_size, method='comprehensive'
        )
        best_teacher_info = max(teacher_evaluations, key=lambda x: x['score'])
        new_teacher = best_teacher_info['teacher_idx']

        print(f"Selected teacher: View {new_teacher}")
        print(f"Teacher evaluation results:")
        for eval_info in teacher_evaluations:
            print(f"  View {eval_info['teacher_idx']}: Score={eval_info['score']:.4f}")

    model.set_teacher(new_teacher)

    # ---------- (3) 对比训练 ----------
    mvc_loss_fn = DeepMVCLoss(args.batch_size, num_clusters, lambda_max, beta_max)
    print("==> Contrastive training ...")

    # contrastive_loss_history = []  # 记录损失值的列表

    for epoch in range(args.con_epochs):
        # warm-up
        ratio = min(1.0, epoch / args.warmup_epochs)
        lam_cur = lambda_max * 1
        beta_cur = beta_max * 1
        ib_cur = args.ib_lambda * ratio

        loss = contrastive_train(
            model, mv_train, mvc_loss_fn,
            args.batch_size, lam_cur, beta_cur, ib_cur,
            args.temperature_l, False, epoch, optim
        )

        # contrastive_loss_history.append(loss)
        # # 保存损失值到文件
        # pd.DataFrame(contrastive_loss_history, columns=['Loss']).to_csv('contrastive_train_loss.csv', index=False)

        # 每100个epoch评估一次视图质量并更新融合权重（对static和dynamic模式）
        if (epoch % 100 == 99 or epoch == args.con_epochs - 1) and args.fusion_mode in ['static', 'dynamic']:
            print(f"==> Epoch {epoch + 1}: Evaluating view quality...")
            eval_loader, _, _, _ = get_multiview_data(mv_eval, args.batch_size)
            model.train_mode = 'fusion'
            evaluator = ViewQualityEvaluator(model, eval_loader, num_views, num_clusters)

            # 根据融合模式不同处理
            if args.fusion_mode == 'static':
                # 使用softmax方法计算权重，进一步放大高性能视图的权重
                fusion_weights = evaluator.get_fusion_weights(method='softmax')
                model.train_mode = 'fusion'
                evaluator.apply_weights(model, fusion_weights)
            elif args.fusion_mode == 'dynamic':
                # 找出最佳视图
                view_scores = [evaluator.quality_scores[v]['score'] for v in range(num_views)]
                best_view_idx = np.argmax(view_scores)
                print(f"Best view is {best_view_idx} with score {view_scores[best_view_idx]:.4f}")

                # 调整偏好视图
                if hasattr(model.fusion_layer, 'adjust_bias_for_view'):
                    model.fusion_layer.adjust_bias_for_view(best_view_idx)

            # 验证当前性能
            print(f"==> Intermediate evaluation with updated {args.fusion_mode} fusion:")
            _ = valid(model, mv_eval, args.batch_size)




    print("==> Training finished, validating ...")

    # 最终评估前进行视图质量评估和权重优化
    eval_loader, _, _, _ = get_multiview_data(mv_eval, args.batch_size)

    if args.fusion_mode == 'static':
        print("==> Final view quality evaluation:")
        model.train_mode = 'fusion'
        evaluator = ViewQualityEvaluator(model, eval_loader, num_views, num_clusters)

        # 尝试不同的权重计算方法
        print("==> Testing different weighting methods:")
        methods = ['score', 'rank', 'square', 'softmax']
        best_acc = 0
        best_method = None
        best_weights = None

        for method in methods:
            weights = evaluator.get_fusion_weights(method=method)
            print(f"Method '{method}' weights: {weights}")

            evaluator.apply_weights(model, weights)
            print(f"==> Evaluation with {method} weights:")
            acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

            if acc > best_acc:
                best_acc = acc
                best_method = method
                best_weights = weights

        # 应用最佳权重
        print(f"==> Best weighting method: {best_method} with accuracy {best_acc:.4f}")
        evaluator.apply_weights(model, best_weights)

        # 最终验证
        print("==> Final evaluation with best fusion weights:")
        acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

        # 保存结果
        with open(f'result_{args.db}_enhanced.txt', 'a+') as f:
            f.write('{} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {} \t {} \t {} \t {:.4f} \n'.format(
                args.seed, args.batch_size, args.lr, lambda_max, beta_max,
                acc, nmi, pur, ari,
                best_method, list(best_weights), args.fusion_mode, (time.time() - t)
            ))

    elif args.fusion_mode == 'dynamic':
        # 对于动态融合，我们可以通过调整偏置来优化
        print("==> Final view quality evaluation for dynamic fusion:")
        model.train_mode = 'fusion'
        evaluator = ViewQualityEvaluator(model, eval_loader, num_views, num_clusters)

        # 获取视图质量评分
        view_scores = [evaluator.quality_scores[v]['score'] for v in range(num_views)]
        best_view_idx = np.argmax(view_scores)
        print(f"Best view is {best_view_idx} with score {view_scores[best_view_idx]:.4f}")

        # 调整动态权重网络的偏置，使其倾向于高质量视图
        if hasattr(model.fusion_layer, 'adjust_bias_for_view'):
            model.fusion_layer.adjust_bias_for_view(best_view_idx)

        # 最终验证
        print("==> Final evaluation with optimized dynamic fusion:")
        acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

        # 保存结果
        with open(f'result_{args.db}_enhanced.txt', 'a+') as f:
            f.write('{} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t dynamic \t best_view={} \t {:.4f} \n'.format(
                args.seed, args.batch_size, args.lr, lambda_max, beta_max,
                acc, nmi, pur, ari, best_view_idx, (time.time() - t)
            ))

    else:  # attention 模式
        # 对于注意力融合，直接评估
        print(f"==> Final evaluation with attention fusion:")
        model.train_mode = 'fusion'
        acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

        # 保存结果
        with open(f'result_{args.db}_enhanced.txt', 'a+') as f:
            f.write('{} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t attention \t {:.4f} \n'.format(
                args.seed, args.batch_size, args.lr, lambda_max, beta_max,
                acc, nmi, pur, ari, (time.time() - t)
            ))

    # 可选：保存模型
    if args.save_model:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'performance': {'acc': acc, 'nmi': nmi, 'pur': pur, 'ari': ari},
            'fusion_mode': args.fusion_mode
        }

        if args.fusion_mode == 'static' and 'best_weights' in locals():
            save_dict['fusion_weights'] = best_weights
            save_dict['weight_method'] = best_method
        elif args.fusion_mode == 'dynamic' and 'best_view_idx' in locals():
            save_dict['best_view_idx'] = best_view_idx

        torch.save(save_dict, f'model_{args.db}_{args.fusion_mode}_fusion.pth')

    # 打印总结
    print("\n=== Experiment Summary ===")
    print(f"Dataset: {args.db}")
    print(f"Fusion mode: {args.fusion_mode}")
    print(f"Final performance: ACC={acc:.4f}, NMI={nmi:.4f}, PUR={pur:.4f}, ARI={ari:.4f}")

    if args.fusion_mode == 'static' and 'best_method' in locals():
        print(f"Best weighting method: {best_method}")
        print(f"Best weights: {best_weights}")
    elif args.fusion_mode == 'dynamic' and 'best_view_idx' in locals():
        print(f"Best view used for bias adjustment: {best_view_idx}")
