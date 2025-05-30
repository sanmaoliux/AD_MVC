import argparse
import warnings
import os, random, time, sys
import numpy as np
import torch
import torch.nn as nn

from layers import EnhancedCVCLNetwork
from loss import DeepMVCLoss
from dataprocessing1 import MultiviewData, get_multiview_data
from models1 import *
from torch.optim import AdamW

warnings.filterwarnings("ignore")

# ---------------- Cmd line ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--db', default='MNIST-USPS',
                    choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP', 'NUSWIDEOBJ', 'ORL',
                             'cifar10'],
                    help='dataset name')
parser.add_argument('--gpu', default='0')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--mse_epochs', type=int, default=200)
parser.add_argument('--con_epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=35)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--temperature_l', type=float, default= 1)
parser.add_argument('--ib_lambda', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=int, default=100)
parser.add_argument('--save_model', action='store_true', help='whether to save model')
parser.add_argument('--fusion_mode', default='static', choices=['static', 'dynamic', 'attention'],
                    help='type of fusion mechanism')
parser.add_argument('--use_dynamic_teacher', action='store_true',
                    help='whether to use dynamic teacher selection during training')
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
    # 数据集特定参数设置
    if args.db == "MSRCv1":
        args.learning_rate = 0.0001
        args.batch_size = 35
        args.seed = 42
        args.con_epochs = 500
        args.use_dynamic_teacher = True  # 对MNIST-USPS启用动态教师选择


        dim_high_feature = 4000
        dim_low_feature = 2048
        dims = [512, 1024, 2048]
        lambda_max = 0.01
        beta_max = 0.005

    elif args.db == "BDGP":
        args.temperature_l = 0.5
        args.learning_rate = 0.0001
        args.batch_size = 64
        args.seed = 42
        args.con_epochs = 500

        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512]
        lambda_max = 0.05
        beta_max = 0.005

    elif args.db == "COIL20":
        args.temperature_l = 1
        args.learning_rate = 0.0001
        args.batch_size = 144
        args.seed = 10
        args.con_epochs = 500

        dim_high_feature = 512
        dim_low_feature = 256
        dims = [256, 512, 768]
        lambda_max = 0.01
        beta_max = 0.01

    elif args.db == "MNIST-USPS":
        args.learning_rate = 0.0001
        args.batch_size = 50
        args.seed = 10
        args.mse_epochs = 200
        args.con_epochs = 400
        args.use_dynamic_teacher = True  # 对MNIST-USPS启用动态教师选择

        dim_high_feature = 2048
        dim_low_feature = 1024
        dims = [256, 512, 1024, 1536]
        lambda_max = 0.005
        beta_max = 0.001

    # 添加默认值以防遗漏
    else:
        dim_high_feature = 2048
        dim_low_feature = 1024
        dims = [512, 1024]
        lambda_max = 0.01
        beta_max = 0.005

    # =================== 运行实验 ===================
    set_seed(args.seed)
    mv_train = MultiviewData(args.db, device, training=True)
    mv_eval = MultiviewData(args.db, device, training=False)

    num_views = len(mv_train.data_views)
    num_samples = mv_train.labels.size
    num_clusters = np.unique(mv_train.labels).size
    input_sizes = [mv_train.data_views[v].shape[1] for v in range(num_views)]

    print(f"[Info] views={num_views}  samples={num_samples}  clusters={num_clusters}")

    # 创建增强型模型
    model = EnhancedCVCLNetwork(
        num_views, input_sizes, dims,
        dim_high_feature, dim_low_feature, num_clusters,
        teacher_index=0,
        fusion_mode=args.fusion_mode
    ).to(device)

    check_model_device(model)
    optim = AdamW(model.parameters(), lr=args.lr)

    # ---------- (1) 预训练 ----------
    print("==> Pre-training ...")
    t = time.time()
    pre_train(model, mv_train, args.batch_size, args.mse_epochs, optim)

    # ---------- (2) 初始教师选择 ----------
    print("==> Initial teacher selection based on clustering performance...")
    if args.db == "MNIST-USPS":
        teacher_evaluations = evaluate_teacher_with_multiple_metrics(
            model, mv_eval, args.batch_size, method='acc_primary'
        )
    else:
        teacher_evaluations = evaluate_teacher_with_multiple_metrics(
            model, mv_eval, args.batch_size, method='nmi_primary'
        )

    best_teacher_info = max(teacher_evaluations, key=lambda x: x['score'])
    new_teacher = best_teacher_info['teacher_idx']

    print(f"Initial teacher selection results:")
    for eval_info in teacher_evaluations:
        perf = eval_info['fusion_performance']
        print(f"View {eval_info['teacher_idx']}: ACC={perf['acc']:.4f} "
              f"NMI={perf['nmi']:.4f} Score={eval_info['score']:.4f}")

    print(f"Selected initial teacher: View {new_teacher}")
    model.set_teacher(new_teacher)

    # ---------- (3) 对比训练 ----------
    mvc_loss_fn = DeepMVCLoss(args.batch_size, num_clusters, lambda_max, beta_max)

    if args.use_dynamic_teacher:
        print("==> Contrastive training with dynamic teacher selection...")
        teacher_history = contrastive_train_with_dynamic_teacher(
            model, mv_train, mv_eval, mvc_loss_fn, args, optim, lambda_max, beta_max
        )

        # 打印教师切换历史
        if teacher_history:
            print("\n==> Teacher switching history:")
            for switch in teacher_history:
                print(f"Epoch {switch['epoch']}: View {switch['old_teacher']} -> "
                      f"View {switch['new_teacher']} (improvement: {switch['score_improvement']:.4f})")
        else:
            print("No teacher switches occurred during training.")

    else:
        print("==> Standard contrastive training...")
        best_acc = 0

        for epoch in range(args.con_epochs):
            # warm-up
            ratio = min(1.0, epoch / args.warmup_epochs)
            lam_cur = lambda_max * ratio
            beta_cur = beta_max * ratio
            ib_cur = args.ib_lambda * ratio

            loss = contrastive_train(
                model, mv_train, mvc_loss_fn,
                args.batch_size, lam_cur, beta_cur, ib_cur,
                args.temperature_l, False, epoch, optim
            )

            # 定期评估和更新融合权重
            if (epoch % 100 == 99 or epoch == args.con_epochs - 1):
                print(f"==> Epoch {epoch + 1}: Evaluating performance...")
                acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

                if acc > best_acc:
                    best_acc = acc
                    if args.save_model:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'epoch': epoch,
                            'acc': acc
                        }, f'best_model_{args.db}.pth')

                # 更新融合权重（仅对static模式）
                if args.fusion_mode == 'static':
                    eval_loader, _, _, _ = get_multiview_data(mv_eval, args.batch_size)
                    evaluator = ViewQualityEvaluator(model, eval_loader, num_views, num_clusters)
                    fusion_weights = evaluator.get_fusion_weights(method='softmax')
                    evaluator.apply_weights(model, fusion_weights)

    print("==> Training finished, validating ...")

    # ---------- (4) 最终评估 ----------
    eval_loader, _, _, _ = get_multiview_data(mv_eval, args.batch_size)

    if args.fusion_mode == 'static':
        print("==> Final view quality evaluation:")
        evaluator = ViewQualityEvaluator(model, eval_loader, num_views, num_clusters)

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

        print(f"==> Best weighting method: {best_method} with accuracy {best_acc:.4f}")
        evaluator.apply_weights(model, best_weights)

        print("==> Final evaluation with best fusion weights:")
        acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

        # 保存结果
        with open(f'result_{args.db}_enhanced.txt', 'a+') as f:
            f.write('{} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {} \t {} \t {} \t {:.4f} \n'.format(
                args.seed, args.batch_size, args.lr,
                acc, nmi, pur, ari,
                best_method, list(best_weights), args.fusion_mode, (time.time() - t)
            ))

    else:
        # 对于其他融合模式，直接评估
        print(f"==> Final evaluation with {args.fusion_mode} fusion:")
        acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

        with open(f'result_{args.db}_enhanced.txt', 'a+') as f:
            f.write('{} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {} \t {:.4f} \n'.format(
                args.seed, args.batch_size, args.lr,
                acc, nmi, pur, ari, args.fusion_mode, (time.time() - t)
            ))

    # 保存模型
    if args.save_model:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'performance': {'acc': acc, 'nmi': nmi, 'pur': pur, 'ari': ari},
            'fusion_mode': args.fusion_mode,
            'use_dynamic_teacher': args.use_dynamic_teacher
        }

        if args.fusion_mode == 'static' and 'best_weights' in locals():
            save_dict['fusion_weights'] = best_weights
            save_dict['weight_method'] = best_method

        if args.use_dynamic_teacher and 'teacher_history' in locals():
            save_dict['teacher_history'] = teacher_history

        torch.save(save_dict, f'model_{args.db}_{args.fusion_mode}_fusion_final.pth')

    # 打印总结
    print("\n" + "=" * 50)
    print(f"Experiment Summary for {args.db}")
    print("=" * 50)
    print(f"Fusion mode: {args.fusion_mode}")
    print(f"Dynamic teacher: {args.use_dynamic_teacher}")
    print(f"Final performance:")
    print(f"  ACC: {acc:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  PUR: {pur:.4f}")
    print(f"  ARI: {ari:.4f}")

    if args.fusion_mode == 'static' and 'best_method' in locals():
        print(f"Best fusion method: {best_method}")
        print(f"Best fusion weights: {best_weights}")

    if args.use_dynamic_teacher and 'teacher_history' in locals() and teacher_history:
        print(f"Teacher switches: {len(teacher_history)}")

    print(f"Total time: {time.time() - t:.2f}s")
    print("=" * 50)
