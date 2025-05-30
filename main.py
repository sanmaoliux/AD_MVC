import argparse
import warnings
import os, random, time
import numpy as np
import torch

from layers1 import EnhancedCVCLNetwork
from loss import DeepMVCLoss
from dataprocessing1 import MultiviewData, get_multiview_data
from models1 import pre_train, contrastive_train, valid, evaluate_teacher_selection, ViewQualityEvaluator
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
parser.add_argument('--temperature_l', type=float, default=0.5)
parser.add_argument('--ib_lambda', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=int, default=100)
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


# ---------------- 数据集特定超参 ----------------
if __name__ == "__main__":
    if args.db == "MSRCv1":
        args.learning_rate = 0.0001
        args.batch_size = 35
        args.seed = 42

        # 超参数设置
        dim_high_feature = 4000
        dim_low_feature = 2048
        dims = [512, 1024, 2048]
        lambda_max = 0.01
        beta_max = 0.005

    elif args.db == "BDGP":
        args.learning_rate = 0.0001
        args.batch_size = 64
        args.seed = 42
        args.con_epochs = 2000

        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512]
        lambda_max = 0.05
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

    # 创建增强型模型，使用自适应融合
    model = EnhancedCVCLNetwork(
        num_views, input_sizes, dims,
        dim_high_feature, dim_low_feature, num_clusters,
        teacher_index=0,  # 初始教师视图
        fusion_mode=args.fusion_mode  # 使用指定的融合模式
    ).to(device)

    optim = AdamW(model.parameters(), lr=args.lr)

    # ---------- (1) 预训练 ----------
    print("==> Pre-training ...")
    pre_train(model, mv_train, args.batch_size, args.mse_epochs, optim)
    t = time.time()
    # ---------- (2) 选择教师视图 ----------
    losses = evaluate_teacher_selection(model, mv_eval, args.batch_size)
    new_teacher = int(np.argmin(losses))
    print("Reconstruction loss per view:", losses)
    print("Selected teacher =", new_teacher)
    model.set_teacher(new_teacher)

    # ---------- (3) 对比训练 ----------
    mvc_loss_fn = DeepMVCLoss(args.batch_size, num_clusters, lambda_max, beta_max)
    print("==> Contrastive training ...")

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

        # 每100个epoch评估一次视图质量并更新融合权重
        if (epoch % 100 == 99 or epoch == args.con_epochs - 1) and args.fusion_mode == 'static':
            print(f"==> Epoch {epoch + 1}: Evaluating view quality...")
            eval_loader, _, _, _ = get_multiview_data(mv_eval, args.batch_size)
            evaluator = ViewQualityEvaluator(model, eval_loader, num_views, num_clusters)

            # 使用softmax方法计算权重，进一步放大高性能视图的权重
            fusion_weights = evaluator.get_fusion_weights(method='softmax')
            evaluator.apply_weights(model, fusion_weights)

            # 验证当前性能
            print(f"==> Intermediate evaluation with updated weights:")
            _ = valid(model, mv_eval, args.batch_size)

    print("==> Training finished, validating ...")

    # 最终评估前进行视图质量评估和权重优化
    eval_loader, _, _, _ = get_multiview_data(mv_eval, args.batch_size)

    if args.fusion_mode == 'static':
        print("==> Final view quality evaluation:")
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
            f.write('{} \t {} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {} \t {} \t {:.4f} \t {} \n'.format(
                dim_high_feature, dim_low_feature, args.seed, args.batch_size, args.lr,
                acc, nmi, pur, ari,
                best_method, list(best_weights), args.fusion_mode, (time.time() - t)
            ))
    else:
        # 对于动态或注意力融合，直接评估
        print(f"==> Final evaluation with {args.fusion_mode} fusion:")
        acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

        # 保存结果
        with open(f'result_{args.db}_enhanced.txt', 'a+') as f:
            f.write('{} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f}  \t {} \n'.format(
                args.seed, args.batch_size, args.lr,
                acc, nmi, pur, ari, (time.time() - t), args.fusion_mode
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

        torch.save(save_dict, f'model_{args.db}_{args.fusion_mode}_fusion.pth')
