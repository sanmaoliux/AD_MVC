import sys
import time
import torch
import numpy as np
import pandas as pd
from dataprocessing1 import get_multiview_data
from metrics import calculate_metrics
from torch.utils.tensorboard import SummaryWriter


class ViewQualityEvaluator:
    """评估各视图质量并调整融合权重"""

    def __init__(self, model, data_loader, num_views, num_clusters):
        self.model = model
        self.loader = data_loader
        self.num_views = num_views
        self.num_clusters = num_clusters

        # 评估视图质量
        self.quality_scores = self._evaluate_view_quality()

    def _evaluate_view_quality(self):
        """基于聚类指标评估视图质量"""
        self.model.eval()
        view_preds = [[] for _ in range(self.num_views)]
        labels_all = []

        with torch.no_grad():
            for sub_views, labels in self.loader:
                # 检查模型类型
                if hasattr(self.model, 'fusion_layer'):
                    lbps, _, _, _, _ = self.model(sub_views)
                else:
                    lbps, _, _, _ = self.model(sub_views)

                for v in range(self.num_views):
                    preds = torch.argmax(lbps[v], dim=1).cpu().numpy()
                    view_preds[v].extend(preds)

                labels_all.extend(labels)

        # 计算每个视图的聚类性能
        view_scores = {}
        for v in range(self.num_views):
            preds = np.array(view_preds[v])
            labels = np.array(labels_all)

            acc, nmi, pur, ari = calculate_metrics(labels, preds)
            view_scores[v] = {
                'acc': acc, 'nmi': nmi, 'pur': pur, 'ari': ari,
                'score': (acc + nmi + ari) / 3.0  # 综合评分
            }

            print(
                f"View {v} quality: ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} Score={view_scores[v]['score']:.4f}")

        return view_scores

    def get_fusion_weights(self, method='softmax'):
        """获取推荐的融合权重"""
        scores = np.array([self.quality_scores[v]['score']
                           for v in range(self.num_views)])

        if method == 'rank':
            # 使用排名作为权重基础
            ranks = self.num_views - np.argsort(np.argsort(scores))
            scores = ranks
        elif method == 'square':
            # 平方放大差异
            scores = scores ** 2
        elif method == 'softmax':
            # softmax放大差异
            temperature = 2.0  # 控制权重差异程度
            exp_scores = np.exp(scores / temperature)
            return exp_scores / exp_scores.sum()

        # 归一化
        return scores / scores.sum()

    def apply_weights(self, model, weights):
        """将权重应用到模型的融合层"""
        if hasattr(model, 'fusion_layer') and hasattr(model.fusion_layer, 'weights'):
            # 应用到静态权重
            with torch.no_grad():
                device = model.fusion_layer.weights.device
                new_weights = torch.tensor(weights, device=device)
                model.fusion_layer.weights.copy_(new_weights)
            print(f"Applied quality-based weights: {weights}")
            return True
        else:
            print("Model doesn't have compatible fusion layer with static weights")
            return False


# ---------------- 预训练 ----------------
def pre_train(network_model, mv_data, batch_size, epochs, optimizer, writer=None):
    t0 = time.time()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    mse = torch.nn.MSELoss()
    history = []

    # 检查是否为增强型模型
    enhanced_model = hasattr(network_model, 'fusion_layer')

    for epoch in range(epochs):
        total = 0.
        for sub_views, _ in loader:
            # 根据模型类型处理不同的返回值
            if enhanced_model:
                _, recons, _, _, _ = network_model(sub_views)
            else:
                _, recons, _, _ = network_model(sub_views)

            loss = sum(mse(sub_views[v], recons[v]) for v in range(num_views))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / num_samples
        history.append(avg)
        if writer:
            writer.add_scalar('pretrain/loss', avg, epoch)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[PreTrain] epoch {epoch:3d}  loss={avg:.6f}")

    print(f"Pre-training finished, time = {time.time() - t0:.2f}s")
    return history


# ---------------- 对比学习 + KL ----------------
def contrastive_train(model,
                      mv_data,
                      mvc_loss,
                      batch_size,
                      lambda_c,
                      beta_e,
                      ib_lambda,
                      temperature_l,
                      normalized,
                      epoch,
                      optimizer,
                      writer=None):
    model.train()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    mse = torch.nn.MSELoss()
    total = 0.

    # 检查是否为增强型模型
    enhanced_model = hasattr(model, 'fusion_layer')

    for bid, (sub_views, _) in enumerate(loader):
        # 前向传播 - 处理不同模型输出
        if enhanced_model:
            lbps, recons, _, kl_total, fused_prob = model(sub_views)
        else:
            lbps, recons, _, kl_total = model(sub_views)

        loss_terms = []

        # 视图间对比 + 熵
        for i in range(num_views):
            for j in range(i + 1, num_views):
                loss_terms.append(lambda_c * mvc_loss.forward_label(lbps[i], lbps[j],
                                                                    temperature_l, normalized))
                loss_terms.append(beta_e * mvc_loss.forward_prob(lbps[i], lbps[j]))
        # 重构损失
        for i in range(num_views):
            loss_terms.append(mse(sub_views[i], recons[i]))

        # KL损失
        loss_terms.append(ib_lambda * kl_total)

        # 如果是增强型模型，增加融合标签和各视图标签之间的一致性损失
        # if enhanced_model:
        #     consistency_loss = 0.0
        #     for i in range(num_views):
        #         consistency_loss += torch.nn.functional.kl_div(
        #             torch.log(lbps[i] + 1e-8),  # log probs
        #             fused_prob,  # target probs
        #             reduction='batchmean')
        #     loss_terms.append(0.5 * consistency_loss / num_views)
        # 在 contrastive_train 函数中，替换现有的一致性损失部分
        if enhanced_model:
            # 改进的一致性损失：只向高质量视图学习
            consistency_loss = 0.0

            # 动态计算各视图的置信度
            view_confidences = []
            for i in range(num_views):
                # 使用预测的最大概率作为置信度指标
                max_probs = torch.max(lbps[i], dim=1)[0]
                confidence = torch.mean(max_probs)
                view_confidences.append(confidence)

            # 只有当视图置信度高于阈值时才参与一致性约束
            confidence_threshold = 0.7  # 可调节
            high_quality_views = []

            for i, confidence in enumerate(view_confidences):
                if confidence > confidence_threshold:
                    high_quality_views.append(i)

            # 如果有高质量视图，让融合结果向它们学习
            if high_quality_views:
                for i in high_quality_views:
                    weight = view_confidences[i]  # 使用置信度作为权重
                    consistency_loss += weight * torch.nn.functional.kl_div(
                        torch.log(fused_prob + 1e-8),
                        lbps[i],
                        reduction='batchmean'
                    )
                consistency_loss = 0.1 * consistency_loss / len(high_quality_views)  # 降低一致性损失权重
                loss_terms.append(consistency_loss)
            # 如果没有高质量视图，则不添加一致性损失

        loss = sum(loss_terms)
        if torch.isnan(loss) or torch.isinf(loss):
            print("[Fatal] loss NaN/Inf, abort.")
            torch.save(model.state_dict(), 'debug_nan.pth')
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    avg = total / num_samples
    if writer and epoch % 1 == 0:
        writer.add_scalar('train/loss', avg, epoch)
    if epoch % 10 == 0:
        print(f"[Contra] epoch {epoch:4d}  loss={avg:.6f}")

    return avg


# ---------------- Inference / 评估 ----------------
@torch.no_grad()
def inference(model, mv_data, batch_size):
    """支持增强型模型的推理函数"""
    model.eval()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    soft_all, labels_all = [], []
    preds_each = [[] for _ in range(num_views)]

    # 检查是否为增强型模型
    enhanced_model = hasattr(model, 'fusion_layer')

    for sub_views, lbl in loader:
        if enhanced_model:
            # 增强型模型
            lbps, _, _, _, fused_prob = model(sub_views)
            soft = fused_prob  # 使用自适应融合结果
        else:
            # 标准模型
            lbps, _, _, _ = model(sub_views)
            soft = sum(lbps) / num_views  # 简单平均

        # 收集每个视图的预测
        for v in range(num_views):
            preds_each[v].extend(torch.argmax(lbps[v], dim=1).cpu().numpy())

        soft_all.extend(soft.cpu().numpy())
        labels_all.extend(lbl)

    soft_all = np.array(soft_all)
    labels_all = np.array(labels_all)
    pred_final = np.argmax(soft_all, axis=1)
    preds_each = [np.array(p) for p in preds_each]
    return pred_final, preds_each, labels_all


def valid(model, mv_data, batch_size):
    """验证聚类效果"""
    pred_final, preds_each, labels = inference(model, mv_data, batch_size)
    num_views = len(preds_each)

    # 各视图性能评估
    for v in range(num_views):
        acc, nmi, pur, ari = calculate_metrics(labels, preds_each[v])
        print(f"View{v + 1}  ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f}")

    # 融合结果评估
    acc, nmi, pur, ari = calculate_metrics(labels, pred_final)
    print(f"[Fusion] ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f}")

    return acc, nmi, pur, ari


def evaluate_teacher_selection(model, mv_data, batch_size):
    """返回各视图平均重构误差"""
    model.eval()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    mse = torch.nn.MSELoss(reduction='sum')
    view_loss = [0.0] * num_views

    # 检查是否为增强型模型
    enhanced_model = hasattr(model, 'fusion_layer')

    with torch.no_grad():
        for sub_views, _ in loader:
            # 根据模型类型处理不同的返回值
            if enhanced_model:
                _, recons, _, _, _ = model(sub_views)
            else:
                _, recons, _, _ = model(sub_views)

            for v in range(num_views):
                view_loss[v] += mse(sub_views[v], recons[v]).item()

    return [l / num_samples for l in view_loss]
#
# def evaluate_teacher_selection_by_clustering(model, mv_data, batch_size):
#     """
#     基于聚类性能评估各视图作为教师的效果
#     返回各视图的聚类性能分数
#     """
#     from metrics import calculate_metrics
#
#     model.eval()
#     num_views = len(mv_data.data_views)
#     view_scores = []
#
#     print("==> Evaluating teacher candidates based on clustering performance...")
#
#     with torch.no_grad():
#         for teacher_candidate in range(num_views):
#             # 临时设置当前视图为教师
#             original_teacher = model.teacher_index
#             model.set_teacher(teacher_candidate)
#
#             # 获取预测结果
#             pred_final, preds_each, labels = inference(model, mv_data, batch_size)
#
#             # 计算聚类性能指标
#             acc, nmi, pur, ari = calculate_metrics(labels, pred_final)
#
#             # 综合评分：可以调整权重
#             score = 0.4 * acc + 0.3 * nmi + 0.2 * ari + 0.1 * pur
#             view_scores.append(score)
#
#             print(f"Teacher candidate {teacher_candidate}: "
#                   f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} "
#                   f"Score={score:.4f}")
#
#             # 恢复原始教师
#             model.set_teacher(original_teacher)
#
#     return view_scores
#
#
# def evaluate_teacher_with_multiple_metrics(model, mv_data, batch_size, method='comprehensive'):
#     """
#     使用多种策略评估教师视图
#
#     Args:
#         method: 'comprehensive' - 综合多个指标
#                 'acc_primary' - 以ACC为主
#                 'nmi_primary' - 以NMI为主
#                 'reconstruction' - 传统重构误差方法
#     """
#     model.eval()
#     num_views = len(mv_data.data_views)
#
#     if method == 'reconstruction':
#         # 原有的重构误差方法
#         return evaluate_teacher_selection(model, mv_data, batch_size)
#
#     teacher_evaluations = []
#
#     with torch.no_grad():
#         for teacher_candidate in range(num_views):
#             original_teacher = model.teacher_index
#             model.set_teacher(teacher_candidate)
#
#             # 获取各视图的预测
#             pred_final, preds_each, labels = inference(model, mv_data, batch_size)
#
#             # 计算融合结果的性能
#             acc_fusion, nmi_fusion, pur_fusion, ari_fusion = calculate_metrics(labels, pred_final)
#
#             # 计算各视图的平均性能
#             view_performances = []
#             for v in range(num_views):
#                 acc_v, nmi_v, pur_v, ari_v = calculate_metrics(labels, preds_each[v])
#                 view_performances.append({
#                     'acc': acc_v, 'nmi': nmi_v, 'pur': pur_v, 'ari': ari_v
#                 })
#
#             # 不同的评分策略
#             if method == 'comprehensive':
#                 # 综合评分：融合结果 + 各视图平均性能
#                 avg_acc = np.mean([v['acc'] for v in view_performances])
#                 avg_nmi = np.mean([v['nmi'] for v in view_performances])
#                 avg_ari = np.mean([v['ari'] for v in view_performances])
#
#                 score = (0.5 * (0.4 * acc_fusion + 0.3 * nmi_fusion + 0.3 * ari_fusion) +
#                          0.5 * (0.4 * avg_acc + 0.3 * avg_nmi + 0.3 * avg_ari))
#
#             elif method == 'acc_primary':
#                 # 以ACC为主要指标
#                 score = 0.7 * acc_fusion + 0.3 * np.mean([v['acc'] for v in view_performances])
#
#             elif method == 'nmi_primary':
#                 # 以NMI为主要指标
#                 score = 0.7 * nmi_fusion + 0.3 * np.mean([v['nmi'] for v in view_performances])
#
#             teacher_evaluations.append({
#                 'teacher_idx': teacher_candidate,
#                 'score': score,
#                 'fusion_performance': {
#                     'acc': acc_fusion, 'nmi': nmi_fusion,
#                     'pur': pur_fusion, 'ari': ari_fusion
#                 },
#                 'view_performances': view_performances
#             })
#
#             print(f"Teacher candidate {teacher_candidate}: "
#                   f"Fusion(ACC={acc_fusion:.4f}, NMI={nmi_fusion:.4f}, ARI={ari_fusion:.4f}) "
#                   f"Score={score:.4f}")
#
#             # 恢复原始教师
#             model.set_teacher(original_teacher)
#
#     return teacher_evaluations


def evaluate_teacher_with_multiple_metrics(model, mv_data, batch_size, method='comprehensive'):
    """
    使用多种策略评估教师视图

    Args:
        method: 'comprehensive' - 综合多个指标
                'acc_primary' - 以ACC为主
                'nmi_primary' - 以NMI为主
                'reconstruction' - 传统重构误差方法
    """
    model.eval()
    num_views = len(mv_data.data_views)

    if method == 'reconstruction':
        # 原有的重构误差方法
        return evaluate_teacher_selection(model, mv_data, batch_size)

    teacher_evaluations = []

    with torch.no_grad():
        for teacher_candidate in range(num_views):
            original_teacher = model.teacher_index
            model.set_teacher(teacher_candidate)

            # 获取各视图的预测
            pred_final, preds_each, labels = inference(model, mv_data, batch_size)

            # 计算融合结果的性能
            acc_fusion, nmi_fusion, pur_fusion, ari_fusion = calculate_metrics(labels, pred_final)

            # 计算各视图的平均性能
            view_performances = []
            for v in range(num_views):
                acc_v, nmi_v, pur_v, ari_v = calculate_metrics(labels, preds_each[v])
                view_performances.append({
                    'acc': acc_v, 'nmi': nmi_v, 'pur': pur_v, 'ari': ari_v
                })

            # 不同的评分策略
            if method == 'comprehensive':
                # 综合评分：融合结果 + 各视图平均性能
                avg_acc = np.mean([v['acc'] for v in view_performances])
                avg_nmi = np.mean([v['nmi'] for v in view_performances])
                avg_ari = np.mean([v['ari'] for v in view_performances])

                score = (0.5 * (0.4 * acc_fusion + 0.3 * nmi_fusion + 0.3 * ari_fusion) +
                         0.5 * (0.4 * avg_acc + 0.3 * avg_nmi + 0.3 * avg_ari))

            elif method == 'acc_primary':
                # 以ACC为主要指标
                score = 0.7 * acc_fusion + 0.3 * np.mean([v['acc'] for v in view_performances])

            elif method == 'nmi_primary':
                # 以NMI为主要指标
                score = 0.7 * nmi_fusion + 0.3 * np.mean([v['nmi'] for v in view_performances])

            teacher_evaluations.append({
                'teacher_idx': teacher_candidate,
                'score': score,
                'fusion_performance': {
                    'acc': acc_fusion, 'nmi': nmi_fusion,
                    'pur': pur_fusion, 'ari': ari_fusion
                },
                'view_performances': view_performances
            })

            # 恢复原始教师
            model.set_teacher(original_teacher)

    return teacher_evaluations


def dynamic_teacher_selection(model, mv_eval, batch_size, current_epoch,
                              teacher_history, min_improvement=0.01):
    """
    动态教师选择：在训练过程中定期重新评估教师
    """
    if current_epoch % 100 != 0 or current_epoch == 0:
        return model.teacher_index, teacher_history

    print(f"==> Epoch {current_epoch}: Re-evaluating teacher selection...")

    teacher_evaluations = evaluate_teacher_with_multiple_metrics(
        model, mv_eval, batch_size, method='comprehensive'
    )

    best_teacher_info = max(teacher_evaluations, key=lambda x: x['score'])
    best_teacher_idx = best_teacher_info['teacher_idx']
    best_score = best_teacher_info['score']

    current_teacher = model.teacher_index
    current_score = teacher_evaluations[current_teacher]['score']

    # 只有当新教师显著优于当前教师时才切换
    if best_teacher_idx != current_teacher and (best_score - current_score) > min_improvement:
        print(f"Switching teacher from View {current_teacher} to View {best_teacher_idx}")
        print(f"Score improvement: {current_score:.4f} -> {best_score:.4f}")
        model.set_teacher(best_teacher_idx)
        teacher_history.append({
            'epoch': current_epoch,
            'old_teacher': current_teacher,
            'new_teacher': best_teacher_idx,
            'score_improvement': best_score - current_score
        })
    else:
        print(f"Keeping current teacher View {current_teacher} (score: {current_score:.4f})")

    return model.teacher_index, teacher_history


def contrastive_train_with_dynamic_teacher(model, mv_train, mv_eval, mvc_loss_fn,
                                           args, optim, lambda_max, beta_max):
    """
    带动态教师选择的对比训练函数
    """
    teacher_history = []
    best_acc = 0

    for epoch in range(args.con_epochs):
        # 动态教师选择（每100轮评估一次）
        current_teacher, teacher_history = dynamic_teacher_selection(
            model, mv_eval, args.batch_size, epoch, teacher_history
        )

        # warm-up策略
        ratio = min(1.0, epoch / args.warmup_epochs)
        lam_cur = lambda_max * ratio
        beta_cur = beta_max * ratio
        ib_cur = args.ib_lambda * ratio

        # 执行一轮对比训练
        loss = contrastive_train(
            model, mv_train, mvc_loss_fn,
            args.batch_size, lam_cur, beta_cur, ib_cur,
            args.temperature_l, False, epoch, optim
        )

        # 定期评估性能
        if epoch % 100 == 99 or epoch == args.con_epochs - 1:
            print(f"\n==> Epoch {epoch + 1}: Evaluating performance...")
            acc, nmi, pur, ari = valid(model, mv_eval, args.batch_size)

            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                if args.save_model:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'acc': acc,
                        'teacher_history': teacher_history,
                        'current_teacher': current_teacher
                    }, f'best_model_{args.db}_dynamic_teacher.pth')

            # 如果使用静态融合，更新融合权重
            if args.fusion_mode == 'static':
                eval_loader, _, _, _ = get_multiview_data(mv_eval, args.batch_size)
                evaluator = ViewQualityEvaluator(model, eval_loader,
                                                 len(mv_train.data_views),
                                                 len(np.unique(mv_train.labels)))
                fusion_weights = evaluator.get_fusion_weights(method='softmax')
                evaluator.apply_weights(model, fusion_weights)

    return teacher_history



