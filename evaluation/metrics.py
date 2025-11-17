import numpy as np
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
import copy

eps = 1e-12

class Metrics:
    def __init__(self, metrics_of_interest = ["mae", "acc", "fbeta", "iou", "sa_sor", "sor", "ap", "ar", "top1", "top2", "top3", "top4", "top5"]):
        self.registerMetrics(metrics_of_interest)
    
    def registerMetrics(self, metrics_of_interest):
        self.metrics_of_interest = [m for m in metrics_of_interest if m in dir(self)]
        print("Register metrics: {}\nNot register metrics: {}".format(
            self.metrics_of_interest, 
            [x for x in metrics_of_interest if x not in self.metrics_of_interest]
        ), flush=True)
        
    def from_config(self, cfg):
        self.registerMetrics(cfg.TEST.METRICS_OF_INTEREST)
    
    def mergeMap(self, lst):
        merge = copy.deepcopy(lst[0])
        for m in lst[1::]:
            merge = np.maximum(merge, m)
        return merge

    def toNumpy(self, lst):
        return [x if isinstance(x, type(np.zeros((2,2)))) else np.array(x.detach().cpu()) for x in lst]

    def process(self, preds, gts, thres=.5):
        """

        Args:
            preds: list of 0-1 map (numpy)
            gts: list of 0-1 map (numpy)
            thres: float 0-1

        Returns:

        """
        if len(gts) <= 0:
            print("warning GT has empty instances", flush=True)
            return {}
        if len(preds) <= 0:
            preds = [np.zeros_like(gts[0])]

        preds = self.toNumpy(preds if isinstance(preds, list) else [preds])
        gts = self.toNumpy(gts if isinstance(gts, list) else [gts])
        merge_pred = self.mergeMap(preds)
        merge_gt = self.mergeMap(gts)
        gt_ind, pd_ind, ious = self.matcher(preds, gts, thres=thres)
        results = {}
        for m in self.metrics_of_interest:
            results[m] = self.__getattribute__(m)(
                pred=merge_pred, 
                gt=merge_gt, 
                preds=preds, 
                gts=gts, 
                gt_ind=gt_ind,
                pd_ind=pd_ind,
                ious=ious,
                thres=thres, 
                beta2=0.3
            )
        return results

    def aggregate(self, results, **argw):
        ''' results: list of dict '''
        report = {}
        count = {}
        n = len(results)
        for m in self.metrics_of_interest:
            report[m] = 0.0
            count[m] = 0.0
            for i in range(n):
                v = results[i][m]
                if isinstance(v, type(None)):
                  pass
                elif np.isnan(v):
                    count[m] += 1.0
                    report[m] += 0.0
                else:
                    count[m] += 1.0
                    report[m] += v
            report[m] = round((report[m]+eps) / (count[m]+eps), 4)
            report[m+"_count"] = count[m]
        return report

    #####################-Numpy-Array-#########################
    ##############   Metrics Implementation       #############
    ##############   Coded By Huankang GUAN       #############
    ###########################################################

    # ===== 添加新的 r_MAE 指标 =====
    def r_mae(self, preds, gts, gt_ind, pd_ind, ious, thres=0.5, **argw):
        """
        计算预测排名和真实排名之间的平均绝对误差
        
        Args:
            preds: list of predicted masks (numpy arrays)
            gts: list of ground truth masks (numpy arrays)
            gt_ind: ground truth indices from matcher
            pd_ind: prediction indices from matcher
            ious: IoU scores from matcher
            thres: IoU threshold for matching
            
        Returns:
            r_mae: ranking MAE score
        """
        if len(gts) == 0 or len(preds) == 0:
            return np.nan
        
        # 获取图像尺寸
        H, W = gts[0].shape
        N = H * W  # 总像素数
        
        # 创建真实排名图 (ground truth ranking map)
        # 0 = 背景，1 = 最难(rank highest)，2 = 中等，3 = 最易(rank lowest)
        gt_rank_map = np.zeros((H, W), dtype=float)
        n_gts = len(gts)
        for i, gt_mask in enumerate(gts):
            # rank从1开始，i=0是rank 1(最显著)，i=n_gts-1是rank n_gts(最不显著)
            rank_value = n_gts - i  # 反转排名：最显著=最大值
            # 或者如果你的标注是：1=最难，使用 i+1
            # rank_value = i + 1
            gt_rank_map[gt_mask > thres] = rank_value
        
        # 创建预测排名图 (predicted ranking map)
        pred_rank_map = np.zeros((H, W), dtype=float)
        n_preds = len(preds)
        
        # 使用匹配结果构建预测排名图
        matched_pairs = {}
        for idx in range(len(gt_ind)):
            if ious[idx] > thres:
                gt_idx = gt_ind[idx]
                pd_idx = pd_ind[idx]
                matched_pairs[gt_idx] = pd_idx
        
        # 为匹配的预测分配排名
        for gt_idx, pred_mask in enumerate(preds):
            # pred_idx = 0 表示最显著的预测
            rank_value = n_preds - gt_idx  # 反转排名
            # 或者: rank_value = gt_idx + 1
            pred_rank_map[pred_mask > thres] = rank_value
        
        # 计算 MAE
        abs_diff = np.abs(gt_rank_map - pred_rank_map)
        r_mae_score = np.sum(abs_diff) / N
        
        return r_mae_score
    
    # ===== 可选：更简单的逐像素版本 =====
    def r_mae_simple(self, preds, gts, **argw):
        """
        简化版本的 r_MAE，直接比较排名图
        """
        if len(gts) == 0 or len(preds) == 0:
            return np.nan
        
        H, W = gts[0].shape
        N = H * W
        
        # 构建 GT 排名图
        gt_rank_map = np.zeros((H, W), dtype=float)
        for i, gt_mask in enumerate(gts):
            rank = len(gts) - i  # 最显著=最大rank
            gt_rank_map[gt_mask > 0.5] = rank
        
        # 构建预测排名图
        pred_rank_map = np.zeros((H, W), dtype=float)
        for i, pred_mask in enumerate(preds):
            rank = len(preds) - i
            pred_rank_map[pred_mask > 0.5] = rank
        
        # 计算 MAE
        r_mae_score = np.sum(np.abs(gt_rank_map - pred_rank_map)) / N
        
        return r_mae_score
    '''
    All inputs (pred and gt) are maps with values between 0.0 and 1.0
    The default threshold is 0.5 if not specific
    '''
    def check(self, pred, gt):
        assert pred.shape==gt.shape, "shape of pred-{} and gt-{} are not matched".format(pred.shape, gt.shape)
        assert pred.max()<=1.0 and pred.min()>=0.0, "max-{}/min-{} value of pred is not btw 0 and 1".format(pred.max(), pred.min())
        assert gt.max()<=1.0 and gt.min()>=0.0, "max-{}/min-{} value of gt is not btw 0 and 1".format(gt.max(), gt.min())

    def matcher(self, preds, gts, thres=.5, **argw):
        costs = []
        for i, g in enumerate(gts):
            for j, p in enumerate(preds):
                costs.append(1.0 - self.iou(p, g, thres))
        costs = np.array(costs).reshape(len(gts), len(preds))
        gt_ind, pd_ind = linear_sum_assignment(costs)
        ious = 1.0 - costs[gt_ind, pd_ind]
        return gt_ind, pd_ind, ious

    def iou(self, pred, gt, thres=.5, **argw):
        self.check(pred, gt)
        inter = np.logical_and(pred>thres, gt>thres).sum()
        union = np.logical_or(pred>thres, gt>thres).sum()
        return inter / (union + 1e-6)

    def mae(self, pred, gt, **argw):
        self.check(pred, gt)
        return np.mean(np.abs(pred.astype(float) - gt.astype(float)))

    def acc(self, pred, gt, thres=.5, **argw):
        self.check(pred, gt)
        p = pred > thres
        g = gt > thres
        return 1.0 - float(np.logical_xor(p, g).sum())/float(np.prod(p.shape))

    def fbeta(self, pred, gt, thres=.5, beta2=0.3, **argw):
        self.check(pred, gt)
        p = (pred > thres) * 1.0
        g = (gt > thres) * 1.0
        tp = (p * g).sum()
        fp = (p * (1.0-g)).sum()
        fn = ((1.0-p) * g).sum()
        pre = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        return ( (1.+beta2) * pre * rec) / ( beta2 * pre + rec + 1e-6 )

    def ap(self, preds, gts, gt_ind, pd_ind, ious, thres=0.5, **argw):
        n_hits = (ious > 0.5).sum()
        n_preds= len(preds)
        return (n_hits+eps) / (n_preds+eps)
    
    def ar(self, preds, gts, gt_ind, pd_ind, ious, thres=0.5, **argw):
        n_hits = (ious > 0.5).sum()
        n_gts = len(gts)
        return (n_hits+eps) / (n_gts+eps)

    def sor(self, preds, gts, thres=.5, **argw):
        gt_ranks = []
        pred_ranks = []
        n_gt = len(gts)
        n_pred = len(preds)
        for i in range(n_gt):
            for j in range(n_pred):
                if self.iou(preds[j], gts[i]) > thres:
                    gt_ranks.append(n_gt - i)
                    pred_ranks.append(n_pred - j)
                    break
        if len(gt_ranks) > 1:
            try:
                spr = stats.spearmanr(pred_ranks, gt_ranks).statistic
            except:
                spr = stats.spearmanr(pred_ranks, gt_ranks).correlation
            return (spr + 1.0)/2.0
        elif len(gt_ranks) == 1:
            return 1.0
        else:
            return np.nan
    
    def sa_sor(self, preds, gts, gt_ind, pd_ind, ious, thres=0.5, **argw):
        n = len(gts)
        m = len(preds)
        ind = np.where(ious > .5)[0]
        gt_ind = gt_ind[ind]
        pd_ind = pd_ind[ind]

        gt_ranks = (n-gt_ind).astype(int).tolist()
        pd_ranks = (m-pd_ind).astype(int).tolist()
        for i in range(1, n+1):
            if i not in gt_ranks:
                gt_ranks.append(i)
                pd_ranks.append(0)
        return np.corrcoef(gt_ranks, pd_ranks)[0, 1]

    def top1(self, preds, gts, thres=.5, **argw):
        k = 1
        if len(gts) >= k and len(preds) >= k:
            v = self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=0.5)
            return 1.0 if v >= thres else 0.0
        elif len(gts) >= k:
            return 0.0
        else:
            return None

    def top2(self, preds, gts, thres=.5, **argw):
        k = 2
        if len(gts) >= k and len(preds) >= k:
            v = self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=0.5)
            return 1.0 if v >= thres else 0.0
        elif len(gts) >= k:
            return 0.0
        else:
            return None

    def top3(self, preds, gts, thres=.5, **argw):
        k = 3
        if len(gts) >= k and len(preds) >= k:
            v = self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=0.5)
            return 1.0 if v >= thres else 0.0
        elif len(gts) >= k:
            return 0.0
        else:
            return None

    def top4(self, preds, gts, thres=.5, **argw):
        k = 4
        if len(gts) >= k and len(preds) >= k:
            v = self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=0.5)
            return 1.0 if v >= thres else 0.0
        elif len(gts) >= k:
            return 0.0
        else:
            return None

    def top5(self, preds, gts, thres=.5, **argw):
        k = 5
        if len(gts) >= k and len(preds) >= k:
            v = self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=0.5)
            return 1.0 if v >= thres else 0.0
        elif len(gts) >= k:
            return 0.0
        else:
            return None


if __name__=="__main__":
    import os, tqdm
    from PIL import Image

    def decompose(m, ignore=[0]):
        vals = np.unique(m)[::-1]
        rets = []
        for v in vals:
            if v in ignore: continue
            rets.append(np.where(m==v, 1.0, 0.0).astype(float))
        return rets

    metrics = Metrics()
    input_path = r"D:\SaliencyRanking\retrain_compared_results\IRSR\IRSR\prediction"
    output_path = r"D:\SaliencyRanking\dataset\irsr\Images\test\gt"
    names = [x for x in os.listdir(output_path) if x.endswith(".png")]
    results = []
    for name in tqdm.tqdm(names[0:100]):
        if os.path.exists(os.path.join(input_path, name)):
            preds = decompose(np.array(Image.open(os.path.join(input_path, name)).convert("L"), dtype=np.uint8))
        if len(preds)<=0:
            preds = [np.zeros((480, 640), dtype=float)]
        gts = decompose(np.array(Image.open(os.path.join(output_path, name)).convert("L"), dtype=np.uint8))
        results.append(metrics.process(preds, gts, thres=.5))
    report = metrics.aggregate(results)
    print(report)
