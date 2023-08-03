from torchmetrics import Metric
from torchmetrics.functional import intersection, union
import torch

class BoundaryIOU(Metric):
    def __init__(self, dilation_ratio=0.02):
        super().__init__()
        self.dilation_ratio = dilation_ratio
        self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        preds, target = self._input_format(preds, target)

        # 使用mask_to_boundary函数将preds和target转换为边界掩码
        preds_boundary = mask_to_boundary(preds, self.dilation_ratio)
        target_boundary = mask_to_boundary(target, self.dilation_ratio)

        # 使用torchmetrics.functional中的函数计算交集和并集
        self.intersection += intersection(preds_boundary, target_boundary)
        self.union += union(preds_boundary, target_boundary)

    def compute(self):
        return self.intersection.float() / self.union

# 示例用法
boundary_iou = BoundaryIOU(dilation_ratio=0.02)
# for batch in dataloader:
#     preds, target = model(batch)
#     boundary_iou.update(preds, target)

result = boundary_iou.compute()
print(result.item())  # 将结果转换为标量值
