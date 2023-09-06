import torch
from torchmetrics import JaccardIndex,Dice
od_Dice = Dice(num_classes=1,multiclass=False)

gt = torch.ones(2,8,dtype=torch.int)
preds1 = torch.ones(2,8,dtype=torch.int)
preds2 = torch.ones(2,8,dtype=torch.int)
preds1[:,:3] = 0
preds2[:,:5] = 0
print("使用dice()计算的")
print("dice1")
dice1 = od_Dice(gt,preds1)
print(dice1)
print("dice2")
dice2 = od_Dice(gt,preds2)
print(dice2)
print("==========")
print("直接使用dice.update计算的")
od_Dice.reset()
print("dice1")
od_Dice.update(gt,preds1)
print(od_Dice.compute())
print("dice2")
od_Dice.reset()
od_Dice.update(gt,preds2)
print(od_Dice.compute())