import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from segment.modules.semseg.nn import ASPP,SelfAttentionBlock,checkpoint
from abc import abstractmethod



class FeaturesMemoryV2(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels, use_hard_aggregate=False,
                 norm_cfg=None, act_cfg=None, align_corners=False):
        super(FeaturesMemoryV2, self).__init__()
        # set attributes
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.use_hard_aggregate = use_hard_aggregate
        # init memory_moudle,每个类别的均值都为0，标准差都为1，因此是一个 类别数k x 特征数D(2) 的矩阵
        self.memory = nn.Parameter(torch.cat([
            torch.zeros(num_classes, 1, dtype=torch.float), torch.ones(num_classes, 1, dtype=torch.float),
        ], dim=1), requires_grad=False)

        # define self_attention module
        self.self_attention = SelfAttentionBlock(
            key_in_channels=feats_channels,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # bottleneck used to fuse feats and selected_memory
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    '''forward'''
    def forward(self, feats, preds=None):
        batch_size, num_channels, h, w = feats.size()
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1)

        memory_means = self.memory.data[:, 0]#[0,0]
        memory_stds = self.memory.data[:, 1]#[1,1]
        memory = []
        # 根据标签个数生成相应个数memory
        for idx in range(self.num_classes):
            torch.manual_seed(idx)
            # 生成一个张量，其中值服从mean、std给定得条件，且形状为1 x self.feats_channels
            cls_memory = torch.normal(
                # torch.full():生成一个1 x self.feats_channels的张量，内容全部用memory_means[idx]填充
                mean=torch.full((1, self.feats_channels), memory_means[idx]),
                std=torch.full((1, self.feats_channels), memory_stds[idx])
            )
            # 每生成一个就append到memory中
            memory.append(cls_memory)
        #利用cat将memory中的全部拼接起来，此时memory形状就是  num_classes * feats_channels的张量
        memory = torch.cat(memory, dim=0).type_as(weight_cls)
        # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        selected_memory = torch.matmul(weight_cls, memory)
        # calculate selected_memory
        # --(B*H*W, C) --> (B, H, W, C)
        selected_memory = selected_memory.view(batch_size, h, w, num_channels)
        # --(B, H, W, C) --> (B, C, H, W)
        selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
        # --feed into the self attention module
        if hasattr(self, 'downsample_before_sa'):#默认为False
            feats_in, selected_memory_in = self.downsample_before_sa(feats), self.downsample_before_sa(selected_memory)
        else:
            feats_in, selected_memory_in = feats, selected_memory
        selected_memory = self.self_attention(feats_in, selected_memory_in)
        if hasattr(self, 'downsample_before_sa'):#默认为False
            selected_memory = F.interpolate(selected_memory, size=feats.size()[2:], mode='bilinear',
                                            align_corners=self.align_corners)
        '''
        memory_output包含两个部分
        1是feats也即最原始的features
        2是被选中的memory
        因此memory再与cwi的内容一起concatenate就满足了原论文的条件
        '''
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        return memory.data, memory_output

    '''update'''
    def update(self, features, segmentation, ignore_index=255, momentum_cfg=None, learning_rate=None):
        batch_size, num_channels, h, w = features.size()
        momentum = 0.5
        # if momentum_cfg['adjust_by_learning_rate']:
        if 1:
            # momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
            momentum = 0.6
        # use features to update memory_moudle
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        # TODO 这里会出现bug，segmentation不止0，1，还会出现负值
        # clsids = segmentation.unique()
        clsids = (0,1)
        for clsid in clsids:
            if clsid == ignore_index: continue
            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)
            # --extract the corresponding feats: (K, C),返回的是某一个语义标签tensor 形状为：个数 x 通道
            feats_cls = features[seg_cls == clsid]
            # --update memory_moudle，计算当前语义标签的mean、std
            feats_cls = feats_cls.mean(0)
            mean, std = feats_cls.mean(), feats_cls.std()
            # 注意，这里的memory是一个二维tensor  第一维的维度代表了每个语义标签种类，第二维代表了当前语义标签在数据集中的特征分布
            self.memory[clsid][0] = (1 - momentum) * self.memory[clsid][0].data + momentum * mean
            self.memory[clsid][1] = (1 - momentum) * self.memory[clsid][1].data + momentum * std
        # syn the memory_moudle
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)

# '''Memory_module'''
class Memory_module(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Memory_module, self).__init__()

        # 定义context_within_image_cfg的配置参数
        context_within_image_cfg = {
            'is_on':'',
            # 定义ASPP的配置参数
            'ASSP_CFG':{
                'in_channels': in_channels,
                'out_channels': out_channels,
                'dilations': [1, 12, 24, 36],
                'align_corners': '',
                'norm_cfg': '',
                'act_cfg': '',
            }
        }


        # 定义memory的配置参数
        MEMORY_CFG = {
            'num_classes': 2,
            'feats_channels': in_channels,
            # transform_channels用于selfattention
            'transform_channels': 256,
            'out_channels': out_channels,
            'use_hard_aggregate': '',
            'downsample_before_sa': '',
            'norm_cfg':'',
            'act_cfg':'',
            'align_corners':'',
        }

        # 定义bottleneck、decoder的配置参数
        head_cfg = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'feats_channels':in_channels,
            'num_classes': 2,
            'decoder':{
                'pr': {'in_channels': in_channels, 'out_channels': out_channels, 'dropout': 0.1},
                'cwi': {'in_channels': in_channels, 'out_channels': out_channels, 'dropout': 0.1},
                'cls': {'in_channels': in_channels, 'out_channels': out_channels, 'dropout': 0.1, 'kernel_size': 3, 'padding': 1},
            }
        }


        # 实例化一个Memory
        self.memory_module = FeaturesMemoryV2(MEMORY_CFG['num_classes'], MEMORY_CFG['feats_channels'],
                                              MEMORY_CFG['transform_channels'], MEMORY_CFG['out_channels'])

        # TODO 初始化一个bottleneck，
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(head_cfg['feats_channels']),
            nn.ReLU()
        )

        # build decoder,配置文件中给出了decoder的配置信息
        for key, value in head_cfg['decoder'].items():
            if key == 'cwi' and (not context_within_image_cfg['is_on']): continue
            setattr(self, f'decoder_{key}', nn.Sequential())
            decoder = getattr(self, f'decoder_{key}')
            decoder.add_module('conv1', nn.Conv2d(value['in_channels'], value['out_channels'],
                                                  kernel_size=value.get('kernel_size', 1), stride=1,
                                                  padding=value.get('padding', 0), bias=False))
            decoder.add_module('bn1', nn.BatchNorm2d(head_cfg['out_channels']))
            decoder.add_module('act1', nn.ReLU())
            decoder.add_module('dropout', nn.Dropout2d(value['dropout']))
            if key == 'cls':
                decoder.add_module('conv2',
                nn.Conv2d(value['out_channels'], value['out_channels'], kernel_size=1, stride=1,padding=0))
            else :
                decoder.add_module('conv2',
                nn.Conv2d(value['out_channels'], head_cfg['num_classes'], kernel_size=1, stride=1, padding=0))


    '''forward
        x:经过backbone处理的特征
        gt：groundtruth
    '''

    def forward(self, x_and_gt, emb):
        return checkpoint(
            self._forward, (x_and_gt, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self,x_and_gt,emb):
        x ,gt = x_and_gt
        # 根据输入的tensor_x获取当前图像的大小
        img_size = x.size(2), x.size(3)
        # 获得backbone的结果
        backbone_outputs = x
        pixel_representations = self.bottleneck(backbone_outputs)

        memory_gather_logits = self.decoder_pr(pixel_representations)

        memory_input = pixel_representations
        # 这里要求memory_input和memory_gather_logits的形状，除了第一维不一样外，其他的都要一样,第一维是batch_size
        assert memory_input.shape[2:] == memory_gather_logits.shape[2:]
        if (self.training):
            # 以下代码将不会受到梯度变换的影响
            with torch.no_grad():
                '''
                gt.unsqueeze(1):由于gt是灰度图像，因此在tensor中其实不存在通道维度吗，因此需要对其进行维度扩张,这样才能在torchvision中进行操作
                size=memory_gather_logits.shape[2:]：根据已有的形状进行变形
                mode='nearest'，差值方法
                [:, 0, :, :]：降维操作
                '''
                # gt = F.interpolate(gt.unsqueeze(1).type(torch.float64), size=memory_gather_logits.shape[2:], mode='nearest')[:, 0, :, :]
                # 如果gt不是3维（batch_size x h x w），则报错
                assert len(gt.shape) == 3, 'segmentation format error'
                # 新建一个和gt同类型的tensor,preds_gt的形状极有可能是4维的
                preds_gt = gt.new_zeros(memory_gather_logits.shape).type_as(memory_gather_logits)
                # 很关键：将GT的大小resize为当前feature的大小特征
                gt = F.interpolate(gt.unsqueeze(1).type(torch.float64), size=memory_gather_logits.shape[2:], mode='nearest')[:, 0, :, :]

                # 取出gt中符合要求的像素标签,返回值是一个bool
                num_class = 2
                valid_mask = (gt >= 0) & (gt < num_class)
                # valid_mask明明是一个bool，但为什么能用nonzero呢？  因为bool的本质就是0，1，1是True，所以返回均为True的索引，
                # 如果valid是二维的，则元组有两个，先行后列
                # 如果是三维，则返回元组有3个，分别是：batchsize、height、weight
                idxs = torch.nonzero(valid_mask, as_tuple=True)
                # idxs[0].numel()计算idex[0]中的元素个数是否大于0,这里的idex[0]是行
                if idxs[0].numel() > 0:
                    # gt[valid_mask]取出gt中被valid_mask标为True对应的值
                    '''
                    idxs[0]：通俗理解为batch_size
                    gt[valid_mask].long():返回gt索引在valid_mask中维True的值
                    虽然这里对preds_gt这个四维的tensor进行了赋值为1的操作，但其逻辑为
                    对每张gt图片，在逻辑上将其分为了class_num个通道，而根据前面的操作，最终生成一个 batch_size x class_num x h x w
                    class_num的通道下值为1的个数代表了该类语义标签在原gt中的个数
                    '''
                    preds_gt[idxs[0], gt[valid_mask].long(), idxs[1], idxs[2]] = 1
            #    使用memory模块
            stored_memory, memory_output = self.memory_module(memory_input, preds_gt.detach())

        else:
            stored_memory, memory_output = self.memory_module(memory_input, memory_gather_logits)

        #    进行时间嵌入
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        memory_output = memory_output + emb_out

        preds_cls = self.decoder_cls(memory_output)

        if self.training:
            with torch.no_grad():
                # 调用memory模块中的update
                self.memory_module.update(
                    features=F.interpolate(pixel_representations, size=img_size, mode='bilinear',
                                           ),
                    segmentation=gt,
                    learning_rate=0.5,
                )

        return preds_cls

# in_channels = 2048
# out_channels = 2048
# memory = Memory_module(in_channels,out_channels)
# print(memory)
num_classes = 2
feats_channels = 256
transform_channels = 256
out_channels = 256
feature_memory = FeaturesMemoryV2(num_classes,feats_channels,transform_channels,out_channels)
print(feature_memory)