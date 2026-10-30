# """
# Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
# """
   
# import torch.nn as nn
# from ..core import register


# __all__ = ['DEIM', ]
   
   
# @register()
# class DEIM(nn.Module):    
#     __inject__ = ['backbone', 'encoder', 'decoder', ]  
     
#     def __init__(self, \
#         backbone: nn.Module,
#         encoder: nn.Module,
#         decoder: nn.Module,  
#     ):
#         super().__init__()
#         self.backbone = backbone
#         self.decoder = decoder    
#         self.encoder = encoder     
 
#     def forward(self, x, targets=None):
#         x = self.backbone(x)
#         x = self.encoder(x)     
#         x = self.decoder(x, targets)

#         return x  

#     def deploy(self, ):
#         self.eval()
#         for m in self.modules(): 
#             if hasattr(m, 'convert_to_deploy'):
#                 m.convert_to_deploy()
#         return self    








# ... 原有导入 ...
import torch.nn as nn
from ..core import register
# 1. 导入 CASAB (确保路径正确)
from ..extre_module.custom_nn.attention.CASAB import CASAB 

__all__ = ['DEIM', ]

@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]
    
    def __init__(self, 
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        # 2. 增加配置参数 (可选，也可以在 yaml 中配置)
        use_casab: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
        # 3. 初始化 CASAB 模块
        self.use_casab = use_casab
        if self.use_casab:
            # 获取 backbone 输出的通道数
            # 假设 return_idx 为 [1, 2, 3]，对应 B0 的通道数为 [256, 512, 1024]
            out_channels = backbone._out_channels
            return_idx = backbone.return_idx
            
            # 为每个输出层创建一个 CASAB 模块
            self.casab_layers = nn.ModuleList([
                CASAB(out_channels[i]) for i in return_idx
            ])

    def forward(self, x, targets=None):
        # 4. Backbone 提取特征
        x = self.backbone(x) # 返回一个 list 的 tensors
        
        # 5. 在进入 Encoder 前应用 CASAB
        if self.use_casab:
            processed_x = []
            for i, feat in enumerate(x):
                # 对每一层特征应用对应的 CASAB
                processed_x.append(self.casab_layers[i](feat))
            x = processed_x
            
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules(): 
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self  
    # ... 其余代码 ...