#模型下载
#基于Transformer 架构,数据集 海量互联网图像（LVD-1689M）
from modelscope import snapshot_download
model_dir = snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m', cache_dir='./allmodels')
model_dir = snapshot_download('facebook/dinov3-vitb16-pretrain-lvd1689m', cache_dir='./allmodels')
model_dir = snapshot_download('facebook/dinov3-vits16-pretrain-lvd1689m', cache_dir='./allmodels')
model_dir = snapshot_download('facebook/dinov3-vits16plus-pretrain-lvd1689m', cache_dir='./allmodels')
model_dir = snapshot_download('facebook/dinov3-vith16plus-pretrain-lvd1689m', cache_dir='./allmodels')
model_dir = snapshot_download('facebook/dinov3-vit7b16-pretrain-lvd1689m', cache_dir='./allmodels')

#基于 ConvNeXt 架构，不是 Transformer,数据集 海量互联网图像（LVD-1689M）
model_dir = snapshot_download('facebook/dinov3-convnext-tiny-pretrain-lvd1689m', cache_dir='./allmodels-convnext')
model_dir = snapshot_download('facebook/dinov3-convnext-small-pretrain-lvd1689m', cache_dir='./allmodels-convnext')
model_dir = snapshot_download('facebook/dinov3-convnext-base-pretrain-lvd1689m', cache_dir='./allmodels-convnext')
model_dir = snapshot_download('facebook/dinov3-convnext-large-pretrain-lvd1689m', cache_dir='./allmodels-convnext')



#基于Transformer 架构,数据集 基于 卫星数据集（SAT-493M ）
model_dir = snapshot_download('facebook/dinov3-vit7b16-pretrain-sat493m', cache_dir='./allmodels-sat')
model_dir = snapshot_download('facebook/dinov3-vitl16-pretrain-sat493m', cache_dir='./allmodels-sat')

####*************  可用于图像匹配、分割、检测、深度估计等
####*************  核心能力：“无需微调，直接使用” 

