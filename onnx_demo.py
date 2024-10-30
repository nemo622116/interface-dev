from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'D:/mmopenlab/mmpretrain/demo/demo.JPEG'
work_dir = 'D:/mmopenlab/mmdeploy/work_dir/onnx/resnet'
save_file = 'end2end.onnx'
deploy_cfg = 'D:/mmopenlab/mmdeploy/configs/mmpretrain/classification_onnxruntime_dynamic.py'
model_cfg = 'D:/mmopenlab/mmpretrain/configs/resnet/resnet18_8xb32_in1k.py'
model_checkpoint = 'D:/mmopenlab/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
device = 'cuda:0'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)