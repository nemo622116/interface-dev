from mmdeploy.apis import torch2onnx
from mmdeploy.backend.tensorrt.onnx2tensorrt import onnx2tensorrt
from mmdeploy.backend.sdk.export_info import export2SDK
import os

img = 'D:/mmopenlab/mmpretrain/demo/demo.JPEG'
work_dir = 'D:/mmopenlab/mmdeploy/work_dir/trt/resnet'
save_file = 'end2end.onnx'
deploy_cfg = 'D:/mmopenlab/mmdeploy/configs/mmpretrain/classification_tensorrt_static-224x224.py'
model_cfg = 'D:/mmopenlab/mmpretrain/configs/resnet/resnet18_8xb32_in1k.py'
model_checkpoint = 'D:/mmopenlab/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
device = 'cuda:0'

# 1. convert model to IR(onnx)
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. convert IR to tensorrt
onnx_model = os.path.join(work_dir, save_file)
save_file = 'end2end.engine'
model_id = 0
device = 'cuda:0'
onnx2tensorrt(work_dir, save_file, model_id, deploy_cfg, onnx_model, device)

# 3. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)