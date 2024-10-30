# interface-dev
The purpose of the project is to invoke OpenMMLab-trained models in C++ models.

**General Idea**

1.Use mmdeploy to convert the mmopenlab model to the common format of oonx or TensorRT

2.Package the model as an SDK

3.Deploy the onnx model using C++

**Step 0: Install MMdeploy**
follow the [official guide of MMdeploy](https://github.com/open-mmlab/mmdeploy/blob/main/README_zh-CN.md)

**Step 1: model covert from mm to ONNX**

onnx_demo.py

```
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
```


