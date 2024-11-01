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

**Step2: Deploy onnx with python SDK**

```
(openmmlab) D:\mmopenlab>python .\mmdeploy\demo\python\image_classification.py cpu .\mmdeploy\work_dir\onnx\resnet\ .\mmpretrain\demo\demo.JPEG
loading mmdeploy_trt_net.dll ...
failed to load library mmdeploy_trt_net.dll
loading mmdeploy_ort_net.dll ...
[2024-10-30 17:40:46.295] [mmdeploy] [info] [model.cpp:35] [DirectoryModel] Load model: ".\mmdeploy\work_dir\onnx\resnet\"
58 0.3177702724933624
62 0.2017456740140915
65 0.12394838780164719
54 0.10745801031589508
49 0.10204800963401794
```

**problems on 2024/11/1**

Considering of mission to compare the diffrent ways to run the model, It's better to train one standard by tutorial.

 VS code suddenly can't work today, but spyder can still work.

 tutorial according to:[使用MMDeploy（预编译包）转换MMxx(MMDeploy支持库均可)pth权重到onnx，并使用C++ SDK加载onnx得到dll动态链接库，实现在windows平台中调用（linux也适用）](https://blog.csdn.net/weixin_43749999/article/details/130308470?spm=1001.2014.3001.5502)

 and[使用MMDeploy（预编译包）转换MMxx(MMDeploy支持库均可)pth权重到onnx，并使用python SDK进行部署验证](https://blog.csdn.net/weixin_43749999/article/details/130307058?spm=1001.2014.3001.5502)

to train model: 
```
cd mmdetection
python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py
```
