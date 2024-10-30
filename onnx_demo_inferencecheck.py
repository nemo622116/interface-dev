from mmdeploy.apis import inference_model

model_cfg = 'D:/mmopenlab/mmpretrain/configs/resnet/resnet18_8xb32_in1k.py'
deploy_cfg = 'D:/mmopenlab/mmdeploy/configs/mmpretrain/classification_onnxruntime_dynamic.py'
backend_files = ['work_dir/onnx/resnet/end2end.onnx']
img = 'D:/mmopenlab/mmpretrain/demo/demo.JPEG'
device = 'cuda:0'
result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)
print('result')