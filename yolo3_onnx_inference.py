from mmdeploy.apis import inference_model

# 加载使用git clone 下载的mmdeploy中相关的deploy config文件
deploy_cfg = './mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py'

# 加载模型权重对应的config文件
model_cfg = './configs/yolo3/yolov3_d53_8xb8-ms-608-273e_coco.py'

# 设置后端推理使用的onnx模型路径
backend_files = ['./work_dir/yolo2onnx/end2end.onnx']

# 设置后端推理所需的demo图片
img = ['./images/demo.jpg']

# 设置使用cpu设备
device = 'cpu'

# 调用mmdeploy中推理函数接口得到result结果
result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)

# 控制台打印得到的results结果
print("result: {}".format(result))
