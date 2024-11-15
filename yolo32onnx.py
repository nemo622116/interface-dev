from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

# 需要在转换pth到onnx时传入一张图片
img = './images/demo.jpg'

# 保存结果路径的文件夹
work_dir = './work_dir/yolo2onnx'

# 注意这里尽量使用mmdeploy原始文档中推荐的end2end.onnx名称，后续加载onnx时，避免一些错误出现
save_file = 'end2end.onnx'

# 使用mmdeploy源码仓库中的mmdet对应config文件（如果你使用的mmcls，那么就需要到mmcls下面找到合适的deploy config文件）
deploy_cfg = './mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py'

# 加载你训练好的模型的config配置文件（这里以“faster-rcnn_r50_fpn_2x_coco.py”举例）
model_cfg = './configs/yolo3/yolov3_d53_8xb8-ms-608-273e_coco.py'

# 加载使用上述模型配置文件得到的训练权重latest.pth
model_checkpoint = './checkpoints/yolo3/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'

# 设置device为cpu
device = 'cuda:0'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,  model_checkpoint, device)

# 2. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)
