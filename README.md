[English](README.md)

![⚡️Yolov7paddleOcr](https://user-images.githubusercontent.com/31974251/185771818-5d4423cd-c94c-4a49-9894-bc7a8d1c29d0.png)

</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://pypi.org/project/Yolov7paddleOcr-python/"><img src="https://img.shields.io/pypi/dm/Yolov7paddleOcr-python?color=9cf"></a>
</p>

**⚡️Yolov7paddleOcr is an **easy-to-use and efficient** inference deployment development kit. Covers the industry 🔥** popular AI models** and provides 📦** out-of-the-box** deployment experience, including image classification, target detection, image segmentation, face detection, face recognition, human key point recognition, text recognition , semantic understanding and other multi-tasks, to meet the industrial deployment needs of developers with multiple scenarios, multiple hardware, and multiple platforms.

| Potrait Segmentation                                                                                        | Image Matting                                                                                             | Semantic Segmentation                                                                                                                                                                                                      | Real-Time Matting                                                                                                                                                                                                                         |
|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src='https://user-images.githubusercontent.com/54695910/188054718-6395321c-8937-4fa0-881c-5b20deb92aaa.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188058231-a5fe1ce1-0a38-460f-9582-e0b881514908.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054711-6119f0e7-d741-43b1-b273-9493d103d49f.gif' height="126px" width="190px">                                                                                                                    | <img src='https://user-images.githubusercontent.com/54695910/188054691-e4cb1a70-09fe-4691-bc62-5552d50bd853.gif' height="126px" width="190px">                                                                                                                                 |
| **OCR**                 |  **Behavior Recognition**           | **Object Detection**                  |**Pose Estimation**
| <img src='https://user-images.githubusercontent.com/54695910/188054669-a85996ba-f7f3-4646-ae1f-3b7e3e353e7d.gif' height="126px" width="190px"> |<img src='https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054680-2f8d1952-c120-4b67-88fc-7d2d7d2378b4.gif' height="126px" width="190px"  >                                                                                                                              |<img src='https://user-images.githubusercontent.com/54695910/188054671-394db8dd-537c-42b1-9d90-468d7ad1530e.gif' height="126px" width="190px">  |
| **Face Alignment**                                                                                        | **3D Object Detection**                                                                                        |  **Face Editing**                                                                                                                                                                                                           | **Image Animation**  
| <img src='https://user-images.githubusercontent.com/54695910/188059460-9845e717-c30a-4252-bd80-b7f6d4cf30cb.png' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188270227-1a4671b3-0123-46ab-8d0f-0e4132ae8ec0.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054663-b0c9c037-6d12-4e90-a7e4-e9abf4cf9b97.gif' height="126px" width="126px">  |  <img src='https://user-images.githubusercontent.com/54695910/188056800-2190e05e-ad1f-40ef-bf71-df24c3407b2d.gif' height="126px" width="190px">

## Latest updates

- 🔥 **2022.8.18：Yolov7paddleOcr [release/v0.2.0](https://github.com/PaddlePaddle/Yolov7paddleOcr/releases/tag/release%2F0.2.0)** <br>
    - **New upgrade for server deployment: faster inference performance, more visual model support**  
        - Released a high-performance inference engine SDK based on x86 CPU and NVIDIA GPU, greatly improving inference speed
        - Integrate inference engines such as Paddle Inference, ONNX Runtime, TensorRT and provide a unified deployment experience
        - Supports a full range of target detection models such as YOLOv7, YOLOv6, YOLOv5, PP-YOLOE and provides [end-to-end deployment example](examples/vision/detection/)
        - Support 40+ key models such as face detection, face recognition, real-time portrait matting, image segmentation, etc. and [Demo example](examples/vision/)
        - Supports both Python and C++ language deployment
    - **Edge mobile terminal deployment adds NPU chip deployment capabilities such as Rockchip, Amlogic, and NXP**
        - Released lightweight target detection [Picodet-NPU deployment Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)，Provide the ultimate INT8 full quantitative reasoning ability
## Table of contents:
* **Server deployment**
    * [Python SDK Quick Start](#Yolov7paddleOcr-quick-start-python)  
    * [C++ SDK Quick Start](#Yolov7paddleOcr-quick-start-cpp)
    * [Server Model Support List](#Yolov7paddleOcr-server-models)
* **End-to-end deployment**
    * [EasyEdge edge deployment](#fYolov7paddleOcr-edge-sdk-arm-linux)  
    * [EasyEdge Mobile Deployment](#Yolov7paddleOcr-edge-sdk-ios-android)  
    * [EasyEdge Custom Model Deployment](#Yolov7paddleOcr-edge-sdk-custom)  
    * [Paddle Lite NPU Deployment](#Yolov7paddleOcr-edge-sdk-npu)
    * [End-to-end model support list](#Yolov7paddleOcr-edge-sdk)
* [Acknowledge](#Yolov7paddleOcr-acknowledge)  
* [License](#Yolov7paddleOcr-license)

## Server deployment:

### Python SDK Quick Start
<div id="Yolov7paddleOcr-quick-start-python"></div>

#### Quick installation:

##### pre-dependencies
- CUDA >= 11.2
- cuDNN >= 8.0
- python >= 3.6
- OS: Linux x86_64/macOS/Windows 10

##### Install the GPU version

```bash
pip install numpy opencv-python Yolov7paddleOcr-gpu-python -f https://www.paddlepaddle.org.cn/whl/Yolov7paddleOcr.html
```
##### [Conda installation (recommended)](docs/quick_start/Python_prebuilt_wheels.md)
```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```
##### Install the CPU version

```bash
pip install numpy opencv-python Yolov7paddleOcr-python -f https://www.paddlepaddle.org.cn/whl/Yolov7paddleOcr.html
```

#### Python inference example

* Prepare models and pictures

```bash
wget https://bj.bcebos.com/paddlehub/Yolov7paddleOcr/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* Test inference results
```python
# GPU/TensorRT Deployment Reference examples/vision/detection/paddledetection/python
import cv2
import Yolov7paddleOcr.vision as vision

model = vision.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                 "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                 "ppyoloe_crn_l_300e_coco/infer_cfg.yml")
im = cv2.imread("000000014439.jpg")
result = model.predict(im.copy())
print(result)

vis_im = vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```

### C++ SDK Quick Start
<div id="Yolov7paddleOcr-quick-start-cpp"></div>

#### Installation:

- Reference [C++ precompiled library download](docs/quick_start/CPP_prebuilt_libraries.md)文档  

#### C++ Reasoning Example

* Prepare models and pictures

```bash
wget https://bj.bcebos.com/paddlehub/Yolov7paddleOcr/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* Test inference results

```C++
// GPU/TensorRT Deployment Reference examples/vision/detection/paddledetection/cpp
#include "Yolov7paddleOcr/vision.h"

int main(int argc, char* argv[]) {
  namespace vision = Yolov7paddleOcr::vision;
  auto model = vision::detection::PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                          "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                          "ppyoloe_crn_l_300e_coco/infer_cfg.yml");
  auto im = cv::imread("000000014439.jpg");

  vision::DetectionResult res;
  model.Predict(&im, &res);

  auto vis_im = vision::Visualize::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
  return 0;
}
```

For more deployment cases, please refer to [Visual Model Deployment Example](examples/vision) .

### Server Model Support List 🔥🔥🔥

<div id="Yolov7paddleOcr-server-models"></div>

Symbol description: (1) ✅: already supported; (2) ❔: future support; (3) ❌: not currently supported; (4) --: not considered for now;<br>
Link description: "Model column" will jump to the model inference Demo code

| Scenario | Model  | API | Linux   |   Linux      |   Win   |  Win    |   Mac     | Mac     |  Linux |   Linux |  
| :--------:  | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |:--------: |
|  --- | --- |  --- |  <font size=2> X86 CPU |  <font size=2> NVIDIA GPU |  <font size=2> Intel  CPU |  <font size=2> NVIDIA GPU |  <font size=2> Intel CPU |  <font size=2> Arm CPU   | <font size=2>  AArch64 CPU  | <font size=2> NVIDIA Jetson |
| <font size=2> Classification | <font size=2> [PaddleClas/ResNet50](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/PP-LCNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |   ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/PP-LCNetv2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/GhostNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/PP-HGNet](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Classification | <font size=2> [PaddleClas/SwinTransformer](./examples/vision/classification/paddleclas) | <font size=2> [Python](./examples/vision/classification/paddleclas/python)/[C++](./examples/vision/classification/paddleclas/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLOE](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅  |
| <font size=2> Detection | <font size=2> [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ✅  |
| <font size=2> Detection | <font size=2> [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ✅  |
| <font size=2> Detection | <font size=2> [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ | ✅  |
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLO](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ❌ | ❌ | ❌ |
| <font size=2> Detection | <font size=2> [PaddleDetection/PP-YOLOv2](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ❌ | ❌ | ❌ |
| <font size=2> Detection | <font size=2> [PaddleDetection/FasterRCNN](./examples/vision/detection/paddledetection) | <font size=2> [Python](./examples/vision/detection/paddledetection/python)/[C++](./examples/vision/detection/paddledetection/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ | ❌ | ❌ | ❌ |
| <font size=2> Detection | <font size=2> [Megvii-BaseDetection/YOLOX](./examples/vision/detection/yolox) | <font size=2> [Python](./examples/vision/detection/yolox/python)/[C++](./examples/vision/detection/yolox/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Detection | <font size=2> [WongKinYiu/YOLOv7](./examples/vision/detection/yolov7) | <font size=2> [Python](./examples/vision/detection/yolov7/python)/[C++](./examples/vision/detection/yolov7/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Detection | <font size=2> [meituan/YOLOv6](./examples/vision/detection/yolov6) | <font size=2> [Python](./examples/vision/detection/yolov6/python)/[C++](./examples/vision/detection/yolov6/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Detection | <font size=2> [ultralytics/YOLOv5](./examples/vision/detection/yolov5) | <font size=2> [Python](./examples/vision/detection/yolov5/python)/[C++](./examples/vision/detection/yolov5/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Detection | <font size=2> [WongKinYiu/YOLOR](./examples/vision/detection/yolor) | <font size=2> [Python](./examples/vision/detection/yolor/python)/[C++](./examples/vision/detection/yolor/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Detection | <font size=2> [WongKinYiu/ScaledYOLOv4](./examples/vision/detection/scaledyolov4) | <font size=2> [Python](./examples/vision/detection/scaledyolov4/python)/[C++](./examples/vision/detection/scaledyolov4/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Detection | <font size=2> [ppogg/YOLOv5Lite](./examples/vision/detection/yolov5lite) | <font size=2> [Python](./examples/vision/detection/yolov5lite/python)/[C++](./examples/vision/detection/yolov5lite/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Detection | <font size=2> [RangiLyu/NanoDetPlus](./examples/vision/detection/nanodet_plus) | <font size=2> [Python](./examples/vision/detection/nanodet_plus/python)/[C++](./examples/vision/detection/nanodet_plus/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅|
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-LiteSeg](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-HumanSegLite](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/PP-HumanSegServer](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Segmentation | <font size=2> [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg) | <font size=2> [Python](./examples/vision/segmentation/paddleseg/python)/[C++](./examples/vision/segmentation/paddleseg/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Face Detection | <font size=2> [biubug6/RetinaFace](./examples/vision/facedet/retinaface) | <font size=2> [Python](./examples/vision/facedet/retinaface/python)/[C++](./examples/vision/facedet/retinaface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Face Detection | <font size=2> [Linzaer/UltraFace](./examples/vision/facedet/ultraface) | [<font size=2> Python](./examples/vision/facedet/ultraface/python)/[C++](./examples/vision/facedet/ultraface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> FaceDetection | <font size=2> [deepcam-cn/YOLOv5Face](./examples/vision/facedet/yolov5face) | <font size=2> [Python](./examples/vision/facedet/yolov5face/python)/[C++](./examples/vision/facedet/yolov5face/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Face Detection | <font size=2> [deepinsight/SCRFD](./examples/vision/facedet/scrfd) | <font size=2> [Python](./examples/vision/facedet/scrfd/python)/[C++](./examples/vision/facedet/scrfd/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Face Recognition | <font size=2> [deepinsight/ArcFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Face Recognition | <font size=2> [deepinsight/CosFace](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Face Recognition | <font size=2> [deepinsight/PartialFC](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Face Recognition | <font size=2> [deepinsight/VPL](./examples/vision/faceid/insightface) | <font size=2> [Python](./examples/vision/faceid/insightface/python)/[C++](./examples/vision/faceid/insightface/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅ |
| <font size=2> Matting | <font size=2> [ZHKKKe/MODNet](./examples/vision/matting/modnet) | <font size=2> [Python](./examples/vision/matting/modnet/python)/[C++](./examples/vision/matting/modnet/cpp) |  ✅       |  ✅    |  ✅     |  ✅    |  ✅ |  ✅ |  ✅ |  ✅|


## End-to-end deployment:

<div id="Yolov7paddleOcr-edge-doc"></div>

### EasyEdge edge deployment

<div id="Yolov7paddleOcr-edge-sdk-arm-linux"></div>

- ARM Linux system
  - [C++ Inference deployment (including video streaming）](./docs/arm_cpu/arm_linux_cpp_sdk_inference.md)
  - [C++ service deployment](./docs/arm_cpu/arm_linux_cpp_sdk_serving.md)
  - [Python Inference Deployment](./docs/arm_cpu/arm_linux_python_sdk_inference.md)
  - [Python service deployment](./docs/arm_cpu/arm_linux_python_sdk_serving.md)

### EasyEdge Mobile Deployment

<div id="Yolov7paddleOcr-edge-sdk-ios-android"></div>

- [iOS system deployment](./docs/arm_cpu/ios_sdk.md)
- [Android system deployment](./docs/arm_cpu/android_sdk.md)  

### EasyEdge Custom Model Deployment

<div id="Yolov7paddleOcr-edge-sdk-custom"></div>

- [Quickly implement personalized model replacement](./docs/arm_cpu/replace_model_with_another_one.md)

### Paddle Lite NPU Deployment
    
<div id="Yolov7paddleOcr-edge-sdk-npu"></div>

- [Rockchip-NPU/Amlogic-NPU/NXP-NPU](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/linux/picodet_detection)

### End-to-end model support list

<div id="Yolov7paddleOcr-edge-sdk"></div>

| Scene | Model | Size(MB) | Linux  | Android  | iOS    |Linux  | Linux  | Linux    |updating...|
|:------------------:|:----------------------------:|:---------------------:|:---------------------:|:----------------------:|:---------------------:| :------------------:|:----------------------------:|:---------------------:|:---------------------:|
| ---                | ---                          | ---                   | ARM CPU |  ARM CPU | ARM CPU |Rockchip NPU <br>RV1109<br>RV1126<br>RK1808 | Amlogic NPU <br>A311D<br>S905D<br>C308X  | NXP NPU<br>  i.MX 8M Plus    |updating...｜
| Classification     | PP-LCNet                     | 11.9                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | PP-LCNetv2                   | 26.6                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | EfficientNet                 | 31.4                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | GhostNet                     | 20.8                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | MobileNetV1                  | 17                    | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | MobileNetV2                  | 14.2                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | MobileNetV3                  | 22                    | ✅                     | ✅                      | ✅                     |❔  | ❔  | ❔  |❔|
| Classification     | ShuffleNetV2                 | 9.2                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | SqueezeNetV1.1               | 5                     | ✅                     | ✅                      | ✅                     |
| Classification     | Inceptionv3                  | 95.5                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | PP-HGNet                     | 59                    | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Classification     | SwinTransformer_224_win7     | 352.7                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-PicoDet_s_320_coco        | 4.1                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-PicoDet_s_320_lcnet       | 4.9                   | ✅                     | ✅                      | ✅                     |✅   | ✅   | ✅     | ❔|
| Detection          | CenterNet                    | 4.8                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | YOLOv3_MobileNetV3           | 94.6                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-YOLO_tiny_650e_coco       | 4.4                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | SSD_MobileNetV1_300_120e_voc | 23.3                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-YOLO_ResNet50vd           | 188.5                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-YOLOv2_ResNet50vd         | 218.7                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | PP-YOLO_crn_l_300e_coco      | 209.1                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Detection          | YOLOv5s                      | 29.3                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Face Detection      | BlazeFace                    | 1.5                   | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Face Detection      | RetinaFace                   | 1.7                   | ✅                     | ❌                      | ❌                     |--  | --  | --    |--|
| Keypoint Detection | PP-TinyPose                  | 5.5                   | ✅                     | ✅                      | ✅                     |❔ | ❔ | ❔ |❔|
| Segmentation       | PP-LiteSeg(STDC1)            | 32.2                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Segmentation       | PP-HumanSeg-Lite             | 0.556                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Segmentation       | HRNet-w18                    | 38.7                  | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Segmentation       | PP-HumanSeg-Server           | 107.2                 | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| Segmentation       | Unet                         | 53.7                  | ❌                     | ✅                      | ❌                     |--  | --  | --    |--|
| OCR                | PP-OCRv1                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| OCR                | PP-OCRv2                     | 2.3+4.4               | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|
| OCR                | PP-OCRv3                     | 2.4+10.6              | ✅                     | ✅                      | ✅                     |❔ | ❔ | ❔  |❔|
| OCR                | PP-OCRv3-tiny                | 2.4+10.7              | ✅                     | ✅                      | ✅                     |--  | --  | --    |--|


## License

<div id="Yolov7paddleOcr-license"></div>

FastDeploy is provided under the [Apache-2.0open source protocol](./LICENSE)。
