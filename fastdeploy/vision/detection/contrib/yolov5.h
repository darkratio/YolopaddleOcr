﻿// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {
namespace detection {

class FASTDEPLOY_DECL YOLOv5 : public FastDeployModel {
 public:
  YOLOv5(const std::string& model_file, const std::string& params_file = "",
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "yolov5"; }

  virtual bool Predict(cv::Mat* im, DetectionResult* result,
                       float conf_threshold = 0.25,
                       float nms_iou_threshold = 0.5);

  static bool Preprocess(Mat* mat, FDTensor* output,
                         std::map<std::string, std::array<float, 2>>* im_info,
                         const std::vector<int>& size = {640, 640},
                         const std::vector<float> padding_value = {114.0, 114.0,
                                                                   114.0},
                         bool is_mini_pad = false, bool is_no_pad = false,
                         bool is_scale_up = false, int stride = 32,
                         float max_wh = 7680.0, bool multi_label = true);

  static bool Postprocess(
      std::vector<FDTensor>& infer_results, DetectionResult* result,
      const std::map<std::string, std::array<float, 2>>& im_info,
      float conf_threshold, float nms_iou_threshold, bool multi_label,
      float max_wh = 7680.0);

  // tuple of (width, height)
  std::vector<int> size_;
  // padding value, size should be same with Channels
  std::vector<float> padding_value_;
  // only pad to the minimum rectange which height and width is times of stride
  bool is_mini_pad_;
  // while is_mini_pad = false and is_no_pad = true, will resize the image to
  // the set size
  bool is_no_pad_;
  // if is_scale_up is false, the input image only can be zoom out, the maximum
  // resize scale cannot exceed 1.0
  bool is_scale_up_;
  // padding stride, for is_mini_pad
  int stride_;
  // for offseting the boxes by classes when using NMS
  float max_wh_;
  // for different strategies to get boxes when postprocessing
  bool multi_label_;

 private:
  bool Initialize();

  bool IsDynamicInput() const { return is_dynamic_input_; }

  static void LetterBox(Mat* mat, std::vector<int> size,
                        std::vector<float> color, bool _auto,
                        bool scale_fill = false, bool scale_up = true,
                        int stride = 32);

  // whether to inference with dynamic shape (e.g ONNX export with dynamic shape
  // or not.)
  // YOLOv5 official 'export_onnx.py' script will export dynamic ONNX by
  // default.
  // while is_dynamic_shape if 'false', is_mini_pad will force 'false'. This
  // value will
  // auto check by fastdeploy after the internal Runtime already initialized.
  bool is_dynamic_input_;
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
