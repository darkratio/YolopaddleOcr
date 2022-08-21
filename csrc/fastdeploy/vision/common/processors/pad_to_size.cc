// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/vision/common/processors/pad_to_size.h"

namespace fastdeploy {
namespace vision {

bool PadToSize::CpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "PadToSize: The input data must be Layout::HWC format!"
            << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "PadToSize: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR
        << "PadToSize: Require input channels equals to size of padding value, "
           "but now channels = "
        << mat->Channels() << ", the size of padding values = " << value_.size()
        << "." << std::endl;
    return false;
  }
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  if (origin_w > width_) {
    FDERROR << "PadToSize: the input width:" << origin_w
            << " is greater than the target width: " << width_ << "."
            << std::endl;
    return false;
  }
  if (origin_h > height_) {
    FDERROR << "PadToSize: the input height:" << origin_h
            << " is greater than the target height: " << height_ << "."
            << std::endl;
    return false;
  }
  if (origin_w == width_ && origin_h == height_) {
    return true;
  }

  cv::Mat* im = mat->GetCpuMat();
  cv::Scalar value;
  if (value_.size() == 1) {
    value = cv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = cv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = cv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }
  // top, bottom, left, right
  cv::copyMakeBorder(*im, *im, 0, height_ - origin_h, 0, width_ - origin_w,
                     cv::BORDER_CONSTANT, value);
  mat->SetHeight(height_);
  mat->SetWidth(width_);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool PadToSize::GpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "PadToSize: The input data must be Layout::HWC format!"
            << std::endl;
    return false;
  }
  if (mat->Channels() > 4) {
    FDERROR << "PadToSize: Only support channels <= 4." << std::endl;
    return false;
  }
  if (mat->Channels() != value_.size()) {
    FDERROR
        << "PadToSize: Require input channels equals to size of padding value, "
           "but now channels = "
        << mat->Channels() << ", the size of padding values = " << value_.size()
        << "." << std::endl;
    return false;
  }

  int origin_w = mat->Width();
  int origin_h = mat->Height();
  if (origin_w > width_) {
    FDERROR << "PadToSize: the input width:" << origin_w
            << " is greater than the target width: " << width_ << "."
            << std::endl;
    return false;
  }
  if (origin_h > height_) {
    FDERROR << "PadToSize: the input height:" << origin_h
            << " is greater than the target height: " << height_ << "."
            << std::endl;
    return false;
  }
  if (origin_w == width_ && origin_h == height_) {
    return true;
  }

  cv::cuda::GpuMat* im = mat->GetGpuMat();
  cv::Scalar value;
  if (value_.size() == 1) {
    value = cv::Scalar(value_[0]);
  } else if (value_.size() == 2) {
    value = cv::Scalar(value_[0], value_[1]);
  } else if (value_.size() == 3) {
    value = cv::Scalar(value_[0], value_[1], value_[2]);
  } else {
    value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
  }

  // top, bottom, left, right
  cv::cuda::copyMakeBorder(*im, *im, 0, height_ - origin_h, 0,
                           width_ - origin_w, cv::BORDER_CONSTANT, value);
  mat->SetHeight(height_);
  mat->SetWidth(width_);
  return true;
}
#endif

bool PadToSize::Run(Mat* mat, int width, int height,
                    const std::vector<float>& value, ProcLib lib) {
  auto p = PadToSize(width, height, value);
  return p(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
