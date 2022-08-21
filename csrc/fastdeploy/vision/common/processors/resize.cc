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

#include "fastdeploy/vision/common/processors/resize.h"

namespace fastdeploy {
namespace vision {

bool Resize::CpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Resize: The format of input is not HWC." << std::endl;
    return false;
  }
  cv::Mat* im = mat->GetCpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  if (width_ > 0 && height_ > 0) {
    if (use_scale_) {
      float scale_w = width_ * 1.0 / origin_w;
      float scale_h = height_ * 1.0 / origin_h;
      cv::resize(*im, *im, cv::Size(0, 0), scale_w, scale_h, interp_);
    } else {
      cv::resize(*im, *im, cv::Size(width_, height_), 0, 0, interp_);
    }
  } else if (scale_w_ > 0 && scale_h_ > 0) {
    cv::resize(*im, *im, cv::Size(0, 0), scale_w_, scale_h_, interp_);
  } else {
    FDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
               "or (scale_w > 0 && scale_h > 0)."
            << std::endl;
    return false;
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Resize::GpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Resize: The format of input is not HWC." << std::endl;
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  if (width_ > 0 && height_ > 0) {
    if (use_scale_) {
      float scale_w = width_ * 1.0 / origin_w;
      float scale_h = height_ * 1.0 / origin_h;
      cv::cuda::resize(*im, *im, cv::Size(0, 0), scale_w, scale_h, interp_);
    } else {
      cv::cuda::resize(*im, *im, cv::Size(width_, height_), 0, 0, interp_);
    }
  } else if (scale_w_ > 0 && scale_h_ > 0) {
    cv::cuda::resize(*im, *im, cv::Size(0, 0), scale_w_, scale_h_, interp_);
  } else {
    FDERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
               "or (scale_w > 0 && scale_h > 0)."
            << std::endl;
    return false;
  }
  mat->SetWidth(im->cols);
  mat->SetHeight(im->rows);
  return true;
}
#endif

bool Resize::Run(Mat* mat, int width, int height, float scale_w, float scale_h,
                 int interp, bool use_scale, ProcLib lib) {
  if (mat->Height() == height && mat->Width() == width) {
    return true;
  }
  auto r = Resize(width, height, scale_w, scale_h, interp, use_scale);
  return r(mat, lib);
}

} // namespace vision
} // namespace fastdeploy
