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
#pragma once
#include "fastdeploy/core/fd_tensor.h"
#include "opencv2/core/core.hpp"

#ifdef ENABLE_OPENCV_CUDA
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#endif

namespace fastdeploy {
namespace vision {

enum Layout { HWC, CHW };

struct FASTDEPLOY_DECL Mat {
  explicit Mat(cv::Mat& mat) {
    cpu_mat = mat;
    device = Device::CPU;
    layout = Layout::HWC;
    height = cpu_mat.rows;
    width = cpu_mat.cols;
    channels = cpu_mat.channels();
  }

 private:
  int channels;
  int height;
  int width;
  cv::Mat cpu_mat;
#ifdef ENABLE_OPENCV_CUDA
  cv::cuda::GpuMat gpu_mat;
#endif

 public:
#ifdef ENABLE_OPENCV_CUDA
  cv::cuda::GpuMat* GetGpuMat();
#endif
  cv::Mat* GetCpuMat();

  FDDataType Type();
  int Channels() const { return channels; }
  int Width() const { return width; }
  int Height() const { return height; }
  void SetChannels(int s) { channels = s; }
  void SetWidth(int w) { width = w; }
  void SetHeight(int h) { height = h; }

  // Transfer the vision::Mat to FDTensor
  void ShareWithTensor(FDTensor* tensor);
  // Only support copy to cpu tensor now
  bool CopyToTensor(FDTensor* tensor);

  // debug functions
  // TODO(jiangjiajun) Develop a right process pipeline with c++ is not a easy
  // things
  // Will add more debug function here to help debug processed image
  // This function will print shape / mean of each channels of the Mat
  void PrintInfo(const std::string& flag);

  Layout layout = Layout::HWC;
  Device device = Device::CPU;
};

}  // namespace vision
}  // namespace fastdeploy
