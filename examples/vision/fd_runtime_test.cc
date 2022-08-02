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
#include <chrono>
#include "fastdeploy/fastdeploy_model.h"

#if ((!defined(_WIN32)) && (!defined(__CYGWIN__)))
#include "fastdeploy/vision.h"
#endif

namespace fd = fastdeploy;

using time_t_ = decltype(std::chrono::high_resolution_clock::now());
static time_t_ Time() { return std::chrono::high_resolution_clock::now(); };
static double TimeDiff(time_t_ t1, time_t_ t2) {
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
             .count() /
         1000.0;
}

#if ((!defined(_WIN32)) && (!defined(__CYGWIN__)))
static bool LoadImageAndPreprocess(const std::string& path, int target_h,
                                   int target_w, std::vector<float>* content) {
  cv::Mat img = cv::imread(path);  // BGR
  if (img.empty()) {
    return false;
  }

  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);  // RGB
  cv::resize(img, img, cv::Size(target_w, target_h));
  // normalize
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> std = {0.5f, 0.5f, 0.5f};
  std::vector<float> min(mean.size(), 0.0);
  std::vector<float> max(mean.size(), 255.0);
  std::vector<float> alpha, beta;
  for (auto c = 0; c < mean.size(); ++c) {
    float alpha_ = 1.0f;
    alpha_ /= (max[c] - min[c]);
    float beta_ = -1.0f * (mean[c] + min[c] * alpha_) / std[c];
    alpha_ /= std[c];
    alpha.push_back(alpha_);
    beta.push_back(beta_);
  }
  std::vector<cv::Mat> split_im;
  cv::split(img, split_im);
  for (int c = 0; c < img.channels(); c++) {
    split_im[c].convertTo(split_im[c], CV_32FC1, alpha[c], beta[c]);
  }
  cv::merge(split_im, img);
  // copy to content
  size_t total_bytes = img.total() * img.elemSize();
  content->resize(total_bytes / sizeof(float));
  std::memcpy(content->data(), img.data, total_bytes);
  return true;
}
#endif

static bool SaveContent(const std::string& path, std::vector<float>* content) {
  std::ofstream file(path, std::ofstream::out);
  if (!file.is_open()) {
    std::cout << "--- Can not open file: " << path << std::endl;
    return false;
  }
  if (content->size() == 0) {
    std::cout << "--- Content is empty" << std::endl;
    return false;
  }
  for (size_t i = 0; i < content->size(); ++i) {
    file << content->at(i) << "\n";
  }
  file.close();
  return true;
}

static bool SaveFDTensor(const std::string& path, fd::FDTensor* tensor) {
  std::ofstream file(path, std::ofstream::out);
  if (!file.is_open()) {
    std::cout << "--- Can not open file: " << path << std::endl;
    return false;
  }
  if (tensor->Numel() == 0) {
    std::cout << "--- Tensor is empty" << std::endl;
    return false;
  }
  float* data = static_cast<float*>(tensor->Data());
  for (size_t i = 0; i < tensor->Numel(); ++i) {
    file << data[i] << "\n";
  }
  file.close();
  return true;
}

static bool LoadContent(const std::string& path, std::vector<float>* content) {
  std::ifstream file(path, std::ofstream::in);
  if (!file.is_open()) {
    std::cout << "--- Can not open file: " << path << std::endl;
    return false;
  }
  content->clear();
  size_t i = 0;
  std::string line;
  while (std::getline(file, line)) {
    float value = atof(line.c_str());
    content->push_back(value);
    ++i;
  }
  file.close();
  return true;
}

static bool LoadFDTensor(const std::string& path, fd::FDTensor* tensor) {
  std::vector<float> content;
  if (!LoadContent(path, &content)) {
    return false;
  }
  tensor->Allocate({static_cast<int>(content.size())}, fd::FDDataType::FP32);
  std::memcpy(tensor->Data(), content.data(), tensor->Nbytes());
  return true;
}

static bool FDTensorDiff(fd::FDTensor& in, fd::FDTensor& out) {
  if ((in.dtype != out.dtype) || (in.dtype != fd::FDDataType::FP32)) {
    return false;
  }
  if ((in.Nbytes() != out.Nbytes())) {
    return false;
  }
  float total_diff = 0.f;
  float* in_data = static_cast<float*>(in.Data());
  float* out_data = static_cast<float*>(out.Data());
  for (size_t i = 0; i < in.Numel(); ++i) {
    total_diff += (in_data[i] - out_data[i]);
  }
  float mean_diff = total_diff / static_cast<float>(in.Numel());
  std::cout << "--- Total Diff: " << total_diff << "\n"
            << "--- Mean Diff: " << mean_diff << "\n"
            << "--- Total Elements: " << in.Numel() << std::endl;
  return true;
}

static void PrintTensorInfo(fd::TensorInfo& info) {
  std::cout << "--- [name]:" << info.name << " [shape]:(";
  for (size_t i = 0; i < info.shape.size(); ++i) {
    std::cout << info.shape[i];
    if (i != (info.shape.size() - 1)) {
      std::cout << ",";
    } else {
      std::cout << ")";
    }
  }
  std::cout << " [dtype]:" << fd::FDDataTypeStr(info.dtype) << std::endl;
}

int main(int argc, char* argv[]) {
// Test infoflow_headseg_model via FastDeploy Runtime with ORT backend.
#if (defined(_WIN32) || defined(__CYGWIN__))
  std::string model_dir = "../infoflow_headseg_model/";
  std::string params_file = model_dir + "__params__";
  std::string model_file = model_dir + "__model__";
  std::string content_path = "../head_seg.txt";
  std::string tensor_path = "../tensor.txt";
#else
  std::string model_dir = "../resources/models/infoflow_headseg_model/";
  std::string params_file = model_dir + "__params__";
  std::string model_file = model_dir + "__model__";
  std::string img_path = "../resources/images/head_seg.png";
  std::string content_path = "../resources/outputs/head_seg.txt";
  std::string tensor_path = "../resources/outputs/tensor_pdi.txt";
#endif

  int CPU_NUM_THRADS = 1;
  if (argc > 2) {
    CPU_NUM_THRADS = atoi(argv[2]);
  }
  std::cout << "--- CPU_NUM_THRADS: " << CPU_NUM_THRADS << std::endl;

  // setup option
  fd::RuntimeOption runtime_option;
  runtime_option.SetModelPath(model_file, params_file, "paddle");
  runtime_option.UseCpu();
  // runtime_option.UseOrtBackend(); // paddle2onnx -> ORT
  runtime_option.UsePaddleBackend();  // Paddle Inference
  runtime_option.SetCpuThreadNum(CPU_NUM_THRADS);
  // init runtime
  std::unique_ptr<fd::Runtime> runtime =
      std::unique_ptr<fd::Runtime>(new fd::Runtime());
  if (!runtime->Init(runtime_option)) {
    std::cerr << "--- Init FastDeploy Runitme Failed! "
              << "\n--- Model:  " << model_file
              << "\n--- Params: " << params_file << std::endl;
    return -1;
  } else {
    std::cout << "--- Init FastDeploy Runitme Done! "
              << "\n--- Model:  " << model_file
              << "\n--- Params: " << params_file << std::endl;
  }
  // init input tensor shape
  fd::TensorInfo info = runtime->GetInputInfo(0);
  if (runtime_option.backend == fd::Backend::PDINFER) {
    // init shape manually for paddle inference
    info.shape = {1, 3, 160, 320};
  }
  PrintTensorInfo(info);
  info.shape[0] = 1;  // force batch size == 1
  if (info.shape.size() == 0) {
    return -1;
  }
  if (info.dtype != fd::FDDataType::FP32) {
    std::cout << "--- Testing only support FP32 now!\n";
    return -1;
  }
  // init input FDTensor from TensorInfo
  int numel = std::accumulate(info.shape.begin(), info.shape.end(), 1,
                              std::multiplies<int>());
  std::vector<fd::FDTensor> input_tensors(1);
  std::vector<fd::FDTensor> output_tensors(1);
  std::vector<float> content(numel, 0.f);

#if (defined(_WIN32) || defined(__CYGWIN__))
  if (!LoadContent(content_path, &content)) {
    std::cerr << "--- Can not load content!\n";
    return -1;
  } else {
    std::cout << "--- Load " << content_path << " done!\n";
  }
#else
  int target_h = info.shape[2];
  int target_w = info.shape[3];
  if (!LoadImageAndPreprocess(img_path, target_h, target_w, &content)) {
    std::cerr << "--- Can not load and preprocess image!\n";
    return -1;
  } else {
    std::cout << "--- Load and preprocee" << img_path << " done!\n";
  }
  if (!SaveContent(content_path, &content)) {
    std::cerr << "--- Can not save content!\n";
    return -1;
  } else {
    std::cout << "--- Save " << content_path << "done!\n";
  }
#endif

  input_tensors[0].SetExternalData(info.shape, info.dtype, content.data());
  input_tensors[0].name = info.name;
  std::cout << "--- Init FDTensor Done!" << std::endl;

  // warmup
  if (!runtime->Infer(input_tensors, &output_tensors)) {
    std::cerr << "--- Runtime Warmup Failed!" << std::endl;
    return -1;
  } else {
    std::cout << "--- Runtime Warmup Done!" << std::endl;
  }

  // repeat testing
  size_t REPEATS = 1000;
  if (argc > 1) {
    REPEATS = atoi(argv[1]);
  }
  auto st = Time();
  for (size_t i = 0; i < REPEATS; ++i) {
    if (!runtime->Infer(input_tensors, &output_tensors)) {
      std::cerr << "--- Runtime Infer Failed: " << i << std::endl;
      return -1;
    }
    if (((i + 1) % (REPEATS / 10)) == 0) {
      std::cout << "--- Runtime Infer: " << (i + 1) << " Done!\n";
    }
  }
  std::cout << "--- Average: Repeats [" << REPEATS << "], mean time used: ["
            << TimeDiff(st, Time()) / static_cast<double>(REPEATS) << " ms]\n";

#if (defined(_WIN32) || defined(__CYGWIN__))
  // show tensor diff
  fd::FDTensor content_tensor;
  if (!LoadFDTensor(tensor_path, &content_tensor)) {
    std::cerr << "--- Can not load tensor!\n";
    return -1;
  } else {
    std::cout << "--- Load " << tensor_path << " done!\n";
  }
  if (!FDTensorDiff(content_tensor, output_tensors[0])) {
    std::cerr << "--- Can not show tensor diff!\n";
    return -1;
  } else {
    std::cout << "--- Show tensor diff done!\n";
  }
#else
  // save tensor
  if (!SaveFDTensor(tensor_path, &output_tensors[0])) {
    std::cerr << "--- Can not save tensor!\n";
    return -1;
  } else {
    std::cout << "--- Save " << tensor_path << " done!\n";
  }
#endif
  return 0;
}
