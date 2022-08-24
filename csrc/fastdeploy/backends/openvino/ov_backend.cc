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

#include "fastdeploy/backends/openvino/ov_backend.h"

namespace fastdeploy {

std::vector<int64_t> PartialShapeToVec(const ov::PartialShape& shape) {
  std::vector<int64_t> res;
  for (int i = 0; i < shape.size(); ++i) {
    auto dim = shape[i];
    if (dim.is_dynamic()) {
      res.push_back(-1);
    } else {
      res.push_back(dim.get_length());
    }
  }
  return res;
}

FDDataType OpenVINODataTypeToFD(const ov::element::Type& type) {
  if (type == ov::element::f32) {
    return FDDataType::FP32;
  } else if (type == ov::element::f64) {
    return FDDataType::FP64;
  } else if (type == ov::element::i8) {
    return FDDataType::INT8;
  } else if (type == ov::element::i32) {
    return FDDataType::INT32;
  } else if (type == ov::element::i64) {
    return FDDataType::INT64;
  } else {
    FDASSERT(false, "Only support float/double/int8/int32/int64 now.");
  }
  return FDDataType::FP32;
}

ov::element::Type FDDataTypeToOV(const FDDataType& type) {
  if (type == FDDataType::FP32) {
    return ov::element::f32;
  } else if (type == FDDataType::FP64) {
    return ov::element::f64;
  } else if (type == FDDataType::INT8) {
    return ov::element::i8;
  } else if (type == FDDataType::INT32) {
    return ov::element::i32;
  } else if (type == FDDataType::INT64) {
    return ov::element::i64;
  }
  FDASSERT(false, "Only support float/double/int8/int32/int64 now.");
  return ov::element::f32;
}

bool OpenVINOBackend::InitFromPaddle(const std::string& model_file,
                                     const std::string& params_file,
                                     const OpenVINOBackendOption& option) {
  if (initialized_) {
    FDERROR << "OpenVINOBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  option_ = option;
  ov::AnyMap properties;
  if (option_.cpu_thread_num > 0) {
    properties["INFERENCE_NUM_THREADS"] = option_.cpu_thread_num;
  }

  std::shared_ptr<ov::Model> model = core_.read_model(model_file, params_file);

  // Get inputs/outputs information from loaded model
  const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    TensorInfo info;
    auto partial_shape = PartialShapeToVec(inputs[i].get_partial_shape());
    info.shape.assign(partial_shape.begin(), partial_shape.end());
    info.name = inputs[i].get_any_name();
    info.dtype = OpenVINODataTypeToFD(inputs[i].get_element_type());
    input_infos_.emplace_back(info);
  }
  const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    TensorInfo info;
    auto partial_shape = PartialShapeToVec(outputs[i].get_partial_shape());
    info.shape.assign(partial_shape.begin(), partial_shape.end());
    info.name = outputs[i].get_any_name();
    info.dtype = OpenVINODataTypeToFD(outputs[i].get_element_type());
    output_infos_.emplace_back(info);
  }

  compiled_model_ = core_.compile_model(model, "CPU", properties);
  request_ = compiled_model_.create_infer_request();
  initialized_ = true;
  return true;
}

TensorInfo OpenVINOBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of outputs: %d.", index,
           NumOutputs());
  return input_infos_[index];
}

TensorInfo OpenVINOBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs: %d.", index,
           NumOutputs());
  return output_infos_[index];
}

bool OpenVINOBackend::InitFromOnnx(const std::string& model_file,
                                   const OpenVINOBackendOption& option) {
  if (initialized_) {
    FDERROR << "OpenVINOBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  option_ = option;
  ov::AnyMap properties;
  if (option_.cpu_thread_num > 0) {
    properties["INFERENCE_NUM_THREADS"] = option_.cpu_thread_num;
  }

  std::shared_ptr<ov::Model> model = core_.read_model(model_file);

  // Get inputs/outputs information from loaded model
  const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    TensorInfo info;
    auto partial_shape = PartialShapeToVec(inputs[i].get_partial_shape());
    info.shape.assign(partial_shape.begin(), partial_shape.end());
    info.name = inputs[i].get_any_name();
    info.dtype = OpenVINODataTypeToFD(inputs[i].get_element_type());
    input_infos_.emplace_back(info);
  }
  const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    TensorInfo info;
    auto partial_shape = PartialShapeToVec(outputs[i].get_partial_shape());
    info.shape.assign(partial_shape.begin(), partial_shape.end());
    info.name = outputs[i].get_any_name();
    info.dtype = OpenVINODataTypeToFD(outputs[i].get_element_type());
    output_infos_.emplace_back(info);
  }

  compiled_model_ = core_.compile_model(model, "CPU", properties);
  request_ = compiled_model_.create_infer_request();
  initialized_ = true;
  return true;
}

int OpenVINOBackend::NumInputs() const { return input_infos_.size(); }

int OpenVINOBackend::NumOutputs() const { return output_infos_.size(); }

bool OpenVINOBackend::Infer(std::vector<FDTensor>& inputs,
                            std::vector<FDTensor>* outputs) {
  if (inputs.size() != input_infos_.size()) {
    FDERROR << "[OpenVINOBackend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << input_infos_.size() << ")." << std::endl;
    return false;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    ov::Shape shape(inputs[i].shape.begin(), inputs[i].shape.end());
    ov::Tensor ov_tensor(FDDataTypeToOV(inputs[i].dtype), shape,
                         inputs[i].Data());
    request_.set_tensor(inputs[i].name, ov_tensor);
  }

  request_.infer();

  outputs->resize(output_infos_.size());
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    auto out_tensor = request_.get_output_tensor(i);
    auto out_tensor_shape = out_tensor.get_shape();
    std::vector<int64_t> shape(out_tensor_shape.begin(),
                               out_tensor_shape.end());
    (*outputs)[i].Allocate(shape,
                           OpenVINODataTypeToFD(out_tensor.get_element_type()),
                           output_infos_[i].name);
    memcpy((*outputs)[i].MutableData(), out_tensor.data(),
           (*outputs)[i].Nbytes());
  }
  return true;
}

} // namespace fastdeploy
