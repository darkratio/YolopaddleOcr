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

#include "fastdeploy/pybind/main.h"

namespace fastdeploy {
void BindVisualize(pybind11::module& m) {
  m.def("vis_detection",
        [](pybind11::array& im_data, vision::DetectionResult& result,
           float score_threshold, int line_size, float font_size) {
          auto im = PyArrayToCvMat(im_data);
          auto vis_im = vision::VisDetection(im, result, score_threshold,
                                             line_size, font_size);
          FDTensor out;
          vision::Mat(vis_im).ShareWithTensor(&out);
          return TensorToPyArray(out);
        })
      .def("vis_face_detection",
           [](pybind11::array& im_data, vision::FaceDetectionResult& result,
              int line_size, float font_size) {
             auto im = PyArrayToCvMat(im_data);
             auto vis_im =
                 vision::VisFaceDetection(im, result, line_size, font_size);
             FDTensor out;
             vision::Mat(vis_im).ShareWithTensor(&out);
             return TensorToPyArray(out);
           })
      .def("vis_segmentation",
           [](pybind11::array& im_data, vision::SegmentationResult& result,
              float weight) {
             cv::Mat im = PyArrayToCvMat(im_data);
             auto vis_im = vision::VisSegmentation(im, result, weight);
             FDTensor out;
             vision::Mat(vis_im).ShareWithTensor(&out);
             return TensorToPyArray(out);
           })
      .def("swap_background",
           [](pybind11::array& im_data, pybind11::array& background_data,
              vision::MattingResult& result, bool remove_small_connected_area) {
             cv::Mat im = PyArrayToCvMat(im_data);
             cv::Mat background = PyArrayToCvMat(background_data);
             auto vis_im = vision::SwapBackground(im, background, result,
                                                  remove_small_connected_area);
             FDTensor out;
             vision::Mat(vis_im).ShareWithTensor(&out);
             return TensorToPyArray(out);
           })
      .def("swap_background",
           [](pybind11::array& im_data, pybind11::array& background_data,
              vision::SegmentationResult& result, int background_label) {
             cv::Mat im = PyArrayToCvMat(im_data);
             cv::Mat background = PyArrayToCvMat(background_data);
             auto vis_im = vision::SwapBackground(im, background, result,
                                                  background_label);
             FDTensor out;
             vision::Mat(vis_im).ShareWithTensor(&out);
             return TensorToPyArray(out);
           })
      .def("vis_ppocr",
           [](pybind11::array& im_data, vision::OCRResult& result) {
             auto im = PyArrayToCvMat(im_data);
             auto vis_im = vision::VisOcr(im, result);
             FDTensor out;
             vision::Mat(vis_im).ShareWithTensor(&out);
             return TensorToPyArray(out);
           })
      .def("vis_matting",
           [](pybind11::array& im_data, vision::MattingResult& result,
              bool remove_small_connected_area) {
             cv::Mat im = PyArrayToCvMat(im_data);
             auto vis_im =
                 vision::VisMatting(im, result, remove_small_connected_area);
             FDTensor out;
             vision::Mat(vis_im).ShareWithTensor(&out);
             return TensorToPyArray(out);
           });

  pybind11::class_<vision::Visualize>(m, "Visualize")
      .def(pybind11::init<>())
      .def_static("vis_detection",
                  [](pybind11::array& im_data, vision::DetectionResult& result,
                     float score_threshold, int line_size, float font_size) {
                    auto im = PyArrayToCvMat(im_data);
                    auto vis_im = vision::Visualize::VisDetection(
                        im, result, score_threshold, line_size, font_size);
                    FDTensor out;
                    vision::Mat(vis_im).ShareWithTensor(&out);
                    return TensorToPyArray(out);
                  })
      .def_static(
          "vis_face_detection",
          [](pybind11::array& im_data, vision::FaceDetectionResult& result,
             int line_size, float font_size) {
            auto im = PyArrayToCvMat(im_data);
            auto vis_im = vision::Visualize::VisFaceDetection(
                im, result, line_size, font_size);
            FDTensor out;
            vision::Mat(vis_im).ShareWithTensor(&out);
            return TensorToPyArray(out);
          })
      .def_static(
          "vis_segmentation",
          [](pybind11::array& im_data, vision::SegmentationResult& result) {
            cv::Mat im = PyArrayToCvMat(im_data);
            auto vis_im = vision::Visualize::VisSegmentation(im, result);
            FDTensor out;
            vision::Mat(vis_im).ShareWithTensor(&out);
            return TensorToPyArray(out);
          })
      .def_static(
          "swap_background_matting",
          [](pybind11::array& im_data, pybind11::array& background_data,
             vision::MattingResult& result, bool remove_small_connected_area) {
            cv::Mat im = PyArrayToCvMat(im_data);
            cv::Mat background = PyArrayToCvMat(background_data);
            auto vis_im = vision::Visualize::SwapBackgroundMatting(
                im, background, result, remove_small_connected_area);
            FDTensor out;
            vision::Mat(vis_im).ShareWithTensor(&out);
            return TensorToPyArray(out);
          })
      .def_static("swap_background_segmentation",
                  [](pybind11::array& im_data, pybind11::array& background_data,
                     int background_label, vision::SegmentationResult& result) {
                    cv::Mat im = PyArrayToCvMat(im_data);
                    cv::Mat background = PyArrayToCvMat(background_data);
                    auto vis_im = vision::Visualize::SwapBackgroundSegmentation(
                        im, background, background_label, result);
                    FDTensor out;
                    vision::Mat(vis_im).ShareWithTensor(&out);
                    return TensorToPyArray(out);
                  })
      .def_static("remove_small_connected_area",
                  [](pybind11::array& alpha_pred_data, float threshold) {
                    cv::Mat alpha_pred = PyArrayToCvMat(alpha_pred_data);
                    auto vis_im = vision::Visualize::RemoveSmallConnectedArea(
                        alpha_pred, threshold);
                  })
      .def_static("vis_ppocr",
                  [](pybind11::array& im_data, vision::OCRResult& result) {
                    auto im = PyArrayToCvMat(im_data);
                    auto vis_im = vision::Visualize::VisOcr(im, result);
                    FDTensor out;
                    vision::Mat(vis_im).ShareWithTensor(&out);
                    return TensorToPyArray(out);
                  })
      .def_static("vis_matting_alpha",
                  [](pybind11::array& im_data, vision::MattingResult& result,
                     bool remove_small_connected_area) {
                    cv::Mat im = PyArrayToCvMat(im_data);
                    auto vis_im = vision::Visualize::VisMattingAlpha(
                        im, result, remove_small_connected_area);
                    FDTensor out;
                    vision::Mat(vis_im).ShareWithTensor(&out);
                    return TensorToPyArray(out);
                  });
}
}  // namespace fastdeploy
