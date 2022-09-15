# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastdeploy as fd
import cv2
import os
import numpy as np
import pynvml
import psutil
import GPUtil
import time


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleSeg model.")
    parser.add_argument(
        "--image", type=str, required=False, help="Path of test image file.")
    parser.add_argument(
        "--cpu_num_thread",
        type=int,
        default=8,
        help="default number of cpu thread.")
    parser.add_argument(
        "--device_id", type=int, default=0, help="device(gpu) id")
    parser.add_argument(
        "--iter_num",
        required=True,
        type=int,
        default=300,
        help="number of iterations for computing performace.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default="ort",
        help="inference backend, ort, ov, trt, paddle.")
    parser.add_argument(
        "--enable_trt_fp16",
        type=bool,
        default=False,
        help="whether enable fp16 in trt backend")
    args = parser.parse_args()
    return args


def build_option(args):
    option = fd.RuntimeOption()
    device = args.device
    backend = args.backend
    option.set_cpu_thread_num(args.cpu_num_thread)
    if device == "gpu":
        option.use_gpu(args.device_id)

    if backend == "trt":
        assert device == "gpu", "the trt backend need device==gpu"
        option.use_trt_backend()
        if args.enable_trt_fp16:
            option.enable_trt_fp16()
    elif backend == "ov":
        assert device == "cpu", "the openvino backend need device==cpu"
        option.use_openvino_backend()
    elif backend == "paddle":
        option.use_paddle_backend()
    elif backend == "ort":
        option.use_ort_backend()
    else:
        print("%s is an unsupported backend" % backend)

    return option


def get_current_memory_mb(gpu_id=None):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    if gpu_id is not None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return cpu_mem, gpu_mem


def get_current_gputil(gpu_id):
    GPUs = GPUtil.getGPUs()
    gpu_load = GPUs[gpu_id].load
    return gpu_load


if __name__ == '__main__':

    args = parse_arguments()
    option = build_option(args)
    model_file = os.path.join(args.model, "model.pdmodel")
    params_file = os.path.join(args.model, "model.pdiparams")
    config_file = os.path.join(args.model, "deploy.yaml")

    gpu_id = args.device_id
    end2end_statis = list()
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    if args.device == "cpu":
        file_path = args.model + "_model_" + args.backend + "_" + \
            args.device + "_" + str(args.cpu_num_thread) + ".txt"
    else:
        if args.enable_trt_fp16:
            file_path = args.model + "_model_" + args.backend + "_fp16_" + args.device + ".txt"
        else:
            file_path = args.model + "_model_" + args.backend + "_" + args.device + ".txt"
    f = open(file_path, "w")
    f.writelines("===={}====: \n".format(file_path.split("/")[1][:-4]))

    try:
        model = fd.vision.segmentation.PaddleSegModel(
            model_file, params_file, config_file, runtime_option=option)
        model.enable_record_time_of_runtime()
        for i in range(args.iter_num):
            im = cv2.imread(args.image)
            start = time.time()
            result = model.predict(im)
            end2end_statis.append(time.time() - start)
            gpu_util += get_current_gputil(gpu_id)
            cm, gm = get_current_memory_mb(gpu_id)
            cpu_mem += cm
            gpu_mem += gm

        runtime_statis = model.print_statis_info_of_runtime()

        warmup_iter = args.iter_num // 5
        repeat_iter = args.iter_num - warmup_iter
        end2end_statis = end2end_statis[warmup_iter:]

        dump_result = dict()
        dump_result["runtime"] = runtime_statis["avg_time"] * 1000
        dump_result["end2end"] = np.mean(end2end_statis) * 1000
        dump_result["cpu_rss_mb"] = cpu_mem / repeat_iter
        dump_result["gpu_rss_mb"] = gpu_mem / repeat_iter
        dump_result["gpu_util"] = gpu_util / repeat_iter

        f.writelines("Runtime(ms): {} \n".format(str(dump_result["runtime"])))
        f.writelines("End2End(ms): {} \n".format(str(dump_result["end2end"])))
        f.writelines("cpu_rss_mb: {} \n".format(
            str(dump_result["cpu_rss_mb"])))
        f.writelines("gpu_rss_mb: {} \n".format(
            str(dump_result["gpu_rss_mb"])))
        f.writelines("gpu_util: {} \n".format(str(dump_result["gpu_util"])))
    except:
        f.writelines("!!!!!Infer Failed\n")

    f.close()
