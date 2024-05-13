import tensorrt as trt
import ctypes
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import cv2
import shutil
import json


TRT_LOGGER = trt.Logger()

engine_file = '/persistent/models/02_05_2024.engine'
lib_path = '/persistent/mmrotate/parking_spaces/models/mmdeploy_plugins/libmmdeploy_tensorrt_ops.so'
ctypes.CDLL(lib_path)
logger = trt.Logger(trt.Logger.WARNING)

def get_bounding_box(image, output_list, threshold):
    image_size = image.shape[0]
    output = []
    for idx, entry in enumerate(output_list):
        class_id = entry["class"]
        score = entry["score"]

        if class_id == -1 or score < threshold:
            continue

        coordinates = entry["coordinates"]
        width = entry["width"]
        length = entry["length"]
        angle = entry["angle"]

        x, y = [coord * image_size / 640 for coord in coordinates]
        width, length = [value * image_size / 640 for value in (width, length)]

        rect = ((x, y), (width, length), np.degrees(angle))
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        output.append({
            "bouding box": idx,
            "cx": x,
            "cy": y,
            "class": class_id,
            "width": width,
            "length": length,
            "angle": angle,
            "score": score,
            "points": box.tolist()  # Convert NumPy array to list for JSON serialization
        })

    return output


def preprocess(image):
    image = cv2.resize(image, (640, 640))
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    return np.moveaxis(data, 2, 0)


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        trt.init_libnvinfer_plugins(logger, namespace='')
        return runtime.deserialize_cuda_engine(f.read())


def decide_type(type):
    if int(type) == 0:
        return np.float32
    elif int(type) == 1:
        return np.char
    elif int(type) == 2:
        return np.int8
    elif int(type) == 3:
        return np.int32
    elif int(type) == 4:
        return np.bool_


def inference(engine, img):
    input_image = preprocess(img)
    image_width = input_image.shape[1]
    image_height = input_image.shape[2]

    output_buffers = []
    output_memories = []

    with engine.create_execution_context() as context:
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = decide_type(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
                output_buffers.append(output_buffer)
                output_memories.append(output_memory)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for i in range(len(output_buffers)):
            cuda.memcpy_dtoh_async(output_buffers[i], output_memories[i], stream)
        stream.synchronize()

    output_boxes = output_buffers[0].reshape((-1, 6))
    output_classes = output_buffers[1].reshape((-1, 1))
    output_combined = np.concatenate((output_boxes, output_classes), axis=1)

    output_list = []
    for entry in output_combined:
        coordinate = entry[:2]
        width = entry[2]
        length = entry[3]
        angle = entry[4]
        score = entry[5]
        class_id = int(entry[6])

        entry_dict = {
            "coordinates": coordinate,
            "width": width,
            "length": length,
            "angle": angle,
            "score": score,
            "class": class_id
        }

        output_list.append(entry_dict)

    return output_list


def main():
    folder_path_out = "/persistent/testes/metrics/02_05/"
    if os.path.exists(folder_path_out):
        shutil.rmtree(folder_path_out)
        print(f"Folder '{folder_path_out}' and its contents removed.")

    os.makedirs(folder_path_out)
    print(f"Folder '{folder_path_out}' created.")

    folder_path_in = "/home/openmmlab/mmrotate/dataset/global_model/600/sub_datasets/bev_imgs_vinicius_new_office/low_high_igi_imgs/"
    files = os.listdir(folder_path_in)

    with load_engine(engine_file) as engine:
        for file in files:
            if file.endswith(".png"):
                print("Reading input image from file {}".format(file))
                image_path = os.path.join(folder_path_in, file)
                image = cv2.imread(image_path)

                # if len(image.shape) == 2:
                #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                output_list = inference(engine, image)
                threshold = 0.2
                boxes = get_bounding_box(image, output_list, threshold)

                new_name = os.path.splitext(file)[0] + "_out.json"
                output_path = os.path.join(folder_path_out, new_name)
                
                with open(output_path, 'w') as json_file:
                    json.dump(boxes, json_file, indent=4)

if __name__ == '__main__':
    main()