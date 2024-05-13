import tensorrt as trt
import ctypes
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import cv2
import shutil

TRT_LOGGER = trt.Logger()

engine_file = '/home/openmmlab/mmdeploy/mmdeploy/mmdeploy_models/mmrotate/rain_06_01/end2end.engine'
lib_path = '/persistent/mmrotate/models/mmdeploy_plugin/libmmdeploy_tensorrt_ops.so'
ctypes.CDLL(lib_path)

logger = trt.Logger(trt.Logger.WARNING)


def draw_bounding_boxes(image, output_list, threshold):
    image_size = image.shape[0]
    for entry in output_list:
        class_id = entry["class"]
        score = entry["score"]
        
        if class_id == -1 or score < threshold:
            continue

        coordinates = entry["coordinates"]
        width = entry["width"]
        length = entry["length"]
        angle = entry["angle"]
        
        x, y = [coord * image_size / 320 for coord in coordinates]
        width, length = [value * image_size / 320 for value in (width, length)]

        rect = ((x, y), (width, length), np.degrees(angle))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    return image

def preprocess(image):
    image = cv2.resize(image, (320, 320))
    # Mean normalization
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
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
            # dtype = trt.nptype(engine.get_binding_dtype(binding))
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
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        for i in range(len(output_buffers)):
            cuda.memcpy_dtoh_async(output_buffers[i], output_memories[i], stream)
        # Synchronize the stream
        stream.synchronize()
    
    
    output_boxes = output_buffers[0].reshape((-1, 6))  
    output_classes = output_buffers[1].reshape((-1, 1))

    output_combined = np.concatenate((output_boxes, output_classes), axis=1)

    output_list = list()
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
    folder_path_out = "/persistent/mmrotate/results/vegas_rain_2/"
    if os.path.exists(folder_path_out):
        shutil.rmtree(folder_path_out)
        print(f"Folder '{folder_path_out}' and its contents removed.")

    os.makedirs(folder_path_out)
    print(f"Folder '{folder_path_out}' created.")
    
    folder_path_in = "dataset/sub_datasets/vegas_rain_church_2/imgs"
    files = os.listdir(folder_path_in)

    # Loop through each file in the folder
    with load_engine(engine_file) as engine:
        for file in files:
            if file.endswith(".png"):
                print("Reading input image from file {}".format(file))
                image_path = os.path.join(folder_path_in, file)
                image = cv2.imread(image_path)

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            output_list = inference(engine, image)
            threshold = 0.5
            img_with_boxes = draw_bounding_boxes(image, output_list, threshold)
            new_name = os.path.splitext(file)[0] + "_out.png"
            output_path = os.path.join(folder_path_out, new_name)
            cv2.imwrite(output_path, img_with_boxes)


if __name__ == '__main__':
    main()