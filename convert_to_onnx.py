import os
import sys
from ultralytics import YOLO
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def pt_to_onnx(pt_path, onnx_path, img_size=640):
    model = YOLO(pt_path)
    model.export(format="onnx", opset=12, imgsz=img_size, dynamic=False)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX oluşturulamadı: {onnx_path}")
    print("✅ ONNX export tamamlandı:", onnx_path)

def onnx_to_engine(onnx_file_path, engine_file_path, input_shape=(1, 3, 640, 640)):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print("✅ Engine export tamamlandı:", engine_file_path)

if __name__ == "__main__":
    pt_path = "C:/Users/enesd/PycharmProjects/PythonProject/best1.pt"
    onnx_path = pt_path.replace(".pt", ".onnx")
    engine_path = pt_path.replace(".pt", ".engine")

    pt_to_onnx(pt_path, onnx_path, img_size=640)
    onnx_to_engine(onnx_path, engine_path)
