# convert_to_engine.py
import tensorrt as trt
import onnx
import sys
import os

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, input_shape=(1, 3, 640, 640)):
    """
    ONNX modelini TensorRT motoruna dönüştürür ve kaydeder.
    :param onnx_file_path: Giriş ONNX modelinin yolu.
    :param engine_file_path: Çıkış TensorRT motor dosyasının kaydedileceği yol.
    :param input_shape: Modelin beklediği giriş boyutu (batch, channels, height, width).
                        YOLOv11 için genellikle (1, 3, 640, 640) veya (1, 3, 1280, 1280) gibi.
    """
    print(f"ONNX modelinden TensorRT motoru oluşturuluyor: {onnx_file_path}")
    print(f"Motor şu adrese kaydedilecek: {engine_file_path}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32) # 4GB (Bellek ihtiyacına göre artırılabilir)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 etkinleştirildi.")
    else:
        print("FP16 desteklenmiyor, FP32 kullanılacak.")

    if not os.path.exists(onnx_file_path):
        print(f"HATA: ONNX dosyası bulunamadı: {onnx_file_path}")
        return None

    print(f"ONNX modelini ayrıştırılıyor: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("ONNX modeli başarıyla ayrıştırıldı.")

    print("TensorRT motoru oluşturuluyor...")
    # YENİ: build_engine yerine build_serialized_network kullanıldı
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("HATA: TensorRT motoru oluşturulamadı.")
        return None
    print("TensorRT motoru başarıyla oluşturuldu (seri hale getirilmiş).")

    # Seri hale getirilmiş motoru dosyaya kaydet
    print(f"Motor {engine_file_path} adresine kaydediliyor...")
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine) # Doğrudan seri hale getirilmiş veriyi yaz
    print("Motor başarıyla kaydedildi.")

    # İsteğe bağlı: Oluşturulan motoru deserialize edip döndürmek isterseniz
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

if __name__ == "__main__":
    # DİKKAT: Bu yolu PC'deki best.onnx dosyanızın gerçek yoluyla güncelleyin!
    ONNX_MODEL_PATH = "C:/Users/enesd/PycharmProjects/PythonProject/best1.onnx"
    # Çıkış .engine dosyasının yolu. ONNX dosyasıyla aynı dizinde olması tavsiye edilir.
    ENGINE_MODEL_PATH = ONNX_MODEL_PATH.replace(".onnx", ".engine")

    # YOLOv11 modelinizin giriş boyutunu doğru ayarlayın.
    # Eğer modeliniz 1280x1280 ise (1, 3, 1280, 1280) olarak değiştirin.
    YOLO_INPUT_SHAPE = (1, 3, 640, 640)

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"HATA: ONNX dosyası bulunamadı: {ONNX_MODEL_PATH}")
        print("Lütfen ONNX_MODEL_PATH değişkenini doğru dosya yolunuzla güncelleyin.")
        sys.exit(1)

    print(f"ONNX model yolu: {ONNX_MODEL_PATH}")
    print(f"Engine model yolu: {ENGINE_MODEL_PATH}")
    print(f"Giriş boyutu: {YOLO_INPUT_SHAPE}")

    build_engine(ONNX_MODEL_PATH, ENGINE_MODEL_PATH, YOLO_INPUT_SHAPE)
    print("Dönüştürme işlemi tamamlandı.")
