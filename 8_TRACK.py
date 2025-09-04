# aim_lead_mode="auto", aim_lead_gain=1.0, aim_lead_lpf_alpha=0.5 ile başla.
# Çok agresif gelirse aim_lead_gain 0.8’e, çok yavaş kalırsa 1.2’ye çıkar.
# Titreme varsa aim_lead_lpf_alpha 0.6–0.7; hantalsa 0.3’e indir.
# Hedefler çok yavaşsa aim_lead_vel_thresh’i 30–40 px/s yapıp boşuna lead verme.
# Zaten feed-forward var; lead aktifken
# feedforward_*_gain küçük (0.0–0.2) tutulursa ikisi çakışmadan birbirini tamamlar (FF anlık hız katkısı, lead ise hatayı öne taşır)
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy, \
    QSpacerItem, QGroupBox, QLineEdit, QMessageBox, QRadioButton, QDialog, QCheckBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtCore import QTimer, Qt, QCoreApplication, QThread, pyqtSignal

import time
import numpy as np
import onnxruntime as ort
import socket
import json
import os
import traceback
import queue  # İş parçacığı güvenli iletişim için

# TensorRT içe aktarmaları
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # CUDA bağlamını otomatik olarak başlatır

    TRT_AVAILABLE = True
    print("HATA AYIKLAMA: TensorRT ve PyCUDA başarıyla içe aktarıldı.")
except ImportError:
    TRT_AVAILABLE = False
    print("UYARI: TensorRT veya PyCUDA bulunamadı. Yalnızca ONNX Runtime (CPU/GPU) kullanılabilir olacak.")
    print(
        "TensorRT kullanmak için 'pip install tensorrt pycuda' komutunu çalıştırdığınızdan ve CUDA/cuDNN kurduğunuzdan emin olun.")
except Exception as e:
    TRT_AVAILABLE = False
    print(f"UYARI: TensorRT/PyCUDA içe aktarma sırasında hata: {e}")
    print("TensorRT veya PyCUDA yanlış kurulmuş olabilir. Yalnızca ONNX Runtime (CPU/GPU) kullanılabilir olacak.")
    traceback.print_exc()

# --- YOLOv11 Model Yapılandırması ---
# DİKKAT: Bu yolu PC'deki best.engine veya best.onnx dosyanızın gerçek yoluyla güncelleyin!
# Her iki formatı da desteklemek için uzantıyı kontrol edeceğiz.
YOLO_MODEL_PATH = "C:/Users/enesd/PycharmProjects/PythonProject/best.engine"  # Veya .onnx
# YENİ: Aşama 3 için yeni model yolu
YOLO_MODEL_PATH_TASK3 = "C:/Users/enesd/PycharmProjects/PythonProject/best1.engine"

CONF_THRESHOLD = 0.4  # Daha iyi tespit için güven eşiği düşürüldü
NMS_THRESHOLD = 0.4
# GÜNCELLEDİ: data.yaml'a dayalı sınıflar
CLASSES = ['blue_balloon', 'red_balloon']  # data.yaml'dan güncellenmiş sınıflar

# YENİ: Aşama 3 için özel sınıflar
CLASSES_TASK3 = ['kir_Dai', 'kir_Kar', 'kir_Uc', 'mav_Dai', 'mav_Kar', 'mav_Uc', 'yes_Dai', 'yes_Kar', 'yes_Uc']

# IMG_HEIGHT ve IMG_WIDTH'i varsayılan değerlerle global olarak başlat
IMG_HEIGHT, IMG_WIDTH = 640, 640  # YOLOv11 için varsayılan

# TensorRT bağlamı ve motor değişkenleri
trt_runtime = None
trt_engine = None
trt_context = None
trt_buffers = None
trt_inputs = None
trt_outputs = None
trt_bindings = None
trt_stream = None

# YENİ: TensorRT için giriş/çıkış bağlama indekslerini saklamak için global değişkenler
trt_input_binding_idx = None
trt_output_binding_idx = None
# YENİ: Farklı modeller için ayrı değişkenler
yolo_model_task12 = None
yolo_model_task3 = None


# YOLO modelini yükleme fonksiyonu
def load_yolo_model(model_path):
    global trt_runtime, trt_engine, trt_context, trt_buffers, trt_inputs, trt_outputs, trt_bindings, trt_stream
    global IMG_HEIGHT, IMG_WIDTH, trt_input_binding_idx, trt_output_binding_idx

    print(f"YOLO modeli yükleniyor: {model_path}")
    session = None
    providers = []

    if model_path.endswith(".engine") and TRT_AVAILABLE:
        try:
            # TensorRT motorunu yükle
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            trt_runtime = trt.Runtime(TRT_LOGGER)
            with open(model_path, "rb") as f:
                trt_engine = trt_runtime.deserialize_cuda_engine(f.read())
            if not trt_engine:
                raise RuntimeError("TensorRT motoru yüklenemedi.")

            print("TensorRT motoru başarıyla yüklendi.")

            # Giriş ve çıkış boyutlarını al
            trt_context = trt_engine.create_execution_context()

            input_name = None
            output_name = None
            input_shape = None
            output_shape = None

            # num_io_tensors üzerinden döngü yap ve indeksleri kullan
            # YENİ: trt_engine.num_io_tensors kullanılıyor
            for i in range(trt_engine.num_io_tensors):
                binding_name = trt_engine.get_tensor_name(i)  # İndeksten ad al
                binding_shape = trt_engine.get_tensor_shape(binding_name)  # Addan şekil al
                binding_is_input = trt_engine.get_tensor_mode(
                    binding_name) == trt.TensorIOMode.INPUT  # Giriş olup olmadığını kontrol et

                if binding_is_input:
                    input_name = binding_name
                    input_shape = binding_shape
                    trt_input_binding_idx = i  # Giriş indeksini sakla
                else:
                    output_name = binding_name
                    output_shape = binding_shape
                    trt_output_binding_idx = i  # Çıkış indeksini sakla

            if input_name is None or output_name is None:
                raise RuntimeError("TensorRT motorundan giriş veya çıkış bağlama adları belirlenemedi.")

            # Dinamik toplu boyut için optimizasyon profilleri burada ayarlanabilir
            # if trt_engine.has_implicit_batch_dimension:
            #     pass

            # Giriş/çıkış boyutlarını güncelle
            if len(input_shape) == 4:
                IMG_HEIGHT = input_shape[2]
                IMG_WIDTH = input_shape[3]
                print(f"TensorRT motoru giriş boyutu algılandı: {IMG_WIDTH}x{IMG_HEIGHT}")
            else:
                print(
                    f"TensorRT motoru giriş boyutu beklenmedik formatta ({input_shape}). Varsayılan {IMG_WIDTH}x{IMG_HEIGHT} kullanılacak.")

            # Giriş/çıkış belleği ayır
            trt_inputs = []
            trt_outputs = []
            trt_bindings = [None] * trt_engine.num_io_tensors  # Tüm bağlamalar için None ile başlat
            trt_stream = cuda.Stream()

            for i in range(trt_engine.num_io_tensors):
                binding_name = trt_engine.get_tensor_name(i)
                binding_shape = trt_engine.get_tensor_shape(binding_name)
                binding_dtype = trt_engine.get_tensor_dtype(binding_name)

                size = trt.volume(binding_shape) * binding_dtype.itemsize
                # Düzeltme: pagelocked_empty eleman sayısını bekler, byte sayısını değil
                host_mem = cuda.pagelocked_empty(trt.volume(binding_shape), dtype=trt.nptype(binding_dtype))
                device_mem = cuda.mem_alloc(size)
                trt_bindings[i] = int(device_mem)  # trt_bindings'de doğru indekse ata

                if trt_engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                    trt_inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    trt_outputs.append({'host': host_mem, 'device': device_mem})

            # Bu durumda, oturum nesnesi None kalacak, doğrudan TensorRT motorunu kullanacağız.
            return "tensorrt"  # Modelin TensorRT olduğunu belirtmek için bir dize döndür

        except Exception as e:
            print(f"HATA: TensorRT motoru yüklenirken veya yapılandırılırken hata: {e}")
            traceback.print_exc()
            trt_engine = None
            trt_context = None
            trt_buffers = None
            trt_inputs = None
            trt_outputs = None
            trt_bindings = None
            trt_stream = None
            trt_input_binding_idx = None
            trt_output_binding_idx = None
            print("TensorRT yüklenemedi, ONNX Runtime'a geri dönülüyor.")
            # TensorRT başarısız olursa, ONNX Runtime'a geri dön
            return load_yolo_model(model_path.replace(".engine", ".onnx"))  # ONNX versiyonunu dene

    elif model_path.endswith(".onnx"):
        # ONNX Runtime ile yükle
        # --- BAŞLANGIÇ: GPU/CPU SEÇİMİ İÇİN DEĞİŞTİR ---
        # Bu bayrağı True yaparak CPU kullanımını zorla, False yaparak GPU'yu dene.
        # Çökme yaşıyorsanız, bunu True yapmayı deneyin.
        FORCE_CPU_FOR_YOLO = False  # <--- ÇÖKME TESTİ İÇİN BUNU TRUE YAP

        if not FORCE_CPU_FOR_YOLO and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            print("CUDAExecutionProvider mevcut. GPU kullanılacak.")
            cuda_provider_options = {
                "device_id": 0,  # Kullanılacak GPU kimliği (genellikle 0)
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                "enable_cuda_graph": False
            }
            # Sağlayıcı seçeneklerini oturum seçeneklerine ekle
            sess_options = ort.SessionOptions()  # sess_options burada tanımlanmalı
            sess_options.add_session_config_entry("session.provider.options",
                                                  json.dumps({"CUDAExecutionProvider": cuda_provider_options}))
        else:
            print("CUDAExecutionProvider mevcut değil veya CPU kullanımı zorlandı. CPU kullanılacak.")
            sess_options = ort.SessionOptions()  # sess_options burada da tanımlanmalı
        providers.append('CPUExecutionProvider')  # Her zaman CPU'yu yedek olarak ekle
        # --- SON: GPU/CPU SEÇİMİ İÇİN DEĞİŞTİR ---

        try:
            # ONNX modelini oturum seçenekleri ve sağlayıcılarla yükle
            session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            print("ONNX modeli başarıyla yüklendi.")

            # Eğer yolo_model yüklendiyse, modelden IMG_HEIGHT ve IMG_WIDTH'i al, aksi takdirde varsayılanları kullan
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            input_shape = session.get_inputs()[0].shape

            # ONNX model giriş şekli (toplu, kanallar, yükseklik, genişlik) formatında olmalı
            if len(input_shape) == 4:
                IMG_HEIGHT = input_shape[2]
                IMG_WIDTH = input_shape[3]
                print(f"ONNX model giriş boyutu algılandı: {IMG_WIDTH}x{IMG_HEIGHT}")
            else:
                # Eğer input_shape 4 elemanlı değilse veya beklenmedik bir formattaysa
                # Varsayılan değerler zaten global olarak tanımlanmıştır, sadece bir uyarı ver.
                print(
                    f"ONNX model giriş boyutu beklenmedik formatta ({input_shape}). Varsayılan {IMG_WIDTH}x{IMG_HEIGHT} kullanılacak.")
            return session
        except Exception as e:
            print(f"YOLO modeli yüklenirken hata: {e}")
            print(
                "Lütfen YOLO_MODEL_PATH'in doğru olduğundan, gerekli kütüphanelerin (onnxruntime-gpu) kurulu olduğundan ve GPU sürücülerinizin güncel olduğundan emin olun.")
            return None
    else:
        print("Desteklenmeyen model formatı. Yalnızca .onnx veya .engine desteklenir.")
        return None


# --- DÜZELTME 1: Model değişkenlerini daha temiz bir şekilde başlat ---
# Model yükleme sürecini güncelledik. Artık 'yolo_model' doğrudan bir InferenceSession nesnesi veya "tensorrt" dizesi olabilir.
yolo_model_task12 = load_yolo_model(YOLO_MODEL_PATH)
# Aşama 3 modeli sadece gerektiğinde yüklenecek şekilde None olarak bırakıldı.
yolo_model_task3 = None

input_name = None
output_name = None

# Eğer model TensorRT ise, giriş/çıkış adlarını TensorRT motorundan al
if yolo_model_task12 == "tensorrt":
    if trt_engine:
        # Bu değişkenler load_yolo_model içinde zaten ayarlandı.
        # Burada ek bir işlem yapmaya gerek yok.
        pass
else:  # Eğer ONNX Runtime ise
    if yolo_model_task12:
        # ONNX oturumundan giriş/çıkış adlarını al
        input_name = yolo_model_task12.get_inputs()[0].name
        output_name = yolo_model_task12.get_outputs()[0].name


class RPiCommunicator(QThread):
    # Sinyaller: Ana arayüze bilgi göndermek için
    status_update_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool)
    angles_update_signal = pyqtSignal(float, float)  # yaw, pitch
    response_received_signal = pyqtSignal(dict)  # Genel yanıtlar için

    def __init__(self, rpi_ip, rpi_port):
        super().__init__()
        self.rpi_ip = rpi_ip
        self.rpi_port = rpi_port
        self.rpi_socket = None
        self.is_connected = False
        self.command_queue = queue.Queue()  # Ana iş parçacığından komut almak için
        self.stop_requested = False
        self.socket_buffer = ""  # Gelen veriler için tampon

    def run(self):
        print("RPiCommunicator iş parçacığı başlatıldı.")
        while not self.stop_requested:
            if not self.is_connected:
                self._connect_to_rpi()
                if not self.is_connected:
                    time.sleep(1)  # Bağlantı başarısız olursa kısa bir süre bekle
                    continue

            # Komut kuyruğunu kontrol et ve gönder
            try:
                command = self.command_queue.get(timeout=0.01)  # Çok kısa zaman aşımı
                if command:
                    self._send_command(command)
            except queue.Empty:
                pass  # Kuyruk boş, devam et

            # Yanıtları dinle (engellemeyen veya kısa engellemeli)
            response = self._receive_response_non_blocking()
            if response:
                self.response_received_signal.emit(response)
                # Eğer bir açı güncellemesi ise, sinyali doğrudan yay
                if response.get("action") in ["get_angles", "set_angles", "move_by_direction",
                                              "set_proportional_angles_delta"] and response.get("status") == "ok":
                    yaw = response.get("current_yaw", 0.0)
                    pitch = response.get("current_pitch", 0.0)
                    self.angles_update_signal.emit(yaw, pitch)

            # CPU kullanımını azaltmak için küçük bir gecikme
            time.sleep(0.001)

        print("RPiCommunicator iş parçacığı durduruldu.")
        self._disconnect_rpi()

    def _connect_to_rpi(self):
        print(f"HATA AYIKLAMA (RPiComm): RPi'ye bağlanılıyor: {self.rpi_ip}:{self.rpi_port}...")
        try:
            self.rpi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.rpi_socket.settimeout(5)  # Bağlantı zaman aşımı
            self.rpi_socket.connect((self.rpi_ip, self.rpi_port))
            self.rpi_socket.settimeout(0.01)  # Veri alışverişi için çok kısa zaman aşımı
            self.is_connected = True
            self.connection_status_signal.emit(True)
            self.status_update_signal.emit("Durum: Raspberry Pi'ye Bağlandı!")
            print(
                f"HATA AYIKLAMA (RPiComm): Raspberry Pi'ye {self.rpi_ip}:{self.rpi_port} üzerinden başarıyla bağlanıldı.")
            # Bağlantıdan sonra RPi'den başlangıç açılarını iste (RPi periyodik olarak gönderdiği için gereksiz olabilir)
            self.command_queue.put({"action": "get_angles"})  # Kuyruğa ekle, iş parçacığı işleyecek
        except socket.error as e:
            print(f"HATA (RPiComm): RPi Bağlantı Hatası: {e}")
            traceback.print_exc()
            self.is_connected = False
            self.connection_status_signal.emit(False)
            self.status_update_signal.emit(f"Hata: RPi Bağlantı Hatası: {e}")
            if self.rpi_socket:
                self.rpi_socket.close()
            self.rpi_socket = None
        except Exception as e:
            print(f"HATA (RPiComm): Beklenmedik RPi bağlantı hatası: {e}")
            traceback.print_exc()
            self.is_connected = False
            self.connection_status_signal.emit(False)
            self.status_update_signal.emit(f"Hata: RPi Bağlantı Hatası: {e}")
            if self.rpi_socket:
                self.rpi_socket.close()
            self.rpi_socket = None
        finally:
            if not self.is_connected and self.rpi_socket:
                self.rpi_socket.close()
                self.rpi_socket = None

    def _disconnect_rpi(self):
        print("HATA AYIKLAMA (RPiComm): _disconnect_rpi çağrıldı.")
        if self.rpi_socket and self.is_connected:
            try:
                self.rpi_socket.shutdown(socket.SHUT_RDWR)
                self.rpi_socket.close()
                self.rpi_socket = None
                self.is_connected = False
                self.socket_buffer = ""
                self.connection_status_signal.emit(False)
                self.status_update_signal.emit("Durum: Raspberry Pi bağlantısı kesildi.")
                print("HATA AYIKLAMA: Raspberry Pi bağlantısı kesildi.")
            except Exception as e:
                print(f"HATA (RPiComm): RPi bağlantısı kesilirken hata: {e}")
                traceback.print_exc()
                self.status_update_signal.emit(f"Hata: RPi bağlantı kesme hatası: {e}")

    def _send_command(self, command_dict):
        if not self.is_connected or self.rpi_socket is None:
            self.status_update_signal.emit("Hata: Raspberry Pi'ye bağlı değil, komut gönderilemedi.")
            print("HATA (RPiComm): Raspberry Pi'ye bağlı değil, komut gönderilemedi.")
            return False
        try:
            message = (json.dumps(command_dict) + '\n').encode('utf-8')
            print(f"HATA AYIKLAMA (RPiComm): Komut gönderildi: {message.decode('utf-8').strip()}")
            self.rpi_socket.sendall(message)
            return True
        except socket.error as e:
            print(f"HATA (RPiComm): Komut gönderilirken soket bağlantı hatası: {e}. Bağlantı kesiliyor.")
            traceback.print_exc()
            self.status_update_signal.emit(f"Hata: RPi bağlantısı kesildi: {e}")
            self._disconnect_rpi()
            return False
        except Exception as e:
            print(f"HATA (RPiComm): Komut gönderilirken hata: {e}")
            traceback.print_exc()
            self.status_update_signal.emit(f"Hata: Komut gönderilemedi: {e}")
            return False

    def _receive_response_non_blocking(self):
        """
        Soket bağlantısından yanıtı engellemeyen bir şekilde okur.
        Tampondaki TÜM mesajları okur ve SADECE SONuncusunu döndürür.
        """
        if not self.is_connected or self.rpi_socket is None:
            return None

        try:
            # Engellemeyen okuma için zaman aşımını 0.01 saniyeye ayarla
            self.rpi_socket.settimeout(0.01)
            chunk = self.rpi_socket.recv(4096).decode('utf-8')  # Tamponu daha hızlı boşaltmak için boyutu artır
            if not chunk:
                print("HATA AYIKLAMA (RPiComm): _receive_response_non_blocking: Sunucu bağlantıyı kapattı (boş parça).")
                self._disconnect_rpi()
                return None

            self.socket_buffer += chunk

            last_valid_message = None
            while '\n' in self.socket_buffer:
                message, self.socket_buffer = self.socket_buffer.split('\n', 1)
                try:
                    # Gelen her mesajı ayrıştır ama sadece en sonuncuyu sakla
                    last_valid_message = json.loads(message)
                except json.JSONDecodeError as e:
                    print(f"HATA (RPiComm): JSON ayrıştırma hatası: {e}. Hatalı veri: '{message[:100]}...'")
                    self.status_update_signal.emit(f"Hata: RPi yanıtı ayrıştırılamadı: {e}")
                    # Geçersiz bir mesaj gelirse, döngüye devam et ama en son geçerli olanı koru

            # Tamponda biriken tüm mesajlar işlendikten sonra sadece en sonuncuyu döndür
            return last_valid_message

        except socket.timeout:
            pass  # Veri yok, normal
        except socket.error as e:
            print(f"HATA (RPiComm): Yanıt alınırken soket hatası: {e}. Bağlantı kesiliyor.")
            traceback.print_exc()
            self.status_update_signal.emit(f"Hata: RPi bağlantısı kesildi: {e}")
            self._disconnect_rpi()
        except Exception as e:
            print(f"HATA (RPiComm): Yanıt alınırken beklenmedik hata: {e}")
            traceback.print_exc()
            self._disconnect_rpi()
        return None

    def request_stop(self):
        self.stop_requested = True


class HavaSavunmaArayuz(QWidget):
    def __init__(self):
        print("HATA AYIKLAMA: HavaSavunmaArayuz başlatıldı.")
        super().__init__()
        self.setWindowTitle('Hava Savunma Sistemi Arayüzü')
        self.setGeometry(100, 100, 1920, 1080)
        self.setStyleSheet("background-color: black;")

        # --- UI Elemanları Oluşturma ---
        print("HATA AYIKLAMA: UI elemanları oluşturuluyor.")
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(1280, 720)
        self.camera_label.setStyleSheet("background-color: black;")

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Durum: Hazır")
        self.status_label.setStyleSheet("color: white; font-size: 14px;")
        self.target_info_label = QLabel("Hedef Bilgisi: Yok")
        self.target_info_label.setStyleSheet("color: white; font-size: 14px;")

        self.info_label = QLabel(self)
        self.info_label.move(1530, 5)
        self.info_label.setFixedSize(350, 30)
        self.update_info_panel("BUKREK Hava Savunma Sistemi")
        print("HATA AYIKLAMA: UI elemanları oluşturuldu.")

        # KCF ile ilgili değişkenler artık kullanılmıyor veya rolleri değişti
        self.kcf_bbox = None
        self.kcf_active = False  # KCF artık ana takipçi değil
        self.last_kcf_restart_time = 0
        self.kcf_restart_cooldown = 0.5  # Bu değer artık kullanılmıyor, ancak hataları önlemek için tutuldu

        # Bu eşik artık kullanılmıyor
        self.yolo_kcf_match_threshold_pixel = 75

        # Ateş kısıtlı bölge tanımları (şimdi varsayılan değerler, kullanıcı tarafından değiştirilebilir)
        self.no_fire_yaw_start = 0.0  # Varsayılan değer
        self.no_fire_yaw_end = 0.0  # Varsayılan değer

        self.movement_restricted_yaw_start = 0
        self.movement_restricted_yaw_end = 0

        # AYARLANDI: Nişan alma toleransı (piksel)
        self.aiming_tolerance = 10  # Daha hassas nişan alma için. Az değer daha hassas ama zorlu nişan.

        # --- ÖN NİŞAN (Lead) AYARLARI ---
        self.enable_aim_lead = False  # Bayrak: UI'dan aç/kapat
        self.aim_lead_mode = "auto"  # "auto" (fps+pipeline'a göre) veya "fixed"
        self.aim_lead_time_s = 0.06  # mode="fixed" ise sabit lead süresi
        self.aim_lead_gain = 1.0  # v*dt çarpanı (0.6–1.2 arası dene)
        self.aim_lead_max_px = 220  # çerçeve içinde sınır
        self.aim_lead_vel_thresh = 25.0  # yavaş hedefte lead uygulama
        self.aim_lead_lpf_alpha = 0.5  # öngörü noktasına LPF (titreşimi azaltır)
        self._lead_px_prev = None  # dahili: son öngörü noktası

        self.current_yaw_angle = 0.0
        self.current_pitch_angle = 0.0
        self.qr_detector = cv2.QRCodeDetector()

        self.last_qr_check_time = 0
        self.qr_check_interval = 2

        self.target_destroyed = False  # Hedefin yok edilip edilmediğini gösterir
        # YENİ: Hedef kaybedildiğinde veya yok edildiğinde yeni hedef aramak için zaman damgası
        self.target_lost_time = 0.0
        # YENİ: Hedef kaybedildiğinde yeni hedef aramadan önce ne kadar bekleneceği için zaman aşımı
        self.prediction_time_limit = 0.5  # 0.5 saniye bekleme süresi eklendi

        self.waiting_for_new_engagement_command = False  # Yeni bir angajman komutu beklenip beklenmediğini gösterir
        self.engagement_home_position_yaw = 0
        self.engagement_home_position_pitch = 0
        # YENİ: Aşama 3 için QR kodundan okunan dereceleri saklayacak sözlük
        self.qr_degrees = {}
        self.current_qr_char = None
        self.active_engagement_target_color = None
        self.active_engagement_target_shape = None
        self.current_tracked_target_class = None  # YENİ: Kilitli hedefin sınıfını saklar
        self.current_tracked_target_bbox = None  # YENİ: Kilitli hedefin sınırlayıcı kutusunu saklar
        self.is_ready_to_engage_from_qr = False  # YENİ: QR okunduktan sonraki angajman başlangıcı

        self.is_aimed_at_target = False  # Genel nişan alma durumu
        self.is_target_active = False  # Genel hedef takip durumu

        # YENİ: Hedefin kaç ardışık karede algılanamadığını takip etmek için
        self.missing_frames = 0
        self.MAX_MISSING_FRAMES = 10  # Hedefi tamamen kaybetmeden önceki maksimum kayıp kare sayısı (5'ten 10'a çıkarıldı)
        self.MAX_REACQUISITION_DISTANCE_PIXELS = 200  # Piksel. Kameranın görüş alanı ve hedef hızına göre ayarlanır. (150'den 200'e çıkarıldı)



        # --- PID Kontrol Değişkenleri ---
        # AYARLANDI: Daha hızlı ve daha agresif yanıt için PID kazançları artırıldı.
        self.KP_YAW = 0.8  # 0.04'ten 0.06'ya yükseltildi
        self.KI_YAW = 0.005  # 0.0001'den 0.0002'ye yükseltildi
        self.KD_YAW = 0.02  # 0.08'den 0.12'ye yükseltildi

        self.KP_PITCH = 0.7  # 0.04'ten 0.06'ya yükseltildi
        self.KI_PITCH = 0.005  # 0.0001'den 0.0002'ye yükseltildi
        self.KD_PITCH = 0.02  # 0.08'den 0.12'ye yükseltildi

        self.pid_update_time = time.time()
        self.integral_yaw = 0.0  # PID integral terimi başlatıldı
        self.last_error_yaw = 0.0  # PID son hata terimi başlatıldı
        self.integral_pitch = 0.0  # PID integral terimi başlatıldı
        self.last_error_pitch = 0.0

        # YENİ: Mevcut PID kazanç aralığını izlemek için değişken (yalnızca bir set olduğu için sabit kalacak)
        self.current_pid_range = "TEK_SET"

        # --- YENİ: İleri Besleme ve Tahminsel Kontrol için Değişkenler ---
        self.last_target_x = None
        self.last_target_y = None
        self.last_frame_time = None
        self.last_target_velocity_x = 0.0  # Hedefin son bilinen X hızı (piksel/saniye)
        self.last_target_velocity_y = 0.0  # Hedefin son bilinen Y hızı (piksel/saniye)
        # self.prediction_time_limit = 0.5  # Bu artık MAX_MISSING_FRAMES ile birlikte kullanılacak

        self.feedforward_yaw_gain = 0.0 # Devre dışı bırakmak için 0.0 olarak ayarla
        self.feedforward_pitch_gain = 0.0  # Devre dışı bırakmak için 0.0 olarak ayarla

        self.feedforward_velocity_threshold_px_s = 30.0

        self.velocity_update_error_threshold_px = 75

        # AYARLANDI: PID Çıkış Limiti (daha hızlı yanıt için daha geniş aralık)
        self.MAX_OUTPUT_DEGREE = 5.0  # 2.0'dan 3.0'a yükseltildi, daha büyük düzeltmeler için

        # AYARLANDI: Minimum hareket eşiği (PID çıktısı bunun altındaysa, hareket yok)
        self.MIN_OUTPUT_DEGREE_THRESHOLD = 0.01  # 0.03'ten 0.01'e düşürüldü, daha ince hareketlere izin vermek için

        # DERECE_BAŞINA_PİKSEL_YAW ve DERECE_BAŞINA_PİKSEL_PITCH işaretleri kontrol edildi.
        self.DEGREES_PER_PIXEL_YAW = 0.02
        self.DEGREES_PER_PIXEL_PITCH = -0.02

        # AYARLANDI: PID çıkışı için ölü bant (hata bunun altındaysa, PID çıkışı 0 olur)
        self.pid_output_deadband_degree = 0.01  # 0.02'den 0.01'e düşürüldü, daha hassas kilitleme için

        # --- Manuel Kontrol için Adım Boyutu ---
        # AYARLANDI: Daha akıcı hareket için manuel kontrol adım boyutu azaltıldı.
        self.manual_step_size = 1.0  # Manuel kontrol için sabit adım boyutu

        # --- AĞ BAĞLANTI AYARLARI ---
        self.rpi_ip = '192.168.137.229'  # Bu IP'nin doğru RPi IP'si olduğundan emin olun
        self.rpi_port = 12345

        # Açı komutları için minimum gönderme aralığı
        self.last_angle_command_send_time = time.time()
        self.angle_command_minimum_interval = 0.015  # Her 30ms'de bir komut göndermeye çalış

        # Ateşleme bekleme süresi
        self.last_fire_time = 0.0
        self.fire_cooldown_interval = 0.3

        # Tekrarlayan hata mesajlarını önlemek için
        self.last_status_message = ""
        self.last_status_time = 0
        self.status_message_cooldown_interval = 0.5  # Saniye

        print("HATA AYIKLAMA: RPiCommunicator başlatılıyor.")
        self.rpi_thread = RPiCommunicator(self.rpi_ip, self.rpi_port)
        self.rpi_thread.status_update_signal.connect(self._update_status_label)
        self.rpi_thread.connection_status_signal.connect(self._update_rpi_connection_status)
        self.rpi_thread.angles_update_signal.connect(self._update_current_angles)
        self.rpi_thread.response_received_signal.connect(self._process_rpi_response)
        self.rpi_thread.start()
        print("HATA AYIKLAMA: RPiCommunicator başlatıldı.")

        print("HATA AYIKLAMA: Ana düzen ve grup kutuları oluşturuluyor.")
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.camera_label, 8)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)
        right_layout.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.target_info_label)
        right_layout.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # --- GÖREVLER Grup Kutusu ---
        tasks_group_box = QGroupBox("GÖREVLER")
        tasks_group_box.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold; border: 2px solid white; border-radius: 8px; padding: 5px;")
        task_layout = QVBoxLayout()

        self.task1_button = QPushButton("Aşama 1", self)
        self.task2_button = QPushButton("Aşama 2", self)
        # GÜNCELLENDİ: Aşama 3 butonu
        self.task3_button = QPushButton("Aşama 3", self)
        self.manual_control_mode_button = QPushButton("Tam Manuel Kontrol", self)

        self.apply_button_style(self.task1_button, font_size=18, padding=10)
        self.apply_button_style(self.task2_button, font_size=18, padding=10)
        # GÜNCELLENDİ: Aşama 3 butonu stili
        self.apply_button_style(self.task3_button, font_size=18, padding=10)
        self.apply_button_style(self.manual_control_mode_button, font_size=18, padding=10)

        task_layout.addWidget(self.task1_button)
        task_layout.addWidget(self.task2_button)
        # GÜNCELLENDİ: Aşama 3 butonu eklendi
        task_layout.addWidget(self.task3_button)
        task_layout.addWidget(self.manual_control_mode_button)
        tasks_group_box.setLayout(task_layout)
        right_layout.addWidget(tasks_group_box)
        right_layout.addSpacerItem(
            QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # --- Aşama 3 Ayarları Grup Kutusu (YENİ) ---
        self.task3_settings_group_box = QGroupBox("Aşama 3 Ayarları")
        self.task3_settings_group_box.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold; border: 2px solid white; border-radius: 8px; padding: 5px;")
        task3_settings_layout = QVBoxLayout()
        self.a_label = QLabel("Angajman Bölgesi A (°):")
        self.a_label.setStyleSheet("color: white; font-size: 12px;")
        self.a_input = QLineEdit(self)
        self.a_input.setPlaceholderText("örn: -30.0")
        self.a_input.setStyleSheet("color: black; background-color: white; font-size: 12px;")

        self.b_label = QLabel("Angajman Bölgesi B (°):")
        self.b_label.setStyleSheet("color: white; font-size: 12px;")
        self.b_input = QLineEdit(self)
        self.b_input.setPlaceholderText("örn: 30.0")
        self.b_input.setStyleSheet("color: black; background-color: white; font-size: 12px;")

        self.task3_start_button = QPushButton("Angajmanı Al", self)
        self.apply_button_style(self.task3_start_button, font_size=18, padding=10, bg_color="#007bff")

        task3_settings_layout.addWidget(self.a_label)
        task3_settings_layout.addWidget(self.a_input)
        task3_settings_layout.addWidget(self.b_label)
        task3_settings_layout.addWidget(self.b_input)
        task3_settings_layout.addWidget(self.task3_start_button)
        self.task3_settings_group_box.setLayout(task3_settings_layout)
        right_layout.addWidget(self.task3_settings_group_box)
        self.task3_settings_group_box.setVisible(False)  # Başlangıçta gizli

        # --- KONTROL Grup Kutusu ---
        control_group_box = QGroupBox("KONTROL")
        control_group_box.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold; border: 2px solid white; border-radius: 8px; padding: 5px;")
        control_layout = QVBoxLayout()
        self.lead_checkbox = QCheckBox("Ön Nişan (Lead)")
        self.lead_checkbox.setStyleSheet("color: white; font-size: 14px;")
        self.lead_checkbox.setChecked(False)
        self.lead_checkbox.toggled.connect(lambda v: setattr(self, 'enable_aim_lead', v))
        control_layout.addWidget(self.lead_checkbox)

        self.connect_rpi_button = QPushButton('RPi Bağla', self)
        self.apply_button_style(self.connect_rpi_button, font_size=16, padding=8)
        control_layout.addWidget(self.connect_rpi_button)

        # Kamera Başlat/Durdur Butonları Yan Yana
        camera_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton('Kamera Başlat', self)
        self.stop_button = QPushButton('Kamera Durdur', self)

        # Buton stillerini güncelle
        self.apply_button_style(self.start_button, font_size=16, padding=8, bg_color="#28a745",
                                hover_color="#218838", pressed_color="#1e7e34")  # Yeşil
        self.apply_button_style(self.stop_button, font_size=16, padding=8, bg_color="#dc3545",
                                hover_color="#c82333", pressed_color="#bd2130")  # Kırmızı

        camera_buttons_layout.addWidget(self.start_button)
        camera_buttons_layout.addWidget(self.stop_button)
        control_layout.addLayout(camera_buttons_layout)

        self.stop_task_button = QPushButton('Görevi Durdur', self)
        self.fire_weapon_button = QPushButton("ATEŞ ET")

        self.apply_button_style(self.stop_task_button, font_size=16, padding=8, bg_color="#ffc107",
                                hover_color="#e0a800", pressed_color="#d39e00")  # Sarı
        self.apply_button_style(self.fire_weapon_button, font_size=20, padding=12, bg_color="#dc3545",
                                hover_color="#c82333", pressed_color="#bd2130")  # Kırmızı

        control_layout.addWidget(self.stop_task_button)
        control_layout.addWidget(self.fire_weapon_button)

        self.reset_angles_button = QPushButton("Açıları Sıfırla (0,0)", self)
        self.apply_button_style(self.reset_angles_button, font_size=16, padding=8, bg_color="#17a2b8",
                                hover_color="#138496", pressed_color="#117a8b")  # Mavi
        control_layout.addWidget(self.reset_angles_button)

        # --- AYARLAR Butonu ---
        self.settings_button = QPushButton("Ayarlar", self)
        self.apply_button_style(self.settings_button, font_size=16, padding=8, bg_color="#6f42c1")  # mor ton
        control_layout.addWidget(self.settings_button)
        self.settings_button.clicked.connect(self.open_settings_dialog)

        control_group_box.setLayout(control_layout)
        right_layout.addWidget(control_group_box)

        # --- Ateş Kontrolü ve Kısıtlı Bölge Ayarları Grup Kutusu ---
        self.fire_control_group_box = QGroupBox("Ateş Kontrolü ve Kısıtlı Bölge Ayarları")
        self.fire_control_group_box.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold; border: 2px solid white; border-radius: 8px; padding: 5px;")
        fire_control_layout = QVBoxLayout()

        # Kısıtlı Alan Başlangıç ve Bitiş Girişleri
        no_fire_start_layout = QHBoxLayout()
        label_no_fire_start = QLabel("Ateşsiz Bölge Başlangıç Yaw (°):")
        label_no_fire_start.setStyleSheet("color: white; font-size: 12px;")
        no_fire_start_layout.addWidget(label_no_fire_start)
        self.no_fire_start_input = QLineEdit(self)
        self.no_fire_start_input.setPlaceholderText(f"örn: -15.0")
        self.no_fire_start_input.setStyleSheet("color: black; background-color: white; font-size: 12px;")
        self.no_fire_start_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        no_fire_start_layout.addWidget(self.no_fire_start_input)
        fire_control_layout.addLayout(no_fire_start_layout)

        no_fire_end_layout = QHBoxLayout()
        label_no_fire_end = QLabel("Ateşsiz Bölge Bitiş Yaw (°):")
        label_no_fire_end.setStyleSheet("color: white; font-size: 12px;")
        no_fire_end_layout.addWidget(label_no_fire_end)
        self.no_fire_end_input = QLineEdit(self)
        self.no_fire_end_input.setPlaceholderText(f"örn: 15.0")
        self.no_fire_end_input.setStyleSheet("color: black; background-color: white; font-size: 12px;")
        self.no_fire_end_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        no_fire_end_layout.addWidget(self.no_fire_end_input)
        fire_control_layout.addLayout(no_fire_end_layout)

        # Kısıtlı Alanı Uygula ve Temizle Butonları Yan Yana
        no_fire_buttons_layout = QHBoxLayout()
        self.apply_no_fire_zone_button = QPushButton("Ateşsiz Bölge Uygula", self)
        self.apply_button_style(self.apply_no_fire_zone_button, font_size=14, padding=6, bg_color="#007bff",
                                hover_color="#0069d9", pressed_color="#0062cc")  # Mavi
        no_fire_buttons_layout.addWidget(self.apply_no_fire_zone_button)

        self.clear_no_fire_zone_button = QPushButton("Ateşsiz Bölgeyi Temizle", self)
        self.apply_button_style(self.clear_no_fire_zone_button, font_size=14, padding=6, bg_color="#6c757d",
                                hover_color="#5a6268", pressed_color="#545b62")  # Gri
        no_fire_buttons_layout.addWidget(self.clear_no_fire_zone_button)
        fire_control_layout.addLayout(no_fire_buttons_layout)

        self.fire_control_group_box.setLayout(fire_control_layout)
        right_layout.addWidget(self.fire_control_group_box)
        self.fire_control_group_box.setVisible(True)

        # --- MANUEL YÖN KONTROLÜ Grup Kutusu ---
        self.direct_manual_control_group_box = QGroupBox("Doğrudan Manuel Kontrol")
        self.direct_manual_control_group_box.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold; border: 2px solid white; border-radius: 8px; padding: 5px;")
        direct_manual_control_layout = QVBoxLayout()

        grid_layout = QVBoxLayout()

        self.up_button = QPushButton("Yukarı", self)
        self.down_button = QPushButton("Aşağı", self)
        self.left_button = QPushButton("Sol", self)
        self.right_button = QPushButton("Sağ", self)

        self.apply_button_style(self.up_button, font_size=16, padding=8)
        self.apply_button_style(self.down_button, font_size=16, padding=8)
        self.apply_button_style(self.left_button, font_size=16, padding=8)
        self.apply_button_style(self.right_button, font_size=16, padding=8)

        grid_layout.addWidget(self.up_button, alignment=Qt.AlignCenter)

        h_layout_lr = QHBoxLayout()
        h_layout_lr.addWidget(self.left_button)
        h_layout_lr.addWidget(self.right_button)
        grid_layout.addLayout(h_layout_lr)

        grid_layout.addWidget(self.down_button, alignment=Qt.AlignCenter)

        direct_manual_control_layout.addLayout(grid_layout)
        self.direct_manual_control_group_box.setLayout(direct_manual_control_layout)
        right_layout.addWidget(self.direct_manual_control_group_box)
        self.direct_manual_control_group_box.setVisible(False)

        right_layout.addSpacerItem(
            QSpacerItem(10, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        main_layout.addLayout(right_layout, 2)

        self.setLayout(main_layout)
        print("HATA AYIKLAMA: Düzen ayarlandı.")

        # Kamera ve Zamanlayıcı Ayarları
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        # Zamanlayıcı 'start_camera' içinde kamera başarıyla başlatıldıktan sonra başlayacaktır.

        self.frame_counter = 0

        # --- FPS ölçümü (hafif) ---
        self._fps = 0.0
        self._fps_accum = 0
        self._fps_last_report_t = time.perf_counter()
        self._fps_update_interval = 0.30  # saniye; 0.2-0.5 arası ideal, gürültüyü azaltır

        # --- DÜZELTME 2: Başlangıç model durumunu doğru değişkenle ayarla ---
        self.yolo_ready = (yolo_model_task12 is not None)
        self.model_is_tensorrt = (yolo_model_task12 == "tensorrt")

        self.crosshair_movable = False
        self.crosshair_fixed_center = True
        self.crosshair_x = 0
        self.crosshair_y = 0
        self.active_task = None

        self.camera_label.setMouseTracking(True)
        self.camera_label.mouseMoveEvent = self.mouse_move_event
        self.camera_label.mousePressEvent = self.mouse_press_event
        print("HATA AYIKLAMA: Kamera ve Zamanlayıcı ayarları yapılandırıldı.")

        # Sinyal Bağlantıları
        print("HATA AYIKLAMA: Sinyaller bağlanıyor.")
        self.task1_button.clicked.connect(self.task1)
        self.task2_button.clicked.connect(self.task2)
        # GÜNCELLENDİ: task3 butonu yeni fonksiyona bağlandı
        self.task3_button.clicked.connect(self.setup_task3)
        self.manual_control_mode_button.clicked.connect(self.set_full_manual_mode)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_task_button.clicked.connect(self.cancel_task)
        self.fire_weapon_button.clicked.connect(self.fire_weapon)
        self.connect_rpi_button.clicked.connect(self.connect_rpi_threaded)
        self.reset_angles_button.clicked.connect(self.reset_rpi_angles)
        self.apply_no_fire_zone_button.clicked.connect(self.apply_no_fire_zone_settings)
        self.clear_no_fire_zone_button.clicked.connect(self.clear_no_fire_zone_settings)
        # YENİ: Aşama 3 başlangıç butonu sinyali
        self.task3_start_button.clicked.connect(self.start_task3_engagement)

        self.movement_states = {
            'yaw_left': False,
            'yaw_right': False,
            'pitch_up': False,
            'pitch_down': False
        }
        self.manual_yaw_direction = 0
        self.manual_pitch_direction = 0

        self.manual_movement_timer = QTimer(self)
        # AYARLANDI: Daha akıcı manuel hareket için manuel hareket zamanlayıcısı daha sık tetiklenecek
        self.manual_movement_timer.timeout.connect(self._continuously_update_motor_position)

        # Manuel hareket butonu sinyal bağlantıları güncellendi
        self.up_button.pressed.connect(lambda: self._handle_manual_button_press('pitch_up'))
        self.up_button.released.connect(lambda: self._set_movement_state('pitch_up', False))
        self.down_button.pressed.connect(lambda: self._handle_manual_button_press('pitch_down'))
        self.down_button.released.connect(lambda: self._set_movement_state('pitch_down', False))
        self.left_button.pressed.connect(lambda: self._handle_manual_button_press('yaw_left'))
        self.left_button.released.connect(lambda: self._set_movement_state('yaw_left', False))
        self.right_button.pressed.connect(lambda: self._handle_manual_button_press('yaw_right'))
        self.right_button.released.connect(lambda: self._set_movement_state('yaw_right', False))
        print("HATA AYIKLAMA: Sinyaller bağlandı.")

        QCoreApplication.instance().aboutToQuit.connect(self.close_event)
        print("HATA AYIKLAMA: HavaSavunmaArayuz başlatma tamamlandı.")

        # AYARLARI YÜKLE
        self.load_settings()

    def save_settings(self):
        """Mevcut ayarları bir JSON dosyasına kaydeder."""
        settings = {
            "MAX_MISSING_FRAMES": self.MAX_MISSING_FRAMES,
            "MAX_REACQUISITION_DISTANCE_PIXELS": self.MAX_REACQUISITION_DISTANCE_PIXELS,
            "KP_YAW": self.KP_YAW,
            "KI_YAW": self.KI_YAW,
            "KD_YAW": self.KD_YAW,
            "angle_command_minimum_interval": self.angle_command_minimum_interval,
            "KP_PITCH": self.KP_PITCH,
            "KI_PITCH": self.KI_PITCH,
            "KD_PITCH": self.KD_PITCH,
            "MAX_OUTPUT_DEGREE": self.MAX_OUTPUT_DEGREE,
            "MIN_OUTPUT_DEGREE_THRESHOLD": self.MIN_OUTPUT_DEGREE_THRESHOLD,
            "aiming_tolerance": self.aiming_tolerance,
            "DEGREES_PER_PIXEL_YAW": self.DEGREES_PER_PIXEL_YAW,
            "DEGREES_PER_PIXEL_PITCH": self.DEGREES_PER_PIXEL_PITCH,
            "pid_output_deadband_degree": self.pid_output_deadband_degree,
            "feedforward_yaw_gain": self.feedforward_yaw_gain,
            "feedforward_pitch_gain": self.feedforward_pitch_gain,
            "feedforward_velocity_threshold_px_s": self.feedforward_velocity_threshold_px_s,
            "manual_step_size": self.manual_step_size,
            "velocity_update_error_threshold_px": self.velocity_update_error_threshold_px,
        }
        try:
            with open("ayarlar.json", "w") as f:
                json.dump(settings, f, indent=4)
            print("Ayarlar başarıyla ayarlar.json dosyasına kaydedildi.")
        except Exception as e:
            print(f"Ayarları kaydederken hata oluştu: {e}")

    def load_settings(self):
        """Ayarları bir JSON dosyasından yükler."""
        try:
            with open("ayarlar.json", "r") as f:
                settings = json.load(f)
                for key, value in settings.items():
                    setattr(self, key, value)
                print("Ayarlar başarıyla ayarlar.json dosyasından yüklendi.")
        except FileNotFoundError:
            print("ayarlar.json dosyası bulunamadı. Varsayılan ayarlar kullanılacak.")
        except Exception as e:
            print(f"Ayarları yüklerken hata oluştu: {e}")

    def apply_button_style(self, button, font_size=30, padding=20, bg_color="#808080", hover_color="#A9A9A9",
                           pressed_color="#696969"):
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: 2px solid {bg_color};
                border-radius: 8px;
                padding: {padding}px;
                font-size: {font_size}px;
                font-weight: bold;
                box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                width: 100%;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border: 2px solid {hover_color};
                box-shadow: 4px 4px 6px rgba(0, 0, 0, 0.7);
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
                border: 2px solid {bg_color};
                box-shadow: inset 2px 2px 4px rgba(0, 0, 0, 0.7);
            }}
        """)
        button.clicked.connect(button.clearFocus)

    def open_settings_dialog(self):
        """Ayarlar panelini açar; her parametre için mevcut değeri gösterir ve 'Değer Ata' ile canlı günceller."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Ayarlar")
        dlg.setModal(True)
        dlg.setStyleSheet("background-color: #111; color: white; font-size: 13px;")
        main_layout = QVBoxLayout(dlg)

        # Parametre tanımları: (görünen_ad, attr_adı, tip)
        params = [
            ("MAX_MISSING_FRAMES", "MAX_MISSING_FRAMES", int),
            ("MAX_REACQUISITION_DISTANCE_PIXELS", "MAX_REACQUISITION_DISTANCE_PIXELS", int),

            ("KP_YAW", "KP_YAW", float),
            ("KI_YAW", "KI_YAW", float),
            ("KD_YAW", "KD_YAW", float),

            ("KP_PITCH", "KP_PITCH", float),
            ("KI_PITCH", "KI_PITCH", float),
            ("KD_PITCH", "KD_PITCH", float),

            ("velocity_update_error_threshold_px", "velocity_update_error_threshold_px", float),

            ("angle_command_minimum_interval", "angle_command_minimum_interval", float),

            ("MAX_OUTPUT_DEGREE (°)", "MAX_OUTPUT_DEGREE", float),
            ("MIN_OUTPUT_DEGREE_THRESHOLD (°)", "MIN_OUTPUT_DEGREE_THRESHOLD", float),

            ("aiming_tolerance (px)", "aiming_tolerance", int),

            ("DEGREES_PER_PIXEL_YAW", "DEGREES_PER_PIXEL_YAW", float),
            ("DEGREES_PER_PIXEL_PITCH", "DEGREES_PER_PIXEL_PITCH", float),

            ("pid_output_deadband_degree (°)", "pid_output_deadband_degree", float),

            ("feedforward_yaw_gain", "feedforward_yaw_gain", float),
            ("feedforward_pitch_gain", "feedforward_pitch_gain", float),
            ("feedforward_velocity_threshold_px_s", "feedforward_velocity_threshold_px_s", float),

            ("manual_step_size (°)", "manual_step_size", float),
        ]

        # Satır satır UI: Etiket | Mevcut Değer | QLineEdit | "Değer Ata"
        for display_name, attr_name, ptype in params:
            row = QHBoxLayout()
            label = QLabel(display_name + ":")
            label.setStyleSheet("color: #ddd;")
            row.addWidget(label)

            current_val = getattr(self, attr_name, None)
            current_lbl = QLabel(f"Mevcut: {current_val}")
            current_lbl.setStyleSheet("color: #aaa;")
            row.addWidget(current_lbl)

            edit = QLineEdit(dlg)
            edit.setPlaceholderText(str(current_val) if current_val is not None else "")
            edit.setStyleSheet("background-color: white; color: black;")
            edit.setText(str(current_val) if current_val is not None else "")
            row.addWidget(edit)

            set_btn = QPushButton("Değer Ata", dlg)
            self.apply_button_style(set_btn, font_size=12, padding=6, bg_color="#007bff",
                                    hover_color="#0069d9", pressed_color="#0062cc")
            # Lambda ile atr adı, tip, edit ve mevcut değer label'ını yakala
            set_btn.clicked.connect(lambda _=False, n=attr_name, t=ptype, e=edit, cl=current_lbl:
                                    self._assign_param(n, t, e, cl))
            row.addWidget(set_btn)

            main_layout.addLayout(row)

        # Kapat butonu
        close_btn = QPushButton("Kapat", dlg)
        self.apply_button_style(close_btn, font_size=14, padding=8, bg_color="#6c757d",
                                hover_color="#5a6268", pressed_color="#545b62")
        close_btn.clicked.connect(dlg.accept)
        main_layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dlg.resize(720, 520)
        dlg.exec_()

    def _assign_param(self, attr_name, ptype, edit_widget, current_label_widget):
        """Edit'teki değeri tipine göre parse eder, self.attr_name'e atar ve ekrandaki 'Mevcut' etiketi günceller."""
        raw = edit_widget.text().strip()
        if raw == "":
            QMessageBox.warning(self, "Hatalı Girdi", f"{attr_name} için bir değer girin.")
            return
        try:
            if ptype is int:
                # int yazılmasa bile 10.0 gibi gelirse güvenli çevir
                val = int(float(raw))
            else:
                val = float(raw)
        except ValueError:
            QMessageBox.warning(self, "Hatalı Girdi", f"{attr_name} için sayı girin.")
            return

        setattr(self, attr_name, val)
        current_label_widget.setText(f"Mevcut: {val}")
        # İsteğe bağlı küçük bir bilgi mesajı:
        # QMessageBox.information(self, "Güncellendi", f"{attr_name} = {val}")

        # DEĞİŞİKLİKTEN SONRA AYARLARI KAYDET
        self.save_settings()

    def _update_status_label(self, message):
        """Tekrarlayan hata mesajlarını önlemek için bekleme süreli yuva."""
        current_time = time.time()
        if message != self.last_status_message or (
                current_time - self.last_status_time > self.status_message_cooldown_interval):
            self.status_label.setText(message)
            self.last_status_message = message
            self.last_status_time = current_time

    def connect_rpi_threaded(self):
        """RPi'ye bağlantıyı ayrı bir iş parçacığında başlatır."""
        if not self.rpi_thread.is_connected:
            self._update_status_label("Durum: Raspberry Pi'ye bağlanılıyor...")
            if not self.rpi_thread.isRunning():
                self.rpi_thread.start()
        else:
            self._update_status_label("Durum: Zaten Raspberry Pi'ye bağlı.")

    def _update_rpi_connection_status(self, is_connected):
        """RPi bağlantı durumuna göre UI'yi güncelleyen yuva."""
        self.rpi_connection_status = is_connected
        if is_connected:
            self.connect_rpi_button.setEnabled(False)
            self._update_status_label("Durum: Raspberry Pi'ye Bağlandı!")
        else:
            self.connect_rpi_button.setEnabled(True)
            self._update_status_label("Durum: Raspberry Pi bağlantısı kesildi.")

    def _update_current_angles(self, yaw, pitch):
        """RPiCommunicator'dan alınan mevcut açıları güncelleyen yuva."""
        self.current_yaw_angle = yaw
        self.current_pitch_angle = pitch
        self.update_info_panel(f"Mevcut Yaw: {self.current_yaw_angle:.1f}°, Pitch: {self.current_pitch_angle:.1f}°")

    def _process_rpi_response(self, response_data):
        """RPiCommunicator'dan gelen genel yanıtları işler."""
        if response_data.get("status") == "ok":
            if response_data.get("action") == "fire":
                self._update_status_label("Durum: Ateşleme Başarılı!")
                print("Ateşleme Başarılı!")
                if self.active_task in ['task1', 'task2', 'task3']:
                    self.target_destroyed = True
                    self.waiting_for_new_engagement_command = True
                    self._update_status_label("Durum: Hedef yok edildi. Yeni angajman bekleniyor...")
                    self.target_info_label.setText("Hedef Bilgisi: Yok Edildi.")
                    self.reset_pid_state()  # PID durumunu sıfırla
                    print("HATA AYIKLAMA: Ateşlemeden sonra PID ve hedef bilgisi sıfırlandı.")
            elif response_data.get("action") == "reset_angles":
                self._update_status_label("Durum: Taret açıları Raspberry Pi'de (0,0) olarak sıfırlandı.")
                self.update_info_panel("Taret açıları sıfırlandı: Yaw 0.0°, Pitch 0.0°")
                print("Taret açıları Raspberry Pi'de (0,0) olarak sıfırlandı.")
                self.reset_pid_state()  # PID durumunu sıfırla
                print("HATA AYIKLAMA: Açı sıfırlamadan sonra PID bilgisi sıfırlandı.")
            elif response_data.get("action") in ["set_angles", "move_by_direction", "set_proportional_angles_delta"]:
                pass  # Açılar zaten angles_update_signal aracılığıyla güncellendi
            elif response_data.get("action") == "test_motor_movement":
                self._update_status_label(f"Durum: Motor Testi: {response_data.get('message')}")
        else:
            error_message = response_data.get('message', 'Bilinmeyen Hata')
            self._update_status_label(f"Hata: RPi yanıtı: {error_message}")
            print(f"RPi'den hata yanıtı (durum: {response_data.get('status')}): {error_message}")

    def send_command_to_rpi(self, command_dict):
        """RPiCommunicator'ın kuyruğuna bir komut ekler."""
        if self.rpi_thread.is_connected:
            self.rpi_thread.command_queue.put(command_dict)
            return True
        else:
            self.status_update_signal.emit("Hata: Raspberry Pi'ye bağlı değil, komut gönderilemedi.")
            return False

    def send_proportional_move_command(self, delta_yaw, delta_pitch):
        """
        Taretin belirli orantılı derece miktarlarında hareket etmesi için Raspberry Pi'ye bir komut gönderir.
        Bu PID kontrolü için kullanılır.
        """
        if not self.rpi_thread.is_connected:
            self._update_status_label(
                "Hata: Raspberry Pi'ye bağlı değil, orantılı hareket komutu gönderilemedi.")
            return False

        current_time = time.time()
        if current_time - self.last_angle_command_send_time < self.angle_command_minimum_interval:
            return False

        command = {"action": "set_proportional_angles_delta", "delta_yaw": delta_yaw, "delta_pitch": delta_pitch}
        self.last_angle_command_send_time = current_time
        return self.send_command_to_rpi(command)

    def start_camera(self):
        try:
            print("Kamera başlatılıyor...")
            camera_indices = [1, 2, 3, 4]
            self.capture = None

            self.cam_nominal_fps = 0.0 #kamera fps'i tutmak için

            for index in camera_indices:
                print(f"cv2.CAP_DSHOW arka ucu ile kamera {index} deneniyor...")
                try:
                    temp_capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    if temp_capture.isOpened():
                        self.capture = temp_capture
                        print(f"Kamera {index} (cv2.CAP_DSHOW) başarıyla açıldı.")
                        break
                    else:
                        print(f"Kamera {index} (cv2.CAP_DSHOW) açılamadı.")
                except Exception as e:
                    print(f"Kamera {index} (cv2.CAP_DSHOW) açılırken hata: {e}")

            if not self.capture or not self.capture.isOpened():
                print("HATA: Hiçbir kamera açılamadı! Lütfen kamera bağlantısını veya numarasını kontrol edin.")
                self._update_status_label("Durum: Kamera Açılamadı!")
                self.capture = None
                return

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Kamera başarıyla başlatıldı. Çözünürlük Ayarlandı: {actual_width}x{actual_height}")

            self.cam_nominal_fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 0.0

            self._update_status_label("Durum: Kamera Başlatıldı.")
            self.timer.start(int(self.angle_command_minimum_interval * 1000))

            if hasattr(self, 'start_button'):
                self.start_button.setEnabled(False)
            if hasattr(self, 'stop_button'):
                self.stop_button.setEnabled(True)
            self.update_info_panel("BUKREK Hava Savunma Sistemi Başlatıldı")

        except cv2.error as e:
            print(f"OpenCV hatası oluştu: {e}")
            self._update_status_label(f"Durum: OpenCV Hatası: {e.msg[:50]}...")
            self.capture = None
        except Exception as e:
            print(f"Kamera başlatılırken beklenmedik hata oluştu: {e}")
            traceback.print_exc()
            self._update_status_label(f"Durum: Hata: {str(e)[:50]}...")
            self.capture = None

    def stop_camera(self):
        if self.capture and self.capture.isOpened():
            self.capture.release()
            self.capture = None
            self.timer.stop()
            self._update_status_label("Durum: Kamera Durduruldu.")
            self.target_info_label.setText("Hedef Bilgisi: Yok")
            self.kcf_active = False
            self.kcf_bbox = None
            self.active_task = None
            self.camera_label.clear()
            self._stop_all_manual_movement()
            self.reset_pid_state()  # PID durumunu sıfırla
            self.is_target_active = False
            self.is_aimed_at_target = False
            self.target_destroyed = False
            self.waiting_for_new_engagement_command = False
            self.target_lost_time = 0.0
            self.current_tracked_target_class = None
            self.current_tracked_target_bbox = None
            self.missing_frames = 0  # Sıfırla
            print("HATA AYIKLAMA: Kamera durduruldu, tüm PID ve hedef bilgisi sıfırlandı.")
        else:
            self._update_status_label("Durum: Kamera zaten kapalı.")

    def cancel_task(self):
        self.active_task = None
        self.kcf_bbox = None
        self.kcf_active = False
        self.target_destroyed = False
        self.waiting_for_new_engagement_command = False
        self.target_lost_time = 0.0
        self.current_tracked_target_class = None
        self.current_tracked_target_bbox = None
        self.missing_frames = 0  # Sıfırla
        # YENİ: Aşama 3 ile ilgili durumları sıfırla
        self.is_ready_to_engage_from_qr = False
        self.qr_degrees = {}
        self.current_qr_char = None
        self.task3_settings_group_box.setVisible(False)

        self._update_status_label("Durum: Görev durduruldu.")
        self.target_info_label.setText("Hedef Bilgisi: Yok")
        self._stop_all_manual_movement()
        self.movement_restricted_yaw_start = 0
        self.movement_restricted_yaw_end = 0
        self.active_engagement_target_color = None
        self.active_engagement_target_shape = None
        self.fire_control_group_box.setVisible(True)
        self.direct_manual_control_group_box.setVisible(False)
        self.is_target_active = False
        self.is_aimed_at_target = False
        self.reset_pid_state()  # PID durumunu sıfırla
        print("HATA AYIKLAMA: Görev iptal edildi, tüm PID ve hedef bilgisi sıfırlandı.")

    def reset_pid_state(self):
        """PID kontrol değişkenlerini sıfırlar."""
        self.integral_yaw = 0.0
        self.last_error_yaw = 0.0
        self.integral_pitch = 0.0
        self.last_error_pitch = 0.0
        self.last_target_x = None
        self.last_target_y = None
        self.last_frame_time = None
        self.last_target_velocity_x = 0.0
        self.last_target_velocity_y = 0.0
        self.current_pid_range = "TEK_SET"
        self.missing_frames = 0  # PID sıfırlanırken missing_frames'i de sıfırla
        # print("HATA AYIKLAMA: PID durumu sıfırlandı.") # Bu mesaj çok sık yazdırıldığı için yorum satırı yapıldı.

    def task1(self):
        self.cancel_task()
        self.active_task = 'task1'
        self.crosshair_movable = False
        self.crosshair_fixed_center = True
        self.task3_settings_group_box.setVisible(False)
        self._update_status_label("Durum: Aşama 1 başlatıldı (Tüm Balonları Takip Et, Manuel Ateş).")
        self.target_info_label.setText("Hedef Bilgisi: Tüm Balonlar.")
        self.kcf_active = False
        self.kcf_bbox = None
        self.target_destroyed = False
        self.waiting_for_new_engagement_command = True
        self.target_lost_time = 0.0
        self.current_tracked_target_class = None
        self.current_tracked_target_bbox = None
        self.missing_frames = 0  # Sıfırla
        self.movement_restricted_yaw_start = 0
        self.movement_restricted_yaw_end = 0
        self.fire_control_group_box.setVisible(True)
        self.direct_manual_control_group_box.setVisible(False)
        self.is_target_active = True
        self.is_aimed_at_target = False
        # --- DÜZELTME 3: Görev değiştiğinde model türü bayrağını güncelle ---
        self.model_is_tensorrt = (yolo_model_task12 == "tensorrt")
        print("HATA AYIKLAMA: Aşama 1 başlatıldı, PID ve hedef bilgisi sıfırlandı.")

    def task2(self):
        self.cancel_task()
        self.active_task = 'task2'
        self.crosshair_movable = False
        self.crosshair_fixed_center = True
        self.task3_settings_group_box.setVisible(False)
        self._update_status_label("Durum: Aşama 2 başlatıldı (Kırmızı Balonu Takip Et, Otomatik Ateş).")
        self.target_info_label.setText("Hedef Bilgisi: Kırmızı Balon.")
        self.kcf_active = False
        self.kcf_bbox = None
        self.target_destroyed = False
        self.waiting_for_new_engagement_command = True
        self.target_lost_time = 0.0
        self.current_tracked_target_class = None
        self.current_tracked_target_bbox = None
        self.missing_frames = 0  # Sıfırla
        self.movement_restricted_yaw_start = 0
        self.movement_restricted_yaw_end = 0
        self.fire_control_group_box.setVisible(True)
        self.direct_manual_control_group_box.setVisible(False)
        self.is_target_active = True
        self.is_aimed_at_target = False
        # --- DÜZELTME 3: Görev değiştiğinde model türü bayrağını güncelle ---
        self.model_is_tensorrt = (yolo_model_task12 == "tensorrt")
        print("HATA AYIKLAMA: Aşama 2 başlatıldı, PID ve hedef bilgisi sıfırlandı.")

    # YENİ: Aşama 3 ayar panelini gösteren fonksiyon
    def setup_task3(self):
        self.cancel_task()
        self.active_task = 'task3_setup'
        self.task3_settings_group_box.setVisible(True)
        self.fire_control_group_box.setVisible(True)
        self.direct_manual_control_group_box.setVisible(False)
        self._update_status_label("Durum: Aşama 3 - Angajman ayarları bekleniyor.")
        self.target_info_label.setText("Hedef Bilgisi: Yok (Ayar Bekleniyor).")
        self.is_target_active = False  # Henüz hedef takibi aktif değil

    # YENİ: "Angajmanı Al" butonuna basıldığında çalışan fonksiyon
    def start_task3_engagement(self):
        self.cancel_task()
        self.active_task = 'task3'
        self.crosshair_movable = False
        self.crosshair_fixed_center = True
        self.fire_control_group_box.setVisible(True)
        self.direct_manual_control_group_box.setVisible(False)
        self.is_ready_to_engage_from_qr = False

        # --- DÜZELTME 4: Aşama 3 modelini doğru şekilde yükle ve ata ---
        global yolo_model_task3
        if yolo_model_task3 is None:
            # Bu, global trt_... değişkenlerini Aşama 3 modeline göre güncelleyecektir.
            yolo_model_task3 = load_yolo_model(YOLO_MODEL_PATH_TASK3)

        # Tespit fonksiyonunun doğru çalışması için model türü bayrağını güncelle
        self.model_is_tensorrt = (yolo_model_task3 == "tensorrt")

        # Girilen dereceleri al ve kaydet
        try:
            a_degree = float(self.a_input.text())
            b_degree = float(self.b_input.text())
            self.qr_degrees = {'A': a_degree, 'B': b_degree}
            self._update_status_label("Durum: Aşama 3 Ayarları kaydedildi. QR kodu bekleniyor...")
            self.target_info_label.setText("Hedef Bilgisi: QR Kod.")
            self.is_target_active = True  # Kare işleme döngüsünü başlat
            self.waiting_for_new_engagement_command = True
            self.reset_pid_state()
            print(f"HATA AYIKLAMA: Aşama 3 angajman başlatıldı. A:{a_degree}, B:{b_degree} dereceleri kaydedildi.")
        except ValueError:
            self._update_status_label("Hata: Lütfen geçerli sayısal değerler girin.")
            self.cancel_task()
            return

    def set_full_manual_mode(self):
        self.cancel_task()
        self.active_task = 'full_manual'
        self.task3_settings_group_box.setVisible(False)
        self._update_status_label("Durum: Tam Manuel Kontrol Modu Aktif.")
        self.target_info_label.setText("Hedef Bilgisi: Yok (Manuel).")
        self.crosshair_movable = True
        self.crosshair_fixed_center = False
        self.fire_control_group_box.setVisible(True)
        self.direct_manual_control_group_box.setVisible(True)
        try:
            self._start_manual_movement_timer()
        except Exception as e:
            print(f"HATA: set_full_manual_mode sırasında _start_manual_movement_timer'da çökme: {e}")
            traceback.print_exc()
            self._update_status_label(f"Hata: Manuel mod başlatma hatası: {str(e)[:50]}...")
        self.is_target_active = False
        self.is_aimed_at_target = False
        print("HATA AYIKLAMA: Tam Manuel Kontrol Modu başlatıldı, PID ve hedef bilgisi sıfırlandı.")

    def _handle_manual_button_press(self, direction_key):
        """
        Manuel hareket butonu basıldığında çağrılır.
        Hareket durumunu ayarlar ve sürekli hareket için zamanlayıcıyı başlatır.
        """
        if self.active_task != 'full_manual':
            return

        self.movement_states[direction_key] = True
        self._update_manual_directions_from_states()
        self._start_manual_movement_timer()
        self._update_status_label(f"Durum: Manuel hareket etkin: {direction_key}.")

    def _set_movement_state(self, direction_key, is_pressed):
        """Manuel hareket butonlarının basılı olup olmadığını günceller."""
        if self.active_task != 'full_manual':
            return

        self.movement_states[direction_key] = is_pressed
        self._update_manual_directions_from_states()
        self._start_manual_movement_timer()

    def _start_manual_movement_timer(self):
        """Manuel hareket zamanlayıcısını başlatır veya durdurur."""
        if any(self.movement_states.values()):
            if not self.manual_movement_timer.isActive():
                self.manual_movement_timer.start(10)  # AYARLANDI: 10ms (daha akıcı manuel hareket için)
        else:
            if self.manual_movement_timer.isActive():
                self.manual_movement_timer.stop()
            self._update_status_label("Durum: Manuel hareket durduruldu.")
            if self.rpi_thread.is_connected:
                # Hareket durduğunda Raspberry Pi'ye sıfır hareket komutu gönder
                self.send_command_to_rpi(
                    {"action": "move_by_direction", "yaw_direction": 0, "pitch_direction": 0, "degrees_to_move": 0})

    def _update_manual_directions_from_states(self):
        """
        movement_states sözlüğüne göre genel yaw ve pitch hareket yönlerini hesaplar.
        """
        self.manual_yaw_direction = 0
        self.manual_pitch_direction = 0

        if self.movement_states['yaw_left']:
            self.manual_yaw_direction = -1
        elif self.movement_states['yaw_right']:
            self.manual_yaw_direction = 1

        if self.movement_states['pitch_up']:
            self.manual_pitch_direction = 1
        elif self.movement_states['pitch_down']:
            self.manual_pitch_direction = -1

        if self.movement_states['yaw_left'] and self.movement_states['yaw_right']:
            self.manual_yaw_direction = 0
        if self.movement_states['pitch_up'] and self.movement_states['pitch_down']:
            self.manual_pitch_direction = 0

    def _continuously_update_motor_position(self):
        """Manuel hareket modunda motorları sürekli günceller."""
        if self.active_task != 'full_manual' or not self.rpi_thread.is_connected:
            self._stop_all_manual_movement()
            return

        # Yalnızca hareket yönü varsa komut gönder
        if self.manual_yaw_direction != 0 or self.manual_pitch_direction != 0:
            self.send_command_to_rpi(
                {"action": "move_by_direction", "yaw_direction": self.manual_yaw_direction,
                 "pitch_direction": self.manual_pitch_direction, "degrees_to_move": self.manual_step_size})
        else:
            # Hareket yönü yoksa, motorları durdurmak için sıfır komutu gönder (önemli!)
            # Bu, rpi_motor_server'daki manual_move_loop'un motor_fire_module'e 0 adım göndermesini sağlar.
            self.send_command_to_rpi(
                {"action": "move_by_direction", "yaw_direction": 0, "pitch_direction": 0, "degrees_to_move": 0})

    def _stop_all_manual_movement(self):
        """Tüm manuel hareketleri durdurur ve zamanlayıcıyı sıfırlar."""
        for key in self.movement_states:
            self.movement_states[key] = False
        if self.manual_movement_timer.isActive():
            self.manual_movement_timer.stop()
        self.manual_yaw_direction = 0
        self.manual_pitch_direction = 0
        self._update_status_label("Durum: Manuel hareket durduruldu.")
        if self.rpi_thread.is_connected:
            # Motorları durdurmak için sıfır hareket komutu gönder
            self.send_command_to_rpi(
                {"action": "move_by_direction", "yaw_direction": 0, "pitch_direction": 0, "degrees_to_move": 0})

    def _preprocess_frame_for_yolo(self, frame):
        """Çerçeveyi YOLO modeli için ön işler."""
        if frame is None:
            print("HATA (_preprocess_frame_for_yolo): Giriş çerçevesi boş.")
            return None
        img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def _process_yolo_output(self, output, img_width, img_height, classes_list):
        """YOLO çıktısını işler ve sınırlayıcı kutuları döndürür."""
        boxes = []
        confidences = []
        class_ids = []

        predictions = np.squeeze(output).T

        scores = np.max(predictions[:, 4:], axis=1)
        valid_predictions = predictions[scores > CONF_THRESHOLD]
        valid_scores = scores[scores > CONF_THRESHOLD]

        for i in range(len(valid_predictions)):
            row = valid_predictions[i]
            score = valid_scores[i]
            class_id = np.argmax(row[4:])

            if score < CONF_THRESHOLD:
                continue

            center_x, center_y, w, h = row[:4]
            x = int((center_x - w / 2) * img_width / IMG_WIDTH)
            y = int((center_y - h / 2) * img_height / IMG_HEIGHT)
            width = int(w * img_width / IMG_WIDTH)
            height = int(h * img_height / IMG_HEIGHT)  # Düzeltildi: Bunun IMG_HEIGHT kullandığından emin ol

            boxes.append([x, y, width, height])
            confidences.append(float(score))
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = [boxes[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]
            filtered_class_ids = [class_ids[i] for i in indices]
            return filtered_boxes, filtered_confidences, filtered_class_ids
        return [], [], []

    def process_yolo_detection(self, frame, model, classes_list):
        """YOLOv11 modelini kullanarak nesne tespiti yapar."""
        global trt_context, trt_inputs, trt_outputs, trt_bindings, trt_stream, trt_output_binding_idx

        if model is None:
            # print("HATA AYIKLAMA (process_yolo_detection): YOLO modeli hazır değil.")
            return []

        if frame is None or frame.size == 0:
            print("HATA (process_yolo_detection): Giriş çerçevesi boş veya geçersiz.")
            return []

        try:
            input_image = self._preprocess_frame_for_yolo(frame)
            if input_image is None:
                print("HATA (process_yolo_detection): Ön işlenmiş görüntü boş.")
                return []

            outputs = []
            if self.model_is_tensorrt and isinstance(model, str) and model == "tensorrt":
                # TensorRT ile çıkarım yap
                if trt_context is None:
                    print("HATA (process_yolo_detection): TensorRT bağlamı başlatılmadı.")
                    return []
                if trt_input_binding_idx is None or trt_output_binding_idx is None:
                    print("HATA (process_yolo_detection): TensorRT giriş/çıkış bağlama indeksleri ayarlanmadı.")
                    return []

                # Giriş verilerini ana bilgisayardan cihaza kopyala
                np.copyto(trt_inputs[0]['host'], input_image.flatten())
                cuda.memcpy_htod_async(trt_inputs[0]['device'], trt_inputs[0]['host'], trt_stream)

                # execute_v2 (senkron) kullanarak çıkarım yap
                # Bu, execute_async / execute_async_v2 mevcut değilse bir yedektir
                trt_context.execute_v2(trt_bindings)

                # Çıkış verilerini cihazdan ana bilgisayara kopyala
                cuda.memcpy_dtoh_async(trt_outputs[0]['host'], trt_outputs[0]['device'], trt_stream)
                trt_stream.synchronize()

                # Saklanan çıkış bağlama indeksini kullanarak çıkış şekline eriş
                outputs = [trt_outputs[0]['host'].reshape(
                    trt_engine.get_tensor_shape(trt_engine.get_tensor_name(trt_output_binding_idx)))]
            else:
                # ONNX Runtime ile çıkarım yap
                outputs = model.run([output_name], {input_name: input_image})

            img_height, img_width, _ = frame.shape
            outputs_np = outputs[0]  # _process_yolo_output için bir numpy dizisi olduğundan emin ol
            boxes, confidences, class_ids = self._process_yolo_output(outputs_np, img_width, img_height, classes_list)

            detections = []
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                class_name = classes_list[class_ids[i]] if class_ids[i] < len(classes_list) else "Bilinmeyen"
                detections.append({
                    'bbox': (x, y, w, h),
                    'score': confidences[i],
                    'class_name': class_name
                })
            return detections
        except Exception as e:
            print(f"HATA (process_yolo_detection): Model işleme hatası: {e}")
            traceback.print_exc()
            self._update_status_label(f"Hata: Model tespit hatası: {str(e)[:50]}...")
            return []

    def update_info_panel(self, text):
        self.info_label.setText(f"<h2 style='color: white; text-align: center;'>{text}</h2>")

    def apply_no_fire_zone_settings(self):
        """Girişlerden ateşsiz bölge değerlerini okur ve uygular."""
        try:
            start_yaw = float(self.no_fire_start_input.text())
            end_yaw = float(self.no_fire_end_input.text())

            self.no_fire_yaw_start = start_yaw
            self.no_fire_yaw_end = end_yaw
            self._update_status_label(
                f"Durum: Ateşsiz Bölge Güncellendi: Yaw [{start_yaw:.1f}°, {end_yaw:.1f}°]")
            print(f"HATA AYIKLAMA: Ateşsiz Bölge Güncellendi: Yaw [{start_yaw:.1f}°, {end_yaw:.1f}°]")
        except ValueError:
            self._update_status_label("Hata: Lütfen ateşsiz bölge için geçerli sayısal değerler girin.")
        except Exception as e:
            self._update_status_label(f"Hata: Ateşsiz bölge ayarları uygulanırken sorun: {e}")

    def clear_no_fire_zone_settings(self):
        """Ateşsiz bölge ayarlarını temizler."""
        self.no_fire_yaw_start = 0.0
        self.no_fire_yaw_end = 0.0
        self.no_fire_start_input.setText("0.0")
        self.no_fire_end_input.setText("0.0")
        self._update_status_label("Durum: Ateşsiz Bölge Temizlendi.")
        print("HATA AYIKLAMA: Ateşsiz Bölge Temizlendi.")

    def send_angle_command(self, yaw, pitch):
        """Taretin belirli bir mutlak açıya gitmesi için Raspberry Pi'ye bir komut gönderir."""
        if not self.rpi_thread.is_connected:
            self._update_status_label("Hata: Raspberry Pi'ye bağlı değil, açı komutu gönderilemedi.")
            return False

        current_time = time.time()
        if current_time - self.last_angle_command_send_time < 0.1:
            return False

        command = {"action": "set_angles", "yaw": yaw, "pitch": pitch}
        self.last_angle_command_send_time = current_time
        return self.send_command_to_rpi(command)

    def fire_weapon(self):
        if not self.rpi_thread.is_connected:
            self._update_status_label("Hata: Raspberry Pi'ye bağlı değil, ateş edilemez.")
            return
        try:
            current_time = time.time()
            if current_time - self.last_fire_time < self.fire_cooldown_interval:
                self._update_status_label("Uyarı: Ateşleme denemesi çok hızlı. Lütfen bekleyin.")
                return

            print(f"HATA AYIKLAMA (fire_weapon): Ateşleme denemesi. Mevcut Yaw: {self.current_yaw_angle:.1f}°, "
                  f"Ateşsiz Bölge: [{self.no_fire_yaw_start:.1f}°, {self.no_fire_yaw_end:.1f}°]")

            if self.is_in_no_fire_zone(self.current_yaw_angle):
                self._update_status_label("Uyarı: Ateşsiz bölgedesiniz! Ateşleme engellendi.")
                print("Uyarı: Ateşsiz bölgedesiniz! Ateşleme engellendi.")
                return

            if self.active_task == 'full_manual':
                print("Manuel kontrol modunda ateşleme komutu gönderiliyor...")
                self.send_command_to_rpi({"action": "fire"})
                self.last_fire_time = current_time
                return

            if self.active_task in ['task2', 'task3']:
                if not self.is_aimed_at_target:
                    self._update_status_label("Uyarı: Hedef nişan alma toleransı dışında! Ateşleme engellendi.")
                    print("Uyarı: Hedef nişan alma toleransı dışında! Ateşleme engellendi.")
                    return

            print("Ateşleme komutu gönderiliyor...")
            self.send_command_to_rpi({"action": "fire"})
            self.last_fire_time = current_time
        except Exception as e:
            print(f"HATA: Ateşleme sırasında beklenmedik hata: {e}")
            traceback.print_exc()
            self._update_status_label(f"Hata: Ateşleme hatası: {str(e)[:50]}...")

    def reset_rpi_angles(self):
        """Taret açılarını Yaw ve Pitch 0.0'a sıfırlamak için Raspberry Pi'ye komut gönderir."""
        if not self.rpi_thread.is_connected:
            self._update_status_label("Hata: Raspberry Pi'ye bağlı değil, açılar sıfırlanamaz.")
            return

        print("Açıları sıfırlama komutu gönderiliyor...")
        self.send_command_to_rpi({"action": "reset_angles"})

    def close_event(self, event):
        """Uygulama kapatıldığında bağlantıyı keser."""
        print("HATA AYIKLAMA: Uygulama kapanıyor (close_event tetiklendi)...")

        # UYGULAMA KAPANIRKEN AYARLARI KAYDET
        self.save_settings()

        self.stop_camera()
        if self.rpi_thread.isRunning():
            self.rpi_thread.request_stop()
            self.rpi_thread.wait(2000)
            if self.rpi_thread.isRunning():
                print("UYARI: RPiCommunicator iş parçacığı zamanında kapanmadı.")
        event.accept()

    def update_frame(self):
        display_frame = None
        try:
            # print("HATA AYIKLAMA (update_frame): Kare güncelleme döngüsü başlatıldı.")
            if self.capture is None or not self.capture.isOpened():
                # print("HATA (update_frame): Kamera yakalama nesnesi eksik veya açık değil.")
                self._update_status_label("Hata: Kamera açık değil. Lütfen başlatın.")
                self.timer.stop()
                return

            ret, frame = self.capture.read()

            # --- FPS sayaç güncellemesi ---
            now_perf = time.perf_counter()
            self._fps_accum += 1
            if (now_perf - self._fps_last_report_t) >= self._fps_update_interval:
                self._fps = self._fps_accum / (now_perf - self._fps_last_report_t)
                self._fps_accum = 0
                self._fps_last_report_t = now_perf

            if not ret or frame is None or frame.size == 0:
                print("HATA (update_frame): Kare okunamadı, boş veya geçersiz boyut. Kamera durduruluyor.")
                self._update_status_label("Hata: Kameradan kare okunamadı.")
                self.stop_camera()
                return

            display_frame = frame.copy()

            current_frame_time = time.time()
            original_h, original_w = frame.shape[:2]
            center_x_frame, center_y_frame = original_w // 2, original_h // 2

            h, w, ch = display_frame.shape
            bytes_per_line = ch * w
            center_x_display, center_y_display = w // 2, h // 2

            # Artı işaretini çiz
            crosshair_color = (0, 255, 0)
            crosshair_size = 10
            cv2.line(display_frame, (center_x_display - crosshair_size, center_y_display),
                     (center_x_display + crosshair_size, center_y_display),
                     crosshair_color, 2)
            cv2.line(display_frame, (center_x_display, center_y_display - crosshair_size),
                     (center_x_display, center_y_display + crosshair_size),
                     crosshair_color, 2)

            cv2.putText(display_frame, f"FPS: {self._fps:4.1f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            self.update_info_panel(
                f"Mevcut Yaw: {self.current_yaw_angle:.1f}°, Pitch: {self.current_pitch_angle:.1f}°")

            detections = []
            current_target_bbox_for_pid = None
            detected_class_status = None

            # print("HATA AYIKLAMA (update_frame): YOLO tespiti başlatılıyor.")

            # --- DÜZELTME 5: Aktif göreve göre doğru model tanıtıcısını ve sınıf listesini seç ---
            current_yolo_model = None
            current_classes = None
            if self.active_task == 'task3':
                current_yolo_model = yolo_model_task3
                current_classes = CLASSES_TASK3
            elif self.active_task in ['task1', 'task2']:
                current_yolo_model = yolo_model_task12
                current_classes = CLASSES

            if self.is_target_active and current_yolo_model is not None:
                detections = self.process_yolo_detection(display_frame, current_yolo_model, current_classes)
                # print(f"HATA AYIKLAMA (update_frame): YOLO {len(detections)} tespit buldu.")

                # GÜNCELLENDİ: Aşama 3 Mantığı
                if self.active_task == 'task3' and self.waiting_for_new_engagement_command and not self.is_ready_to_engage_from_qr:
                    # Aşama 3: QR kodunu oku ve açıya dön
                    # self.process_tracking_to_home_position() # Önce başlangıç konumuna dön
                    data, bbox_qr, _ = self.qr_detector.detectAndDecode(display_frame)
                    if data and data in self.qr_degrees:
                        # QR koduna en yakın hedefi bul
                        closest_target_to_qr = None
                        min_dist_to_qr = float('inf')
                        qr_center_x = bbox_qr[0][0][0] + (bbox_qr[0][2][0] - bbox_qr[0][0][0]) / 2

                        for det in detections:
                            det_center_x = det['bbox'][0] + det['bbox'][2] / 2
                            dist = abs(det_center_x - qr_center_x)
                            if dist < min_dist_to_qr:
                                min_dist_to_qr = dist
                                closest_target_to_qr = det

                        if closest_target_to_qr:
                            self.current_qr_char = data
                            self.current_tracked_target_class = closest_target_to_qr['class_name']
                            target_yaw_from_qr = self.qr_degrees[self.current_qr_char]

                            self.send_angle_command(target_yaw_from_qr, self.current_pitch_angle)
                            self.is_ready_to_engage_from_qr = True
                            self.waiting_for_new_engagement_command = False

                            self._update_status_label(
                                f"Durum: QR Kodu '{data}' okundu. Hedef '{self.current_tracked_target_class}' kilitlendi. Açıya dönülüyor: {target_yaw_from_qr}°")
                            self.target_info_label.setText(
                                f"Hedef: {self.current_tracked_target_class}. Açıya dönülüyor.")
                        else:
                            self._update_status_label(f"Uyarı: QR Kodu okundu ancak yanında hedef bulunamadı.")

                    elif data and data not in self.qr_degrees:
                        self._update_status_label(f"Uyarı: QR Kodu okundu ancak geçerli değil: '{data}'")

                elif self.current_tracked_target_class is not None and not self.target_destroyed:
                    # print(f"HATA AYIKLAMA: Kilitli hedef '{self.current_tracked_target_class}' takip ediliyor.")
                    closest_locked_detection = None
                    min_locked_distance = float('inf')

                    # Son bilinen veya tahmin edilen hedef merkezini al
                    last_tracked_center_x = self.current_tracked_target_bbox[0] + self.current_tracked_target_bbox[
                        2] // 2 if self.current_tracked_target_bbox else center_x_frame
                    last_tracked_center_y = self.current_tracked_target_bbox[1] + self.current_tracked_target_bbox[
                        3] // 2 if self.current_tracked_target_bbox else center_y_frame

                    # Eğer kayıp kareler varsa, yeniden edinme için tahmini merkezi kullan
                    if self.missing_frames > 0 and self.last_target_x is not None and self.last_target_y is not None:
                        time_since_last_known = current_frame_time - self.last_frame_time
                        predicted_x_for_reacq = self.last_target_x + self.last_target_velocity_x * time_since_last_known
                        predicted_y_for_reacq = self.last_target_y + self.last_target_velocity_y * time_since_last_known
                        last_tracked_center_x = int(predicted_x_for_reacq)
                        last_tracked_center_y = int(predicted_y_for_reacq)
                        # print(
                        #     f"HATA AYIKLAMA: Yeniden edinme için tahmini merkez kullanılıyor: ({last_tracked_center_x}, {last_tracked_center_y})")

                    for det in detections:
                        # Sadece mevcut kilitli hedef sınıfıyla eşleşenleri kontrol et
                        if det['class_name'] == self.current_tracked_target_class:
                            x, y, det_w, det_h = det['bbox']
                            det_center_x = x + det_w // 2
                            det_center_y = y + det_h // 2
                            distance = np.sqrt(
                                (det_center_x - last_tracked_center_x) ** 2 + (
                                        det_center_y - last_tracked_center_y) ** 2)

                            # Yalnızca son bilinen/tahmin edilen konumdan belirli bir mesafe içindeki tespitleri dikkate al
                            if distance < min_locked_distance and distance < self.MAX_REACQUISITION_DISTANCE_PIXELS:
                                min_locked_distance = distance
                                closest_locked_detection = det

                    if closest_locked_detection:
                        current_target_bbox_for_pid = closest_locked_detection['bbox']
                        detected_class_status = closest_locked_detection['class_name']
                        self.target_info_label.setText(
                            f"Hedef: YOLO Takip Ediyor ({detected_class_status}).")
                        self.target_lost_time = 0.0
                        self.missing_frames = 0  # Hedef bulunduğunda sayacı sıfırla
                        # current_tracked_target_bbox'u yeni bulunan tespitle güncelle
                        self.current_tracked_target_bbox = current_target_bbox_for_pid

                        # Hedef hızını yeni tespitten güncelle
                        center_x_display = w // 2  # w = display_frame.shape[1]
                        target_center_x = current_target_bbox_for_pid[0] + current_target_bbox_for_pid[2] // 2
                        error_pixel = abs(target_center_x - center_x_display)

                        # SADECE hedef merkeze yeterince yakınsa (yani taret büyük bir manevra yapmıyorsa) hızı GÜNCELLE
                        if self.last_target_x is not None and self.last_frame_time is not None and error_pixel < self.velocity_update_error_threshold_px:
                            delta_time_for_velocity = current_frame_time - self.last_frame_time
                            if delta_time_for_velocity > 0:
                                raw_velocity_x = (target_center_x - self.last_target_x) / delta_time_for_velocity
                                raw_velocity_y = (current_target_bbox_for_pid[1] + current_target_bbox_for_pid[
                                    3] // 2 - self.last_target_y) / delta_time_for_velocity

                                alpha = 0.4
                                self.last_target_velocity_x = (alpha * raw_velocity_x) + (
                                            1 - alpha) * self.last_target_velocity_x
                                self.last_target_velocity_y = (alpha * raw_velocity_y) + (
                                            1 - alpha) * self.last_target_velocity_y

                        # Son konum ve zamanı her zaman güncelle (bir sonraki hız hesaplaması için)
                        self.last_target_x = target_center_x
                        self.last_target_y = current_target_bbox_for_pid[1] + current_target_bbox_for_pid[3] // 2
                        self.last_frame_time = current_frame_time
                        # print(f"HATA AYIKLAMA: Kilitli hedef yeniden edinildi. Missing frames sıfırlandı.")

                    else:  # closest_locked_detection is None
                        # print(
                        #     f"HATA AYIKLAMA: Kilitli hedef sınıfı için YOLO tespiti bulunamadı ({self.current_tracked_target_class}) veya çok uzakta.")
                        self.missing_frames += 1  # Hedef bulunamadığında sayacı artır

                        if self.missing_frames <= self.MAX_MISSING_FRAMES:
                            self._update_status_label(
                                f"Durum: Hedef kaybedildi, tahminle takip etmeye çalışılıyor ({self.MAX_MISSING_FRAMES - self.missing_frames} kare kaldı).")
                            self.target_info_label.setText("Hedef: Takip Kayboldu. Tahminle hareket ediyor.")

                            if self.last_target_x is not None and self.last_frame_time is not None and self.current_tracked_target_bbox is not None:
                                # Sadece geçerli bir son bilinen konum ve bbox boyutu varsa tahmin et
                                time_since_last_detection = current_frame_time - self.last_frame_time
                                predicted_x = self.last_target_x + self.last_target_velocity_x * time_since_last_detection
                                predicted_y = self.last_target_y + self.last_target_velocity_y * time_since_last_detection

                                _, _, w_last, h_last = self.current_tracked_target_bbox  # Son takip edilen bbox'un boyutlarını kullan
                                current_target_bbox_for_pid = (
                                    int(predicted_x - w_last / 2), int(predicted_y - h_last / 2), w_last, h_last)
                                # print(f"HATA AYIKLAMA: Hedef tahmin edildi: X={predicted_x:.1f}, Y={predicted_y:.1f}")
                            else:
                                current_target_bbox_for_pid = None  # Tahmin için yeterli veri yok
                                # print("HATA AYIKLAMA: Tahmin için yeterli veri yok, PID hedefi yok.")
                        else:
                            # Hedef gerçekten kayboldu
                            print(
                                f"HATA AYIKLAMA: Hedef takibi {self.MAX_MISSING_FRAMES} kare boyunca kaybedildi. Hedef kalıcı olarak kaybedildi.")
                            current_target_bbox_for_pid = None
                            self.current_tracked_target_class = None
                            self.current_tracked_target_bbox = None  # Gerçekten kaybolduğunda bbox'u temizle
                            self.target_destroyed = True
                            self.waiting_for_new_engagement_command = True
                            self.target_info_label.setText("Hedef: Takip Kayboldu. Yeni hedef aranıyor.")
                            self.reset_pid_state()
                            print(
                                "HATA AYIKLAMA: Hedef kaybedildi ve zaman aşımı doldu, PID ve hedef bilgisi sıfırlandı.")
                elif self.waiting_for_new_engagement_command or self.current_tracked_target_class is None:
                    # print("HATA AYIKLAMA: Yeni hedef edinme/yeniden edinme süreci başlatıldı.")
                    candidate_target = None
                    minimum_distance = float('inf')

                    if self.active_task == 'task1':
                        for det in detections:
                            x, y, det_w, det_h = det['bbox']
                            det_center_x = x + det_w // 2
                            det_center_y = y + det_h // 2
                            distance = np.sqrt(
                                (det_center_x - center_x_frame) ** 2 + (det_center_y - center_y_frame) ** 2)
                            if distance < minimum_distance:
                                minimum_distance = distance
                                candidate_target = det
                        if candidate_target:
                            self.target_info_label.setText(
                                f"Hedef Bilgisi: {candidate_target['class_name']} algılandı.")
                            detected_class_status = candidate_target['class_name']
                            self._update_status_label("Durum: Aşama 1 - Hedef kilitlendi.")
                        else:
                            self.target_info_label.setText("Hedef Bilgisi: Balon algılanmadı.")
                            self._update_status_label("Durum: Yeni hedef bekleniyor...")

                    elif self.active_task == 'task2':
                        red_balloons = [d for d in detections if d['class_name'] == 'red_balloon']
                        for det in red_balloons:
                            x, y, det_w, det_h = det['bbox']
                            det_center_x = x + det_w // 2
                            det_center_y = y + det_h // 2
                            distance = np.sqrt(
                                (det_center_x - center_x_frame) ** 2 + (det_center_y - center_y_frame) ** 2)
                            if distance < minimum_distance:
                                minimum_distance = distance
                                candidate_target = det
                        if candidate_target:
                            self.target_info_label.setText(f"Hedef Bilgisi: Kırmızı Balon algılandı.")
                            detected_class_status = "red_balloon"
                            self._update_status_label("Durum: Aşama 2 - Düşman hedef kilitlendi.")
                        else:
                            self.target_info_label.setText(f"Hedef Bilgisi: Kırmızı Balon Yok.")
                            self._update_status_label("Durum: Yeni hedef bekleniyor...")

                    # GÜNCELLENDİ: Bu blok artık kullanılmıyor, Aşama 3 mantığı yukarıda ele alındı
                    # elif self.active_task == 'task3' and self.is_ready_to_engage_from_qr:
                    #     pass

                    if candidate_target:
                        self.current_tracked_target_class = candidate_target['class_name']
                        self.current_tracked_target_bbox = candidate_target[
                            'bbox']  # Yeni hedef kilitlendiğinde bbox'u ayarla
                        self.target_destroyed = False
                        self.waiting_for_new_engagement_command = False
                        self.target_lost_time = 0.0
                        self.missing_frames = 0  # Yeni hedef kilitlendiğinde sayacı sıfırla
                        current_target_bbox_for_pid = candidate_target['bbox']
                        self.reset_pid_state()  # Yeni hedef için PID durumunu sıfırla
                        # Yeni hedef için last_target_x, last_target_y, last_frame_time'ı başlat
                        self.last_target_x = current_target_bbox_for_pid[0] + current_target_bbox_for_pid[2] // 2
                        self.last_target_y = current_target_bbox_for_pid[1] + current_target_bbox_for_pid[3] // 2
                        self.last_frame_time = current_frame_time
                        self.last_target_velocity_x = 0.0  # Yeni hedef için hızı sıfırla
                        self.last_target_velocity_y = 0.0  # Yeni hedef için hızı sıfırla

                        print(
                            f"HATA AYIKLAMA: Yeni hedef kilitlendi: {self.current_tracked_target_class}. PID sıfırlandı.")
                    else:
                        current_target_bbox_for_pid = None
                        self.current_tracked_target_class = None
                        self.current_tracked_target_bbox = None
                        if self.target_destroyed and self.waiting_for_new_engagement_command:
                            # print("HATA AYIKLAMA: PID sıfırlandı (Hedef Yok Edildi/Bekleniyor).")
                            self.target_info_label.setText("Hedef Bilgisi: Yok Edildi. Yeni angajman bekleniyor.")
                        self.target_lost_time = 0.0

            # print("HATA AYIKLAMA (update_frame): YOLO tespitleri çiziliyor.")
            for det in detections:
                x, y, w_det, h_det = [int(v) for v in det['bbox']]
                yolo_draw_color = (0, 255, 0)

                if self.current_tracked_target_class and det[
                    'class_name'] == self.current_tracked_target_class:
                    if current_target_bbox_for_pid and det['bbox'] == current_target_bbox_for_pid:
                        yolo_draw_color = (0, 0, 255)  # Kilitli hedef kırmızı
                    else:
                        yolo_draw_color = (0, 255, 255)  # Diğer aynı sınıftan hedefler sarı
                else:
                    yolo_draw_color = (0, 255, 0)  # Diğer hedefler yeşil

                # Çerçeve kalınlığı 1'den 2'ye çıkarıldı
                cv2.rectangle(display_frame, (x, y), (x + w_det, y + h_det), yolo_draw_color,
                              2)  # Kalınlık 2 olarak ayarlandı
                cv2.putText(display_frame, f"YOLO: {det['class_name']} ({det['score']:.2f})", (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, yolo_draw_color, 1)

            # print("HATA AYIKLAMA (update_frame): PID kontrolü başlatılıyor.")
            if current_target_bbox_for_pid and not self.target_destroyed:
                x_pid, y_pid, w_pid, h_pid = [int(v) for v in current_target_bbox_for_pid]
                target_center_x = x_pid + w_pid // 2
                target_center_y = y_pid + h_pid // 2
                self.process_tracking(target_center_x, target_center_y, display_frame, 0, current_frame_time)
            else:
                self.reset_pid_state()
                self.is_aimed_at_target = False
                if self.active_task not in ['full_manual']:
                    if not self.target_destroyed and not self.waiting_for_new_engagement_command:
                        self.target_info_label.setText("Hedef Bilgisi: Yok.")
                # GÜNCELLENDİ: Aşama 3'te hedef yok edildiğinde veya kaybolduğunda ana konuma dön
                if self.active_task == 'task3' and self.target_destroyed and self.waiting_for_new_engagement_command:
                    self.process_tracking_to_home_position()
                    self.is_ready_to_engage_from_qr = False  # Yeni QR için hazırla
                    self.current_qr_char = None
                    self.current_tracked_target_class = None

            # print("HATA AYIKLAMA (update_frame): Göreve özel durum güncellemeleri.")
            if self.active_task == 'task1':
                if current_target_bbox_for_pid and not self.target_destroyed:
                    self._update_status_label("Durum: Aşama 1 - Hedefe Nişan Alıyor (Manuel Ateş).")
                elif self.target_destroyed:
                    self._update_status_label("Durum: Aşama 1 - Hedef yok edildi. Yeni hedef bekleniyor.")
                else:
                    self._update_status_label("Durum: Aşama 1 - Hedef Aranıyor (Manuel Ateş).")
            elif self.active_task == 'task2':
                if current_target_bbox_for_pid and detected_class_status == "red_balloon" and self.is_aimed_at_target and not self.target_destroyed:
                    try:
                        self.fire_weapon()
                        self._update_status_label("Durum: Aşama 2 - Düşman yok edildi! (Otonom)")
                        print("HATA AYIKLAMA: Düşman hedef algılandı ve ateşlendi (Aşama 2 - Otonom)!")
                    except Exception as e:
                        print(f"HATA: Aşama 2 otomatik ateşleme sırasında hata: {e}")
                        traceback.print_exc()
                        self._update_status_label(f"Hata: Aşama 2 ateşleme hatası: {str(e)[:50]}...")
                elif current_target_bbox_for_pid and detected_class_status == 'blue_balloon':
                    self._update_status_label("Durum: Aşama 2 - Dost hedef algılandı, ateşleme engellendi.")
                elif current_target_bbox_for_pid and not self.target_destroyed:
                    self._update_status_label("Durum: Aşama 2 - Düşman hedefe nişan alıyor...")
                elif self.target_destroyed:
                    self._update_status_label("Durum: Aşama 2 - Hedef yok edildi. Yeni hedef bekleniyor.")
                else:
                    self._update_status_label("Durum: Aşama 2 - Hedef Yok.")
            elif self.active_task == 'task3':
                # GÜNCELLENDİ: Aşama 3 durum mesajları
                if current_target_bbox_for_pid and self.is_aimed_at_target and not self.target_destroyed:
                    if not self.is_in_no_fire_zone(self.current_yaw_angle):
                        try:
                            self.fire_weapon()
                            # self.target_destroyed = True # Bu fire_weapon yanıtında ayarlanacak
                            # self.waiting_for_new_engagement_command = True
                            # self.is_ready_to_engage_from_qr = False
                            self._update_status_label("Durum: Aşama 3 - Hedef yok edildi! (Otonom)")
                            print("HATA AYIKLAMA: Hedef algılandı ve ateşlendi (Aşama 3 - Otonom)!")
                        except Exception as e:
                            print(f"HATA: Aşama 3 otomatik ateşleme sırasında hata: {e}")
                            traceback.print_exc()
                            self._update_status_label(f"Hata: Aşama 3 ateşleme hatası: {str(e)[:50]}...")
                    else:
                        self._update_status_label("Durum: Aşama 3 - Ateşsiz bölgede ateşleme engellendi!")
                elif current_target_bbox_for_pid and not self.target_destroyed:
                    self._update_status_label(
                        "Durum: Aşama 3 - Angajman hedefine nişan alıyor...")
                elif self.target_destroyed:
                    self._update_status_label("Durum: Aşama 3 - Hedef yok edildi. Ana konuma dönülüyor...")
                elif not self.is_ready_to_engage_from_qr:
                    self._update_status_label("Durum: Aşama 3 - Angajman için QR Kodu bekleniyor.")
                else:
                    self._update_status_label("Durum: Aşama 3 - Angajman Hedefi Aranıyor.")
            elif self.active_task == 'full_manual':
                if self.is_in_no_fire_zone(self.current_yaw_angle):
                    self._update_status_label(
                        f"Durum: Tam Manuel Kontrol Modu - Ateşsiz Bölgede! Yaw: {self.current_yaw_angle:.1f}°, Pitch: {self.current_pitch_angle:.1f}°")
                else:
                    self._update_status_label(
                        f"Durum: Tam Manuel Kontrol Modu - Yaw: {self.current_yaw_angle:.1f}°, Pitch: {self.current_pitch_angle:.1f}°")
                self.target_info_label.setText("Hedef Bilgisi: Kullanıcı Kontrollü.")
            elif self.active_task != 'task3_setup':
                self._update_status_label("Durum: Hazır.")
                self.target_info_label.setText("Hedef Bilgisi: Yok.")

            # print("HATA AYIKLAMA (update_frame): Ekran üzerinde çerçeve gösteriliyor.")
            self._display_frame(display_frame)

            self.frame_counter += 1
            # print("HATA AYIKLAMA (update_frame): Kare güncelleme döngüsü tamamlandı.")
        except Exception as main_loop_error:
            print(f"KRİTİK HATA: update_frame ana döngüsünde beklenmedik hata: {main_loop_error}")
            traceback.print_exc()
            self._update_status_label(f"KRİTİK HATA: UI Güncelleme Hatası: {str(main_loop_error)[:50]}...")
            self.stop_camera()

    def is_in_no_fire_zone(self, current_yaw_angle):
        """
        Mevcut yaw açısının ateşsiz bölge içinde olup olmadığını kontrol eder, 0/360 derece etrafında dönmeyi yönetir.
        Mevcut yaw açısı ve bölge girişlerinin derece cinsinden olduğunu varsayar.
        """
        zone_start = self.no_fire_yaw_start
        zone_end = self.no_fire_yaw_end

        # print(
        #     f"HATA AYIKLAMA (is_in_no_fire_zone): current_yaw_angle={current_yaw_angle:.1f}°'nin [{zone_start:.1f}°, {zone_end:.1f}°] bölgesi içinde olup olmadığı kontrol ediliyor.")

        normalized_yaw = (current_yaw_angle + 180) % 360 - 180

        if zone_start <= zone_end:
            is_within_zone = self.no_fire_yaw_start <= normalized_yaw <= self.no_fire_yaw_end
        else:
            is_within_zone = normalized_yaw >= self.no_fire_yaw_start or normalized_yaw <= self.no_fire_yaw_end

        return is_within_zone

    def is_in_movement_restricted_zone(self, target_yaw_angle):
        """Aşama 3 için kısıtlı hareket bölgesini kontrol eder, 0/360 derece etrafında dönmeyi yönetir."""
        is_within_zone = False
        start = self.movement_restricted_yaw_start
        end = self.movement_restricted_yaw_end

        target_yaw_angle = target_yaw_angle % 360
        if target_yaw_angle < 0:
            target_yaw_angle += 360

        start_normalized = start % 360
        if start_normalized < 0:
            start_normalized += 360

        end_normalized = end % 360
        if end_normalized < 0:
            end_normalized += 360

        if start_normalized <= end_normalized:
            if start_normalized <= target_yaw_angle <= end_normalized:
                is_within_zone = True
        else:
            if target_yaw_angle >= start_normalized or target_yaw_angle <= end_normalized:
                is_within_zone = True
        return is_within_zone

    def _display_frame(self, frame):
        """Çerçeveyi QLabel'de gösterir."""
        try:
            if frame is None or frame.size == 0:
                print("HATA (_display_frame): Görüntülenecek çerçeve boş veya geçersiz.")
                return

            try:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                if not rgb_image.flags['C_CONTIGUOUS']:
                    rgb_image = np.ascontiguousarray(rgb_image)
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap_obj = qt_image.scaled(self.camera_label.width(), self.camera_label.height(),
                                             Qt.KeepAspectRatio)
                self.camera_label.setPixmap(QPixmap.fromImage(pixmap_obj))
            except Exception as e:
                print(f"HATA (_display_frame - Görüntü Dönüşümü): Çerçeve görüntülenirken hata: {e}")
                traceback.print_exc()
                self._update_status_label(f"Hata: Görüntü Dönüşüm Hatası: {str(e)[:50]}...")

        except Exception as e:
            print(f"HATA (_display_frame - Genel): Çerçeve görüntülenirken genel hata: {e}")
            traceback.print_exc()
            self._update_status_label(f"Hata: Görüntüleme Hatası: {str(e)[:50]}...")

    def process_tracking(self, target_x, target_y, frame, target_area_unused, current_frame_time):
        """
        Hedef merkez koordinatlarına göre PID kontrolü kullanarak taret hareketini ayarlar.
        PID çıktısı doğrudan motorlara iletilir.
        """
        if not self.rpi_thread.is_connected or self.active_task == 'full_manual' or self.target_destroyed:
            return

        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2

        error_yaw_pixel = target_x - center_x
        error_pitch_pixel = target_y - center_y

        # --- ÖN NİŞAN: hedef merkezini ileriye projekte et ---
        # mevcut merkez: target_x, target_y
        pred_x, pred_y = target_x, target_y
        if self.enable_aim_lead:
            # hedefin son tahmini hızı (px/s) zaten tutuluyor
            vx = float(self.last_target_velocity_x)
            vy = float(self.last_target_velocity_y)

            # çok yavaşsa lead uygulama (titreşimi büyütmesin)
            if abs(vx) >= self.aim_lead_vel_thresh or abs(vy) >= self.aim_lead_vel_thresh:
                # lead süresi: otomatik ya da sabit
                if self.aim_lead_mode == "auto":
                    # fps + pipeline + komut aralığına göre kaba gecikme kestirimi
                    fps = self.cam_nominal_fps if getattr(self, 'cam_nominal_fps', 0.0) > 0 else 30.0
                    cam_dt = 1.0 / max(1.0, fps)
                    pipe_dt = 0.02  # tipik işleme gecikmesi (isteğe göre değiştir)
                    cmd_dt = float(self.angle_command_minimum_interval)  # zaten mevcut
                    dt = cam_dt + pipe_dt + cmd_dt  # örn ~ 50–80 ms
                else:
                    dt = float(self.aim_lead_time_s)

                # öngörü noktası (px): mevcut merkez + v*dt
                lead_dx = vx * dt * self.aim_lead_gain
                lead_dy = vy * dt * self.aim_lead_gain
                pred_x = target_x + lead_dx
                pred_y = target_y + lead_dy

                # ek titreşimi azaltmak için basit LPF
                if self._lead_px_prev is None:
                    self._lead_px_prev = (pred_x, pred_y)
                else:
                    ax = self.aim_lead_lpf_alpha
                    prevx, prevy = self._lead_px_prev
                    pred_x = ax * pred_x + (1 - ax) * prevx
                    pred_y = ax * pred_y + (1 - ax) * prevy
                    self._lead_px_prev = (pred_x, pred_y)

                # çerçeve sınırları ve maksimum lead kısıtı
                h, w = frame.shape[0], frame.shape[1]
                pred_x = max(0, min(int(pred_x), w - 1))
                pred_y = max(0, min(int(pred_y), h - 1))
                # ayrıca çok uç lead’leri bastır
                if abs(pred_x - target_x) > self.aim_lead_max_px:
                    pred_x = target_x + np.sign(pred_x - target_x) * self.aim_lead_max_px
                if abs(pred_y - target_y) > self.aim_lead_max_px:
                    pred_y = target_y + np.sign(pred_y - target_y) * self.aim_lead_max_px

                # görsel: öngörü noktasına küçük artı çiz (debug)
                try:
                    cv2.drawMarker(frame, (int(pred_x), int(pred_y)), (0, 255, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
                except Exception:
                    pass

            # Hata artık öngörü noktasına göre hesaplanacak
            error_yaw_pixel = int(pred_x) - center_x
            error_pitch_pixel = int(pred_y) - center_y

        error_yaw_degree = error_yaw_pixel * self.DEGREES_PER_PIXEL_YAW
        error_pitch_degree = error_pitch_pixel * self.DEGREES_PER_PIXEL_PITCH

        self.is_aimed_at_target = abs(error_yaw_pixel) <= self.aiming_tolerance and \
                                  abs(error_pitch_pixel) <= self.aiming_tolerance

        if self.is_aimed_at_target and self.active_task in ['task2', 'task3'] and not self.target_destroyed:
            current_time = time.time()
            if current_time - self.last_fire_time < self.fire_cooldown_interval:
                pass
            else:
                try:
                    # self.fire_weapon() # Otomatik ateşleme update_frame'e taşındı
                    self._update_status_label("Durum: Hedefe nişan alındı, otomatik ateş bekleniyor...")
                except Exception as e:
                    print(f"HATA: Otomatik ateşleme sırasında hata: {e}")
                    traceback.print_exc()
                    self._update_status_label(f"Hata: Otomatik ateşleme hatası: {str(e)[:50]}...")

        delta_time = current_frame_time - self.pid_update_time
        if delta_time <= 0:  # Sıfıra bölme hatasını önlemek için
            return
        self.pid_update_time = current_frame_time

        # --- İLERİ BESLEME HESAPLAMASINI AKTİF ET ---
        feedforward_yaw = self.last_target_velocity_x * self.DEGREES_PER_PIXEL_YAW * self.feedforward_yaw_gain
        feedforward_pitch = self.last_target_velocity_y * self.DEGREES_PER_PIXEL_PITCH * self.feedforward_pitch_gain

        if abs(self.last_target_velocity_x) < self.feedforward_velocity_threshold_px_s:
            feedforward_yaw = 0.0
        if abs(self.last_target_velocity_y) < self.feedforward_velocity_threshold_px_s:
            feedforward_pitch = 0.0

        self.integral_yaw += error_yaw_degree * delta_time
        self.integral_yaw = max(min(self.integral_yaw, 20.0), -20.0)

        derivative_yaw = (error_yaw_degree - self.last_error_yaw) / delta_time if delta_time > 0 else 0
        output_yaw = (self.KP_YAW * error_yaw_degree +
                      self.KI_YAW * self.integral_yaw +
                      self.KD_YAW * derivative_yaw +
                      feedforward_yaw)
        self.last_error_yaw = error_yaw_degree

        self.integral_pitch += error_pitch_degree * delta_time
        self.integral_pitch = max(min(self.integral_pitch, 20.0), -20.0)

        derivative_pitch = (error_pitch_degree - self.last_error_pitch) / delta_time if delta_time > 0 else 0
        output_pitch = (self.KP_PITCH * error_pitch_degree +
                        self.KI_PITCH * self.integral_pitch +
                        self.KD_PITCH * derivative_pitch +
                        feedforward_pitch)
        self.last_error_pitch = error_pitch_degree

        # PID çıktısı için ölü bant
        if abs(error_yaw_degree) < self.pid_output_deadband_degree:
            output_yaw = 0.0
            self.integral_yaw = 0.0
        if abs(error_pitch_degree) < self.pid_output_deadband_degree:
            output_pitch = 0.0
            self.integral_pitch = 0.0

        # Minimum hareket eşiği (PID çıktısı bu eşiğin altındaysa, hareket yok)
        if 0 < abs(output_yaw) < self.MIN_OUTPUT_DEGREE_THRESHOLD:
            output_yaw = 0.0  # Hareket etmemek için sıfırla
        if 0 < abs(output_pitch) < self.MIN_OUTPUT_DEGREE_THRESHOLD:
            output_pitch = 0.0  # Hareket etmemek için sıfırla

        # Çıkışları maksimum dereceye sınırla
        output_yaw = max(min(output_yaw, self.MAX_OUTPUT_DEGREE), -self.MAX_OUTPUT_DEGREE)
        output_pitch = max(min(output_pitch, self.MAX_OUTPUT_DEGREE), -self.MAX_OUTPUT_DEGREE)

        if self.missing_frames > 0:
            damping_factor = 0.5  # Tepkiyi %50 oranında azalt. Bu değeri ayarlayabilirsiniz.
            output_yaw *= damping_factor
            output_pitch *= damping_factor

            # Ayrıca, hatalı tahminlerle integralin birikmesini önlemek için sıfırla.
            self.integral_yaw = 0.0
            self.integral_pitch = 0.0

        # Hareket kısıtlı bölge kontrolü (Aşama 3 için)
        # Eğer hareket kısıtlı bölgeye girerse, o eksendeki hareketi sıfırla
        if self.active_task == 'task3':
            # Tahmini yeni açıları hesapla
            predicted_yaw_after_move = self.current_yaw_angle + output_yaw
            predicted_pitch_after_move = self.current_pitch_angle + output_pitch  # Şu anda kullanılmıyor

            if self.is_in_movement_restricted_zone(predicted_yaw_after_move):
                output_yaw = 0.0  # Yaw hareketini engelle
                print("Uyarı: Hedef Yaw açısı kısıtlı hareket bölgesinde! Yaw hareketi engellendi.")

            # Pitch için benzer bir kontrol eklenebilir (eğer pitch kısıtlamaları varsa)
            # if self.is_in_movement_restricted_zone_pitch(predicted_pitch_after_move):
            #     output_pitch = 0.0

        # Yalnızca sıfır olmayan hareket varsa komut gönder
        if output_yaw != 0.0 or output_pitch != 0.0:
            # print(
            #     f"PID Hata Ayıklama (Orantılı): Hata Yaw (px): {error_yaw_pixel}, Hata Yaw (°): {error_yaw_degree:.2f}, Çıkış Yaw (°): {output_yaw:.2f}, "
            #     f"Hata Pitch (px): {error_pitch_pixel}, Hata Pitch (°): {error_pitch_degree:.2f}, Çıkış Pitch (°): {output_pitch:.2f}")
            # print(f"Orantılı Hareket Gönderildi: Delta Yaw: {output_yaw:.2f}, Delta Pitch: {output_pitch:.2f}")

            # Yeni orantılı hareket komutu gönder
            self.send_proportional_move_command(output_yaw, output_pitch)
        else:
            # print(
            #     f"HATA AYIKLAMA (process_tracking): Hedef ölü bant içinde veya minimum eşiğin altında. Hata Yaw: {error_yaw_pixel}px ({error_yaw_degree:.2f}°), Pitch: {error_pitch_pixel}px ({error_pitch_degree:.2f}°). Hareket komutu gönderilmedi.")
            self.target_info_label.setText(
                f"Hedef: Nişan Alındı. Hata: Yaw {error_yaw_pixel}px, Pitch {error_pitch_pixel}px")

        self.last_target_x = target_x
        self.last_target_y = target_y
        self.last_frame_time = current_frame_time

        self.target_info_label.setText(
            f"Hedef: Takip Ediliyor. Hata: Yaw {error_yaw_pixel}px, Pitch {error_pitch_pixel}px")

    def process_tracking_to_home_position(self):
        """
        Taretin tanımlı bir 'ana' konumuna dönmesini sağlar.
        QR kodu kaybedildiğinde Aşama 3'te kullanılır. Orantılı PID kullanır.
        """
        if not self.rpi_thread.is_connected or self.active_task == 'full_manual':
            return

        current_yaw, current_pitch = self.current_yaw_angle, self.current_pitch_angle

        target_yaw_home = self.engagement_home_position_yaw
        target_pitch_home = self.engagement_home_position_pitch

        error_yaw_degree = target_yaw_home - current_yaw
        error_pitch_degree = target_pitch_home - current_pitch

        # En kısa yolu bulmak için hatayı normalleştir
        error_yaw_degree = (error_yaw_degree + 180) % 360 - 180
        error_pitch_degree = (error_pitch_degree + 180) % 360 - 180

        # Eğer zaten ana konumdaysak, hiçbir şey yapma
        if abs(error_yaw_degree) < 0.5 and abs(error_pitch_degree) < 0.5:
            self._update_status_label(f"Durum: Ana konuma ulaşıldı. Yeni QR bekleniyor.")
            return

        current_time = time.time()
        delta_time = current_time - self.pid_update_time
        self.pid_update_time = current_time

        new_pid_range_home = "ANA_KONUM"
        if self.current_pid_range != new_pid_range_home:
            # print(
            #     f"HATA AYIKLAMA: PID aralığı değişti: {self.current_pid_range} -> {new_pid_range_home}. PID sıfırlanıyor (Ana Konum).")
            self.reset_pid_state()
            self.current_pid_range = new_pid_range_home

        actual_Kp_yaw = self.KP_YAW
        actual_Ki_yaw = self.KI_YAW
        actual_Kd_yaw = self.KD_YAW
        actual_Kp_pitch = self.KP_PITCH
        actual_Ki_pitch = self.KI_PITCH
        actual_Kd_pitch = self.KD_PITCH

        self.integral_yaw += error_yaw_degree * delta_time
        self.integral_yaw = max(min(self.integral_yaw, 20.0), -20.0)

        derivative_yaw = (error_yaw_degree - self.last_error_yaw) / delta_time if delta_time > 0 else 0
        output_yaw = actual_Kp_yaw * error_yaw_degree + actual_Ki_yaw * self.integral_yaw + actual_Kd_yaw * derivative_yaw
        self.last_error_yaw = error_yaw_degree

        self.integral_pitch += error_pitch_degree * delta_time
        self.integral_pitch = max(min(self.integral_pitch, 20.0), -20.0)

        derivative_pitch = (error_pitch_degree - self.last_error_pitch) / delta_time if delta_time > 0 else 0
        output_pitch = actual_Kp_pitch * error_pitch_degree + actual_Ki_pitch * self.integral_pitch + actual_Kd_pitch * derivative_pitch
        self.last_error_pitch = error_pitch_degree

        if abs(error_yaw_degree) < self.pid_output_deadband_degree:
            output_yaw = 0.0
            self.integral_yaw = 0.0
        if abs(error_pitch_degree) < self.pid_output_deadband_degree:
            output_pitch = 0.0
            self.integral_pitch = 0.0

        if 0 < abs(output_yaw) < self.MIN_OUTPUT_DEGREE_THRESHOLD:
            output_yaw = 0.0
        if 0 < abs(output_pitch) < self.MIN_OUTPUT_DEGREE_THRESHOLD:
            output_pitch = 0.0

        output_yaw = max(min(output_yaw, self.MAX_OUTPUT_DEGREE), -self.MAX_OUTPUT_DEGREE)
        output_pitch = max(min(output_pitch, self.MAX_OUTPUT_DEGREE), -self.MAX_OUTPUT_DEGREE)

        if output_yaw != 0.0 or output_pitch != 0.0:
            self.send_proportional_move_command(output_yaw, output_pitch)
        else:
            self._update_status_label(
                f"Durum: Ana konum ulaşıldı: Yaw {current_yaw:.1f}°, Pitch {current_pitch:.1f}°")
            # print("Ana konum ulaşıldı.")
            return

        self.target_info_label.setText(
            f"Hedef: Ana Konuma Dönülüyor. Hata: Yaw {error_yaw_degree:.1f}°, Pitch {error_pitch_degree:.1f}°")

    def mouse_move_event(self, event):
        if self.crosshair_movable:
            self.crosshair_x = event.x()
            self.crosshair_y = event.y()
            self.camera_label.update()

    def mouse_press_event(self, event):
        if self.crosshair_movable and event.button() == Qt.LeftButton:
            target_x = event.x()
            target_y = event.y()
            print(f"Fare tıklaması: X={target_x}, Y={target_y}")

            center_x = self.camera_label.width() // 2
            center_y = self.camera_label.height() // 2

            error_yaw_pixel = target_x - center_x
            error_pitch_pixel = target_y - center_y

            delta_yaw_degree = error_yaw_pixel * self.DEGREES_PER_PIXEL_YAW
            delta_pitch_degree = error_pitch_pixel * self.DEGREES_PER_PIXEL_PITCH

            current_yaw, current_pitch = self.current_yaw_angle, self.current_pitch_angle

            target_yaw_angle = current_yaw + delta_yaw_degree
            target_pitch_angle = current_pitch + delta_pitch_degree

            print(f"Manuel Tıklama Hedef Açılar: Yaw {target_yaw_angle:.1f}°, Pitch {target_pitch_angle:.1f}°")
            self.send_angle_command(target_yaw_angle, target_pitch_angle)


if __name__ == '__main__':
    print("HATA AYIKLAMA: __main__ bloğuna girildi.")
    print("HATA AYIKLAMA: YOLO modeli ve global görüntü boyutları ayarlandı.")
    print("HATA AYIKLAMA: QApplication örneği oluşturulmaya çalışılıyor.")
    app = QApplication(sys.argv)
    print("HATA AYIKLAMA: QApplication örneği oluşturuldu.")
    window = HavaSavunmaArayuz()
    print("HATA AYIKLAMA: HavaSavunmaArayuz örneği oluşturuldu.")
    window.showMaximized()
    print("HATA AYIKLAMA: window.showMaximized() çağrıldı. QApplication olay döngüsü başlatılıyor.")
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Uygulama beklenmedik bir hata ile kapandı: {e}")
        traceback.print_exc()
    print("HATA AYIKLAMA: QApplication olay döngüsünden çıkıldı.")
