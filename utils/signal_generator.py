import time
import numpy as np
import requests, random

SEGMENT_LENGTH = 1024
SAMPLING_RATE = 12000  # 采样率 12kHz
API_URL = "http://127.0.0.1:8000/predict/signal"

def generate_signal(length):
    t = np.arange(length) / SAMPLING_RATE
    """
    生成从 0 到 length-1 的整数数组，表示每个采样点的索引。
    然后将这些索引除以采样率，得到每个采样点对应的时间值 t, 作为时间轴
    示例:
    t = [0.00000000, 0.00008333, 0.00016667,...] 
    """
    x= np.random.randn(length)
    signal = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t) + x

    return signal.tolist()

def main():
    while True:
        seg = generate_signal(SEGMENT_LENGTH)
        try:
            resp = requests.post(API_URL, json={"signal": seg}, timeout=1.5)
            res = resp.json()
            print(f"[{time.strftime('%H:%M:%S')}] Pred: {res['class_name']} "
                  f"(idx={res['class_idx']}, conf={res['confidence']})")
        except Exception as e:
            print("Request error:", e)
        time.sleep(1.5)

if __name__ == "__main__":
    main()
