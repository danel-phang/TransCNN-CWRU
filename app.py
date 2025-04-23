import io, logging
from typing import List
import numpy as np
import scipy.io
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "final_model.keras"
CLASS_NAMES = ["正常","内圈故障1","内圈故障2","内圈故障3","外圈故障1","外圈故障2","外圈故障3","滚动体故障1","滚动体故障2","滚动体故障3",]
SEGMENT_LENGTH = 1024
OVERLAP = 0

class Result(BaseModel):
    """
    返回的预测结果格式
    """
    class_idx: int                  # 故障类别索引
    class_name: str                 # 故障类别名称
    confidence: float               # 可信度
    segment_classes: List[int]      # 分块索引列表
    segment_confidences: List[float]# 分块可信度列表

class SignalData(BaseModel):
    signal: List[float]
    
def load_data_from_file(filename):
    content = filename.file.read()
    stream = io.BytesIO(content) 

    if filename.endswith(".mat"):
        mat_data = scipy.io.loadmat(stream)
        keys = [k for k in mat_data.keys() if not k.startswith("__")]
        for key in keys:
            arr = mat_data[key]
            if isinstance(arr, np.ndarray) and arr.size > 10:
                signal = arr
                break
        return signal.ravel()
    elif filename.endswith(".csv"):
        df = pd.read_csv(stream)
        col = df.columns[0]
        return df[col].values
    else:
        raise HTTPException(status_code=415, detail="文件读取失败")

def preprocess_signal(signal: np.ndarray, segment_len=SEGMENT_LENGTH, overlap=OVERLAP):
    """
    对信号进行分段处理，返回分段后的信号数据
    parameters:
        signal: 输入信号数据
        segment_len: 每段的长度 1024
        overlap: 重叠长度 0
    returns:
        np.array(segments) # 分段后的信号数据
    """
    signal = signal.ravel()
    step = segment_len - overlap
    segments = []
    for i in range(0, len(signal) - segment_len + 1, step):
        seg = signal[i : i + segment_len]
        segments.append(seg.reshape(segment_len, 1))
    return np.array(segments)

def predict(signal_data: np.ndarray):
    """
    对信号数据进行预测，返回分类结果
    parameters:
        signal_data: 输入信号数据
    returns:
        Result: 预测结果
    """
    segments = preprocess_signal(signal_data)
    if len(segments) == 0:
        raise HTTPException(status_code=400, detail="信号数据长度不足，无法进行分段")
    x = segments.astype(np.float32)  # 转换为 float32 类型
    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict(x)

    seg_classes = np.argmax(preds, axis=1)
    seg_confs = np.max(preds, axis=1)
    counts = np.bincount(seg_classes)  # 统计每个类别的出现次数
    final_idx = int(counts.argmax())   # 选择出现次数最多的类别
    final_conf = float(seg_confs[seg_classes == final_idx].mean()) # 计算该类别的平均置信度

    return Result(
        class_idx=final_idx,
        class_name=CLASS_NAMES[final_idx] if final_idx < len(CLASS_NAMES) else str(final_idx),
        confidence=round(final_conf, 4),
        segment_classes=seg_classes.tolist(),
        segment_confidences=np.round(seg_confs, 4).tolist(),
    )

@app.post("/predict/file", response_model=Result)
async def predict_file(file: UploadFile = File()):
    """
    接收上传文件，返回故障分类结果
    """
    raw_signal = load_data_from_file(file)
    return predict(raw_signal)

@app.post("/predict/signal", response_model=Result)
async def predict_signal(data: SignalData):
    """
    直接接收信号数据，返回故障分类结果
    """
    signal_array = np.array(data.signal)
    return predict(signal_array)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)