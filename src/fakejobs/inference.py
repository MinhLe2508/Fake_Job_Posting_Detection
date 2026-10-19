"""Dự đoán tin mới; không fit lại và không đổi ngưỡng đã khóa."""
from pathlib import Path
import hashlib, json, time
import joblib
import numpy as np
import pandas as pd
from .features import BINARY_COLUMNS
from .experiment import scores

ROOT=Path(__file__).resolve().parents[2]

class Detector:
    def __init__(self, root=ROOT):
        self.root=Path(root)
        path=self.root/'models/fake_job_detector.joblib'
        lock=json.loads((self.root/'models/lock.json').read_text(encoding='utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=lock['model_sha256']:
            raise ValueError('Mô hình không khớp checksum đã khóa.')
        # Chỉ nạp bundle tin cậy đi kèm bộ nộp; joblib không phải định dạng an toàn cho file lạ.
        self.bundle=joblib.load(path)

    def predict(self, fields):
        if not isinstance(fields,dict):raise ValueError('Dữ liệu phải là một đối tượng JSON.')
        unknown=set(fields)-set(self.bundle['input_columns'])
        if unknown:raise ValueError('Trường không được hỗ trợ: '+', '.join(sorted(unknown)))
        row={c:np.nan for c in self.bundle['input_columns']}
        for key,value in fields.items():
            if value is None or value=='':continue
            if key in BINARY_COLUMNS:
                if value not in [0,1,'0','1']:raise ValueError(f'{key} chỉ nhận 0, 1 hoặc để trống.')
                row[key]=float(value)
            elif isinstance(value,str):row[key]=value.strip() or np.nan
            else:raise ValueError(f'{key} phải là chuỗi văn bản.')
        if not any(isinstance(row[c],str) and row[c].strip() for c in ['title','description']):
            raise ValueError('Cần nhập ít nhất tiêu đề hoặc mô tả công việc bằng tiếng Anh.')
        if sum(len(v) for v in row.values() if isinstance(v,str))>100000:
            raise ValueError('Tổng nội dung tối đa 100.000 ký tự.')
        # Ép cột text thành object để .str hoạt động cả khi cả cột đều thiếu.
        frame=pd.DataFrame([row]).astype({c:'object' for c in row if c not in BINARY_COLUMNS})
        start=time.perf_counter()
        score=float(scores(self.bundle['pipeline'],frame)[0])
        threshold=float(self.bundle['threshold']);alert=score>=threshold
        return {'score':score,'threshold':threshold,'score_type':self.bundle['score_type'],
                'model':self.bundle['model_name'],'flagged':bool(alert),
                'action':'Chuyển kiểm duyệt thủ công' if alert else 'Chưa vượt ngưỡng cảnh báo',
                'latency_ms':1000*(time.perf_counter()-start),
                'note':'Điểm mô hình không xác minh doanh nghiệp; kết quả không phải kết luận gian lận.'}
