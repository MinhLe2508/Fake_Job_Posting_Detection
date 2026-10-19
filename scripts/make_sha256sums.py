"""Tạo lại SHA256SUMS.txt cho toàn bộ bộ nộp (trừ chính file này, __pycache__, .git, .venv, *.pyc).
Chạy SAU KHI đã sửa xong mọi file: python scripts/make_sha256sums.py
Kiểm tra bằng: python scripts/verify_bundle.py"""
from pathlib import Path
import hashlib
ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'__pycache__', '.git', '.venv', '.ipynb_checkpoints'}
lines = []
for path in sorted(p for p in ROOT.rglob('*') if p.is_file()):
    relative = path.relative_to(ROOT)
    if path.name == 'SHA256SUMS.txt' or path.suffix == '.pyc' or SKIP_DIRS & set(relative.parts) or any(p.endswith('.egg-info') for p in relative.parts):
        continue
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative.as_posix()}")
(ROOT / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
print(f'Đã ghi {len(lines)} dòng vào SHA256SUMS.txt')
