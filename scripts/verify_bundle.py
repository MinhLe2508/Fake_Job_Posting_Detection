"""Kiểm tra tính toàn vẹn file sau giải nén, không cần thư viện ngoài Python chuẩn."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'SHA256SUMS.txt'
if not manifest.exists():raise SystemExit('Thiếu SHA256SUMS.txt')
failed=[];count=0
for line in manifest.read_text(encoding='utf-8').splitlines():
    expected,relative=line.split('  ',1);path=(ROOT/relative).resolve()
    if ROOT not in path.parents:raise SystemExit('Đường dẫn manifest không hợp lệ')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:failed.append(relative)
    count+=1
print(json.dumps({'checked_files':count,'failed':failed,'passed':not failed},ensure_ascii=False,indent=2))
sys.exit(bool(failed))
