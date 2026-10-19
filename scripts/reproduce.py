"""Chạy mới hoàn toàn trong thư mục chưa tồn tại, giữ nguyên kết quả đã nộp."""
from pathlib import Path
import argparse, os, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,help='Thư mục kết quả mới, phải chưa tồn tại')
args=parser.parse_args();dest=Path(args.output).resolve()
if dest.exists():raise SystemExit('Thư mục đã tồn tại. Chọn tên mới để không ghi đè kết quả.')
if ROOT==dest or ROOT in dest.parents:raise SystemExit('Đặt thư mục tái lập bên ngoài bộ nộp.')
dest.mkdir(parents=True)
for folder in ['src','data','tests','scripts']:
    shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
for folder in ['results','models','evidence']:(dest/folder).mkdir()
env=os.environ.copy();env['PYTHONPATH']=str(dest/'src')+os.pathsep+env.get('PYTHONPATH','')
env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['PYTHONIOENCODING']='utf-8'
for command in [['-m','fakejobs.experiment','--stage','all'],['-m','fakejobs.analysis'],['-m','unittest','discover','-s','tests','-v']]:
    subprocess.run([sys.executable,*command],cwd=dest,env=env,check=True)
print('Đã tái lập tại',dest)
