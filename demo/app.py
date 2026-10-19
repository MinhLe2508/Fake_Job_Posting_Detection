"""Demo nội bộ: python demo/app.py --port 8765. Chỉ bind localhost, không lưu tin nhập."""
from pathlib import Path
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fakejobs.inference import Detector

class Handler(BaseHTTPRequestHandler):
    def respond(self,status,content,mime='application/json; charset=utf-8'):
        payload=content if isinstance(content,bytes) else json.dumps(content,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        if self.path=='/':self.respond(200,(ROOT/'demo/index.html').read_bytes(),'text/html; charset=utf-8')
        elif self.path=='/api/health':self.respond(200,{'status':'ok','model':self.server.detector.bundle['model_name']})
        elif self.path=='/api/metrics':
            self.respond(200,{name:json.loads((ROOT/f'results/{name}.json').read_text(encoding='utf-8')) for name in ['test_metrics','validation_metrics','selection','test_bootstrap']})
        elif self.path=='/api/examples':self.respond(200,json.loads((ROOT/'evidence/demo_examples.json').read_text(encoding='utf-8')))
        else:self.respond(404,{'error':'Không tìm thấy tài nguyên.'})
    def do_POST(self):
        if self.path!='/api/predict':self.respond(404,{'error':'Không tìm thấy API.'});return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=600000:raise ValueError('Kích thước yêu cầu không hợp lệ.')
            fields=json.loads(self.rfile.read(length).decode('utf-8'))
            result=self.server.detector.predict(fields);self.respond(200,result)
        except (ValueError,TypeError,UnicodeDecodeError) as exc:self.respond(400,{'error':str(exc)})
        except Exception:self.respond(500,{'error':'Lỗi xử lý; kiểm tra môi trường và bundle mô hình.'})
    def log_message(self,fmt,*args):
        # Chỉ ghi HTTP status/path, không ghi văn bản tuyển dụng của người dùng.
        print(fmt%args,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.detector=Detector()
    print(f'Demo: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
