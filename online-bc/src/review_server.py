"""Tailscale-local review UI with durable optional exclusions and HF sync."""
import argparse,json,re,subprocess,threading,time,os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from data_control import atomic_json

class ReviewServer(ThreadingHTTPServer):
    def __init__(self,args):
        self.args=args;self.root=Path(args.root);self.root.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock();self.stop_event=threading.Event()
        self.control_path=self.root/'controls.json';self.catalog=Path(args.tracking)/'reports/pi-cup-data-catalog.json';self.learner_path=self.root/'learner-state/state.json';self.sync_path=self.root/'sync-status.json'
        if not self.control_path.exists():atomic_json(self.control_path,dict(run=args.run,revision=0,excluded={},paused=False,bootstrap_steps=0,review_required=False,updated_at=time.time()))
        super().__init__((args.bind,args.port),Handler);self.origin=f'http://{args.bind}:{args.port}'
    def state(self):
        with self.lock:
            controls=json.loads(self.control_path.read_text());catalog=json.loads(self.catalog.read_text());sync=json.loads(self.sync_path.read_text()) if self.sync_path.exists() else {'published_revision':-1,'status':'pending'}
            learner=json.loads(self.learner_path.read_text()) if self.learner_path.exists() else None
            return dict(controls=controls,episodes=catalog['episodes'],sync=sync,learner=learner)
    def change(self,value):
        with self.lock:
            d=json.loads(self.control_path.read_text());ids={x['id'] for x in json.loads(self.catalog.read_text())['episodes'] if x['eligible']}
            if value.get('revision')!=d['revision']:raise RuntimeError('다른 변경이 반영됐습니다. 새로고침 후 다시 선택하세요.')
            if value.get('action')=='exclude':
                eid=value.get('id');reason=value.get('reason','')
                if eid not in ids or type(value.get('excluded')) is not bool or not isinstance(reason,str) or len(reason)>500:raise ValueError('Invalid episode selection')
                if value['excluded']:d['excluded'][eid]={'reason':reason,'at':time.time()}
                else:d['excluded'].pop(eid,None)
            elif value.get('action')=='pause':
                if type(value.get('paused')) is not bool:raise ValueError('Invalid pause flag')
                d['paused']=value['paused']
            else:raise ValueError('Unknown action')
            d['revision']+=1;d['updated_at']=time.time();atomic_json(self.control_path,d)
            with (self.root/'review-audit.jsonl').open('a') as f:f.write(json.dumps({'at':time.time(),'revision':d['revision'],**value},ensure_ascii=False)+'\n')
            return d
    def transport(self,direction,folder,prefix):
        env=os.environ.copy();env['HF_HOME']=str(self.root/'hf-cache')
        return subprocess.run([self.args.transport_python,str(Path(__file__).with_name('hf_transfer.py')),direction,str(folder),prefix,'--token-file',self.args.token_file],capture_output=True,text=True,timeout=45,env=env)
    def sync_loop(self):
        last=-1;pull_at=0
        while not self.stop_event.is_set():
            try:
                with self.lock:d=json.loads(self.control_path.read_text())
                if d['revision']!=last:
                    folder=self.root/'outgoing-controls';atomic_json(folder/'controls.json',d);r=self.transport('upload',folder,f'{self.args.run}/controls')
                    if r.returncode:raise RuntimeError('HF control upload failed')
                    last=d['revision'];atomic_json(self.sync_path,dict(published_revision=last,status='synced',at=time.time()))
                if time.time()-pull_at>=15:
                    self.transport('download',self.root/'learner-state',f'{self.args.run}/learner-state/pi05');pull_at=time.time()
            except (OSError,ValueError,subprocess.TimeoutExpired,RuntimeError):
                atomic_json(self.sync_path,dict(published_revision=last,status='retrying',at=time.time()))
            self.stop_event.wait(2)

class Handler(BaseHTTPRequestHandler):
    def json(self,status,d):
        b=json.dumps(d,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/api/state':return self.json(200,self.server.state())
        if path not in ('/','/index.html'):return self.json(404,{'error':'Not found'})
        b=(Path(__file__).resolve().parents[1]/'web/index.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_POST(self):
        if self.path!='/api/control':return self.json(404,{'error':'Not found'})
        if self.headers.get('Origin')!=self.server.origin or 'http://'+self.headers.get('Host','')!=self.server.origin:return self.json(403,{'error':'Same-origin required'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if self.headers.get('Content-Type','').split(';')[0]!='application/json' or not 1<=length<=2048:raise ValueError('JSON body required')
            value=json.loads(self.rfile.read(length));controls=self.server.change(value);return self.json(200,{'controls':controls})
        except RuntimeError as e:return self.json(409,{'error':str(e)})
        except (ValueError,KeyError,TypeError) as e:return self.json(400,{'error':str(e)})
    def log_message(self,*args):pass

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--tracking',required=True);p.add_argument('--run',default='pi05-cup-online-bc-20261001');p.add_argument('--bind',default='100.86.183.64');p.add_argument('--port',type=int,default=8897);p.add_argument('--transport-python',required=True);p.add_argument('--token-file',required=True);a=p.parse_args()
    server=ReviewServer(a);threading.Thread(target=server.sync_loop,daemon=True).start();server.serve_forever()
