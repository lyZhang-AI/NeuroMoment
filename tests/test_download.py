import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import requests

from scripts import download_eeg


def test_bounded_range_resume_and_sha256(tmp_path, monkeypatch):
    content = b'neural-evidence-' * 600000  # >8 MiB: exercise multiple requests.
    ranges = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            value=self.headers['Range']; ranges.append(value)
            a,b=map(int,value.removeprefix('bytes=').split('-'))
            body=content[a:b+1]
            self.send_response(206)
            self.send_header('Content-Range',f'bytes {a}-{b}/{len(content)}')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    session=requests.Session();session.trust_env=False
    monkeypatch.setattr(download_eeg,'get_session',lambda:session)
    rec={'path':'recording.eeg','target':'recording.eeg','size':len(content),
         'sha256':hashlib.sha256(content).hexdigest(),
         'url':f'http://127.0.0.1:{server.server_port}/recording.eeg'}
    (tmp_path/'recording.eeg.part').write_bytes(content[:1024])
    try:
        method,size=download_eeg.fetch(tmp_path,rec,None)
        assert method=='downloaded_sha256_verified' and size==len(content)
        assert ranges[0].startswith('bytes=1024-') and len(ranges)==2
        assert (tmp_path/'recording.eeg').read_bytes()==content
        assert download_eeg.fetch(tmp_path,rec,None)[0]=='verified_existing'
    finally:
        server.shutdown();server.server_close();session.close()


def test_local_reuse_requires_hash_match(tmp_path):
    root=tmp_path/'new';root.mkdir()
    old=tmp_path/'old';old.mkdir()
    data=b'original EEG fixture'
    (old/'recording.eeg').write_bytes(data)
    rec={'path':'recording.eeg','target':'recording.eeg','size':len(data),
         'sha256':hashlib.sha256(data).hexdigest(),'url':'unused'}
    assert download_eeg.fetch(root,rec,old)[0]=='verified_local_copy'
    assert (root/'recording.eeg').read_bytes()==data
    assert (old/'recording.eeg').read_bytes()==data
