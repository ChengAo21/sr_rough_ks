"""Local-only TLS bridge for Windows Julia Downloads/schannel setup failures.

Only official Julia Pkg paths are forwarded. Python verifies upstream HTTPS;
Pkg still verifies package/artifact hashes. This is a setup helper, not an app
dependency or a general-purpose proxy. Run only while initializing PySR.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import re
import urllib.request
from pathlib import Path
import requests


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not re.fullmatch(r"/(registries(?:\.[a-z]+)?|(?:registry|package|artifact)/[a-zA-Z0-9/\-]+)",self.path):
            self.send_error(404);return
        try:
            response=requests.get("https://us-east.pkg.julialang.org"+self.path,timeout=180)
            self.send_response(response.status_code)
            self.send_header("Content-Length",str(len(response.content)))
            self.send_header("Content-Type",response.headers.get("Content-Type","application/octet-stream"))
            self.end_headers();self.wfile.write(response.content)
        except Exception as exc:
            self.send_error(502,str(exc))


if __name__=="__main__":
    print("Julia package TLS bridge at http://127.0.0.1:8510 (official Julia origin only)",flush=True)
    ThreadingHTTPServer(("127.0.0.1",8510),Handler).serve_forever()
