# Flyppy live viewer postmortem

- experiment: `/Users/<local-user>/Desktop/virtual-fly/artifacts/experiments/flyppy-v3`
- diagnosis: **VIEWER_HTTP_ACTIVE_BUT_TRAINER_TELEMETRY_NEVER_MATERIALIZED**
- run episode range: 96..143
- population: 4
- vision runtime: `direct-ray`
- rays/ommatidium: 13
- viewer telemetry mode: `on-demand`

## Residual live artifacts

| artifact | exists | bytes | modified UTC | logical key |
|---|---|---:|---|---|
| camera | true | 110 | 2026-09-15T10:14:46+00:00 | - |
| status | false | 0 | - | - |
| body | false | 0 | - | - |
| neural | false | 0 | - | - |
| frame | true | 81239 | 2026-09-15T10:24:24+00:00 | - |

## Interpretation

- `camera.json` exists only if the browser successfully reached the viewer server's camera POST endpoint.
- `body.json` / `neural.json` require the trainer to observe an active viewer lease and enter its telemetry path.
- `fly.png` requires `live_body_viewer.py` to consume `body.json` and render a frame.
- The trainer control socket is expected to be absent after training exits, so its current absence is not treated as a failure.

## Viewer server log tail

```text
----------------------------------------
Exception occurred during processing of request from ('127.0.0.1', 51057)
Traceback (most recent call last):
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py", line 697, in process_request_thread
    self.finish_request(request, client_address)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py", line 362, in finish_request
    self.RequestHandlerClass(request, client_address, self)
    ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/live_viewer_server.py", line 96, in __init__
    super().__init__(*args, directory=str(root), **kwargs)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 672, in __init__
    super().__init__(*args, **kwargs)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py", line 766, in __init__
    self.handle()
    ~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 436, in handle
    self.handle_one_request()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 424, in handle_one_request
    method()
    ~~~~~~^^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/live_viewer_server.py", line 137, in do_GET
    super().do_GET()
    ~~~~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 679, in do_GET
    self.copyfile(f, self.wfile)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 881, in copyfile
    shutil.copyfileobj(source, outputfile)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/shutil.py", line 204, in copyfileobj
    fdst_write(buf)
    ~~~~~~~~~~^^^^^
Traceback (most recent call last):
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py", line 845, in write
    self._sock.sendall(b)
    ~~~~~~~~~~~~~~~~~~^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py", line 697, in process_request_thread
    self.finish_request(request, client_address)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
BrokenPipeError: [Errno 32] Broken pipe
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py", line 362, in finish_request
    self.RequestHandlerClass(request, client_address, self)
    ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
----------------------------------------
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/live_viewer_server.py", line 96, in __init__
    super().__init__(*args, directory=str(root), **kwargs)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 672, in __init__
    super().__init__(*args, **kwargs)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py", line 766, in __init__
    self.handle()
    ~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 436, in handle
    self.handle_one_request()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 424, in handle_one_request
    method()
    ~~~~~~^^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/live_viewer_server.py", line 137, in do_GET
    super().do_GET()
    ~~~~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 679, in do_GET
    self.copyfile(f, self.wfile)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/server.py", line 881, in copyfile
    shutil.copyfileobj(source, outputfile)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/shutil.py", line 204, in copyfileobj
    fdst_write(buf)
    ~~~~~~~~~~^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/socketserver.py", line 845, in write
    self._sock.sendall(b)
    ~~~~~~~~~~~~~~~~~~^^^
BrokenPipeError: [Errno 32] Broken pipe
----------------------------------------
```
