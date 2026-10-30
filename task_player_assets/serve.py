"""Serve the exact-frame player and import videos on loopback only."""
import argparse
import errno
import functools
import html
import io
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import re
import subprocess
import threading
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import urlopen
import uuid
import webbrowser

MAX_UPLOAD = 2 * 1024 ** 3
ASSETS = Path(__file__).resolve().parent


def render_player(directory):
    """Read the single source of UI code each time; never generate code copies."""
    payload = json.loads((Path(directory) / 'player-data.json').read_text(encoding='utf-8'))
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = (ASSETS / 'player.html').read_text(encoding='utf-8')
    replacements = {
        '__PLAYER_DATA__': encoded,
        '__PLAYER_CORE__': (ASSETS / 'core.js').read_text(encoding='utf-8'),
        '__PLAYER_APP__': (ASSETS / 'app.js').read_text(encoding='utf-8'),
        '__PLAYER_TITLE__': html.escape(payload['episode_id']),
        '__PLAYER_META__': f"{payload['frame_count']} 帧 · {payload['fps']:g} FPS",
    }
    return re.sub(r'__PLAYER_(?:DATA|CORE|APP|TITLE|META)__', lambda match: replacements[match[0]], page).encode('utf-8')


class ImportManager:
    def __init__(self, root):
        self.root = Path(root)
        self.jobs = {}
        self.lock = threading.Lock()
        self.active = False

    def begin(self, filename, task):
        config = json.loads((self.root / 'server-config.json').read_text())
        for key in ('python', 'builder', 'sheet'):
            if not Path(config[key]).is_file():
                raise ValueError('导入环境已移动，请重新生成播放器后再试。')
        if task:
            tasks = json.loads(Path(config['sheet']).read_text())['tasks']
            if task not in {item['task'] for item in tasks}:
                raise ValueError('请选择有效的 task sheet。')
        with self.lock:
            if self.active:
                raise ValueError('已有视频正在导入，请等待完成。')
            self.active = True
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {'state': 'uploading', 'message': '正在接收视频…'}
        upload = self.root / '.uploads' / job_id / filename
        try:
            upload.parent.mkdir(parents=True)
        except OSError:
            with self.lock:
                self.active = False
            raise
        return job_id, upload, config

    def fail(self, job_id, message):
        with self.lock:
            self.jobs[job_id] = {'state': 'error', 'message': message}
            self.active = False

    def decode(self, job_id, upload, config, task):
        output = self.root / 'imports' / job_id
        try:
            with self.lock:
                self.jobs[job_id] = {'state': 'decoding', 'message': '正在准备逐帧画面…'}
            command = [config['python'], '-u', config['builder'], '--video', str(upload),
                       '--task-sheet', config['sheet'], '--out', str(output)]
            if task:
                command.extend(['--task', task])
            tail = []
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, cwd=str(Path(config['builder']).parent)) as process:
                for line in process.stdout:
                    line = line.strip()
                    tail = (tail + [line])[-12:]
                    if line.startswith('Decoded '):
                        with self.lock:
                            self.jobs[job_id]['message'] = '正在提取帧：' + line.removeprefix('Decoded ')
                code = process.wait()
            if code:
                raise ValueError('\n'.join(tail) or '视频解码失败。')
            with self.lock:
                self.jobs[job_id] = {'state': 'ready', 'message': '导入完成，可以打开视频。',
                                     'url': f'/imports/{job_id}/index.html'}
                self.active = False
        except Exception as exc:
            shutil.rmtree(output, ignore_errors=True)
            self.fail(job_id, str(exc))
        finally:
            shutil.rmtree(upload.parent, ignore_errors=True)

    def catalog(self):
        entries = []
        paths = [self.root / 'manifest.json', *sorted((self.root / 'imports').glob('*/manifest.json'))]
        for path in paths:
            try:
                data = json.loads(path.read_text())
                if not data.get('complete') or not (path.parent / 'player-data.json').is_file():
                    continue
                meta = data['meta']
                url = '/' + (path.parent / 'index.html').relative_to(self.root).as_posix()
                entries.append({'episode_id': meta['episode_id'], 'frame_count': meta['frame_count'], 'url': url})
            except (OSError, ValueError, KeyError):
                continue
        return entries


class PlayerHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        # Keep existing URLs and browser draft origin while serving live shared code.
        path = unquote(urlsplit(self.path).path)
        if path in ('/', '/index.html') or re.fullmatch(r'/imports/[a-f0-9]{32}/index.html', path):
            directory = Path(self.directory) if path in ('/', '/index.html') else Path(self.directory) / path.lstrip('/').rsplit('/', 1)[0]
            try:
                page = render_player(directory)
            except (OSError, ValueError, KeyError):
                self.send_error(404, 'Video data is not ready')
                return None
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page)))
            self.end_headers()
            return io.BytesIO(page)
        # Only video metadata and exact frame images are exposed as static files.
        if not re.fullmatch(r'/(?:imports/[a-f0-9]{32}/)?(?:manifest\.json|frames/[0-9]{6,}\.jpg)', path):
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        if not self.path.split('?', 1)[0].endswith('.jpg'):
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404, 'Not found')
        return None

    def json_response(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/videos':
            return self.json_response(200, {'videos': self.server.imports.catalog()})
        if path.startswith('/api/imports/'):
            with self.server.imports.lock:
                job = self.server.imports.jobs.get(path.rsplit('/', 1)[-1])
                job = dict(job) if job else None
            return self.json_response(200 if job else 404, job or {'message': '导入任务不存在，请重新导入。'})
        # Uploaded originals and runtime configuration are not public assets.
        parts = Path(unquote(path)).parts
        if any(part.startswith('.') and part != '/' for part in parts) or any(
                part in ('server-config.json', 'serve.py', '打开播放器.command') for part in parts):
            return self.send_error(404)
        super().do_GET()

    def do_POST(self):
        parsed = urlsplit(self.path)
        if parsed.path != '/api/import':
            return self.json_response(404, {'message': 'Not found'})
        expected = f'127.0.0.1:{self.server.server_port}'
        # Cross-origin pages cannot issue this custom-header request. No CORS is enabled.
        if (self.headers.get('Host') != expected or
                self.headers.get('Origin') != f'http://{expected}' or
                self.headers.get('X-Player-Import') != '1'):
            return self.json_response(403, {'message': '请从本机播放器页面导入。'})
        if self.headers.get('Content-Type') != 'application/octet-stream':
            return self.json_response(415, {'message': '需要视频文件。'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_UPLOAD:
                raise ValueError('请选择非空视频，最大支持 2 GB。')
            query = parse_qs(parsed.query)
            filename = query.get('filename', [''])[0]
            task = query.get('task', [''])[0]
            if (not filename or Path(filename).name != filename or '\\' in filename or
                    any(ord(c) < 32 for c in filename) or
                    Path(filename).suffix.lower() not in {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}):
                raise ValueError('请选择 MP4 / MOV / M4V / AVI / MKV / WebM 视频。')
            job_id, upload, config = self.server.imports.begin(filename, task)
        except (ValueError, OSError, KeyError) as exc:
            self.close_connection = True
            return self.json_response(400, {'message': str(exc)})
        try:
            self.connection.settimeout(120)
            with upload.open('wb') as stream:
                remaining = size
                while remaining:
                    block = self.rfile.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError('视频传输中断，请重新导入。')
                    stream.write(block)
                    remaining -= len(block)
        except (OSError, ValueError) as exc:
            shutil.rmtree(upload.parent, ignore_errors=True)
            self.server.imports.fail(job_id, str(exc))
            return self.json_response(400, {'message': str(exc)})
        threading.Thread(target=self.server.imports.decode, args=(job_id, upload, config, task), daemon=True).start()
        self.json_response(202, {'job_id': job_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--directory', type=Path, default=ASSETS.parent / 'players' / '12_fold_four_towels')
    args = parser.parse_args()
    root = args.directory.resolve()
    if not (root / 'player-data.json').is_file():
        parser.error('player-data.json is missing; run task_player.py to prepare the video first')
    handler = functools.partial(PlayerHandler, directory=str(root))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            try:
                expected = json.loads((root / 'manifest.json').read_text())['config']['source_sha256']
                with urlopen(f'http://127.0.0.1:{args.port}/manifest.json', timeout=2) as response:
                    existing = json.loads(response.read(65536))
                if existing.get('config', {}).get('source_sha256') == expected:
                    url = f'http://127.0.0.1:{args.port}/index.html'
                    print(f'播放器已在运行：{url}', flush=True)
                    if not args.no_browser:
                        webbrowser.open(url)
                    return
            except (OSError, KeyError, ValueError):
                pass
        parser.exit(1, f'Cannot start player: {exc}. Try --port 8767.\n')
    server.imports = ImportManager(root)
    url = f'http://127.0.0.1:{server.server_port}/index.html'
    print(f'播放器地址：{url}\n请保持此窗口打开；Ctrl+C 停止。', flush=True)
    if not args.no_browser:
        threading.Timer(.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
