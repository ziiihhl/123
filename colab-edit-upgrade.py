import hashlib
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
_manifest_lock=threading.RLock()
def safetensors_complete(path, expected_size=None):
    try:
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if not 2 <= length <= 100_000_000:
                return False
            header = json.loads(stream.read(length))
        ends = [value['data_offsets'][1] for name, value in header.items() if name != '__metadata__']
        return bool(ends) and path.stat().st_size == 8 + length + max(ends)
    except (OSError, ValueError, TypeError, KeyError, struct.error):
        return False

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def save_manifest(path, manifest):
    with _manifest_lock:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        temporary.replace(path)

def remember_verified(path, spec, manifest):
    stat = path.stat()
    with _manifest_lock:
        manifest[spec['name']] = {'sha256': spec['sha256'], 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

def verified_model(path, spec, manifest):
    if not safetensors_complete(path, spec['size']):
        return False
    stat = path.stat()
    old = manifest.get(spec['name'], {})
    if old == {'sha256': spec['sha256'], 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
        return True
    print('校验 SHA256：', spec['name'], flush=True)
    if file_sha256(path) != spec['sha256']:
        return False
    remember_verified(path, spec, manifest)
    return True

def sources_for(spec, civitai_token):
    if spec.get('_resolved_download_url'):
        return [spec['_resolved_download_url']]
    if spec.get('civitai_only'):
        parsed=urllib.parse.urlsplit(spec['download_url'])
        if parsed.scheme!='https' or parsed.hostname not in ('civitai.com','www.civitai.com'):
            raise ValueError('Only official Civitai download URLs are allowed')
        query=dict(urllib.parse.parse_qsl(parsed.query))
        query.pop('token',None)
        if civitai_token:query['token']=civitai_token
        return [urllib.parse.urlunsplit((parsed.scheme,parsed.netloc,parsed.path,urllib.parse.urlencode(query),''))]
    if spec.get('repo'):
        return ['https://huggingface.co/' + spec['repo'] + '/resolve/main/' + spec['remote']]
    parameters = {'fileId': spec['file_id']}
    if civitai_token:
        parameters['token'] = civitai_token
    official = 'https://civitai.com/api/download/models/' + str(spec['version_id']) + '?' + urllib.parse.urlencode(parameters)
    # The mirror appeared in the original notebook. Never send the API token to it.
    mirror = 'https://civitai.red/api/download/models/' + str(spec['version_id']) + '?fileId=' + str(spec['file_id'])
    return [official, mirror]

def download_with_aria2(spec, destination, manifest_path, manifest, token, progress_callback=None):
    """Use the original notebook's 16-connection engine, with private RPC progress."""
    import secrets
    part = destination.with_suffix(destination.suffix + '.part')
    failures = []
    for url in sources_for(spec, token):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        rpc_secret = secrets.token_hex(24)
        command = ['aria2c', '--input-file=-', '--continue=true', '--max-connection-per-server=16',
                   '--split=16', '--min-split-size=1M', '--file-allocation=none',
                   '--auto-file-renaming=false', '--allow-overwrite=true', '--max-tries=3',
                   '--retry-wait=2', '--connect-timeout=30', '--timeout=90',
                   '--check-integrity=true', '--checksum=sha-256=' + spec['sha256'],
                   '--enable-rpc=true', '--rpc-listen-all=false', '--rpc-listen-port=' + str(port),
                   '--rpc-secret=' + rpc_secret, '--console-log-level=error', '--summary-interval=0',
                   '--dir=' + str(destination.parent), '--out=' + part.name,
                   '--user-agent=Mozilla/5.0']
        print('aria2：16 连接分段下载，下载后校验 SHA256：', spec['name'], flush=True)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        progress = DownloadProgress(spec['name'], spec['size'], part.stat().st_size if part.exists() else 0, callback=progress_callback)
        # Pass URLs through stdin, keeping Civitai tokens out of process arguments and output.
        process.stdin.write((url + '\n  out=' + part.name + '\n').encode())
        process.stdin.close()
        terminal_status = None
        terminal_detail = ''
        terminal_code = None
        try:
            while process.poll() is None:
                request = urllib.request.Request('http://127.0.0.1:' + str(port) + '/jsonrpc',
                    data=json.dumps({'jsonrpc':'2.0','id':'progress','method':'aria2.tellActive',
                                     'params':['token:' + rpc_secret, ['completedLength','totalLength','downloadSpeed','verifiedLength','verifyIntegrityPending']]}).encode(),
                    headers={'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request, timeout=2) as response:
                        active = json.load(response).get('result', [])
                    if active:
                        item = active[0]
                        completed = int(item.get('completedLength', 0))
                        if progress.offset > completed:
                            progress.offset = completed
                            progress.last_bytes = completed
                        verifying = completed >= spec['size'] or item.get('verifyIntegrityPending') == 'true' or int(item.get('verifiedLength', 0)) > 0
                        progress.update(completed, status='校验 SHA256 中' if verifying else '16 连接下载中',
                                        speed=int(item.get('downloadSpeed', 0)))
                    else:
                        def rpc(method, params):
                            query = urllib.request.Request('http://127.0.0.1:' + str(port) + '/jsonrpc',
                                data=json.dumps({'jsonrpc':'2.0','id':'state','method':method,
                                                 'params':['token:' + rpc_secret, *params]}).encode(),
                                headers={'Content-Type':'application/json'})
                            with urllib.request.urlopen(query, timeout=2) as response:
                                return json.load(response).get('result')
                        stopped = rpc('aria2.tellStopped', [0, 1, ['status','errorCode','errorMessage']])
                        if stopped:
                            terminal_status = stopped[0]['status']
                            terminal_code = stopped[0].get('errorCode')
                            error_message=str(stopped[0].get('errorMessage',''))
                            # Publish only numeric HTTP/aria2 codes; messages may contain signed URLs.
                            import re
                            http_code=re.search(r'(?:status|response)[^\d]{0,30}([45]\d\d)',error_message,re.I)
                            terminal_detail=' (HTTP '+http_code[1]+')' if http_code else ''
                            rpc('aria2.shutdown', [])
                            process.wait(timeout=10)
                            break
                except (OSError, ValueError, urllib.error.URLError):
                    pass  # RPC may not be listening yet; the process exit code remains authoritative.
                time.sleep(0.5)
            if process.returncode == 0 and terminal_status != 'error' and safetensors_complete(part, spec['size']):
                part.replace(destination)
                remember_verified(destination, spec, manifest)
                save_manifest(manifest_path, manifest)
                progress.update(spec['size'], final=True)
                return destination
            host='Civitai CDN' if spec.get('civitai_only') else urllib.parse.urlparse(url).hostname
            if terminal_status == 'error':
                detail = 'aria2 task error ' + str(terminal_code) + ' (process exit ' + str(process.returncode) + ')'
            elif process.returncode == 0 and part.is_file() and part.stat().st_size != spec['size']:
                detail = '文件大小不符：实际 ' + str(part.stat().st_size) + '，配置 ' + str(spec['size']) + ' 字节'
            elif process.returncode == 0:
                detail = 'aria2 已退出，但 safetensors 结构完整性检查未通过'
            else:
                detail = 'aria2 exit ' + str(process.returncode)
            failures.append(host + ': ' + detail + terminal_detail)
            print('此下载源未完成或校验失败，保留断点并尝试备用源。', flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    raise RuntimeError('aria2 下载未完成：' + spec['name'] + '；' + '; '.join(failures))


class DownloadProgress:
    """Notebook progress without extra dependencies; resume bytes do not inflate speed."""
    def __init__(self, name, total, offset=0, callback=None):
        self.name, self.total, self.offset = name, total, offset
        self.callback = callback
        self.started = self.last_time = time.monotonic()
        self.last_bytes = offset
        self.handle = None
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
            if get_ipython() is not None:
                self.handle = display(HTML('准备下载进度…'), display_id=True)
        except ImportError:
            pass
        self.update(offset, force=True)

    def update(self, completed, force=False, final=False, status=None, speed=None):
        now = time.monotonic()
        if not force and not final and now - self.last_time < 1:
            return
        elapsed = max(now - self.started, 1e-9)
        average = (completed - self.offset) / elapsed
        current = (completed - self.last_bytes) / max(now - self.last_time, 1e-9)
        if speed is not None:
            current = speed
        percent = min(100, completed / self.total * 100) if self.total else 100
        eta = (self.total - completed) / average if average > 0 else None
        def duration(seconds):
            if seconds is None:
                return '--:--'
            seconds = max(0, int(seconds))
            hours, rest = divmod(seconds, 3600)
            minutes, seconds = divmod(rest, 60)
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        status = '下载并校验完成' if final else (status or '下载中')
        if self.callback:
            self.callback(dict(completed=completed,total=self.total,speed=current,average_speed=average,
                               eta_seconds=0 if final else eta,percent=percent,status=status,finished=final))
        message = (f'{self.name} | {status} {percent:.1f}% | '
                   f'{completed / 1e9:.2f}/{self.total / 1e9:.2f} GB | '
                   f'当前 {current / 1e6:.1f} MB/s | 平均 {average / 1e6:.1f} MB/s | '
                   f'已用 {duration(elapsed)} | 剩余 {duration(0 if final else eta)}')
        if self.handle is not None:
            import html
            from IPython.display import HTML
            self.handle.update(HTML('<div style="font:13px monospace;white-space:pre-wrap">'
                                    + html.escape(message) + '</div><progress style="width:100%" max="100" value="'
                                    + str(percent) + '"></progress>'))
        else:
            print('\r' + message, end='\n' if final else '', flush=True)
        self.last_bytes, self.last_time = completed, now



"""Civitai URL parsing and exact LoRA version/file selection; shared with Colab."""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

LORA_TARGETS={
    'hassaku':{'label':'Hassaku / Illustrious','family':'SDXL'},
    'anima':{'label':'Anima Turbo','family':'Anima'},
    'krea_anime':{'label':'Krea2 Anime','family':'Krea2'},
    'lustify':{'label':'Lustify Krea2','family':'Krea2'},
}


class CivitaiLoraError(ValueError):pass


def parse_civitai_link(link):
    if not isinstance(link,str) or len(link)>4096:raise CivitaiLoraError('请输入有效的 Civitai 链接。')
    try:
        parsed=urllib.parse.urlsplit(link.strip());port=parsed.port
    except ValueError:raise CivitaiLoraError('Civitai 链接格式无效。') from None
    if parsed.scheme!='https' or parsed.hostname not in ('civitai.com','www.civitai.com') or parsed.username or port not in (None,443):
        raise CivitaiLoraError('当前只支持 https://civitai.com/ 的模型页面和下载链接。')
    query=urllib.parse.parse_qs(parsed.query)
    model=re.fullmatch(r'/models/(\d+)(?:/[^/]*)?/?',parsed.path)
    download=re.fullmatch(r'/api/download/models/(\d+)/?',parsed.path)
    version=re.fullmatch(r'/(?:api/v1/)?model-versions/(\d+)/?',parsed.path)
    def number(key):
        value=(query.get(key) or [''])[0]
        if value and not value.isdigit():raise CivitaiLoraError('链接中的版本或文件编号无效。')
        return int(value) if value else None
    if model:return dict(model_id=int(model[1]),version_id=number('modelVersionId'),file_id=number('fileId'))
    if download or version:return dict(model_id=None,version_id=int((download or version)[1]),file_id=number('fileId'))
    raise CivitaiLoraError('链接应为 Civitai 模型页面、模型版本或官方模型下载链接。')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def civitai_json(path,token=''):
    token=str(token).strip()
    if token and (len(token)>2048 or any(character.isspace() for character in token)):
        raise CivitaiLoraError('CIVITAI_API_TOKEN 格式不正确，请填写 Civitai API Key 本身，不要填写链接或 SSH 私钥。')
    headers={'User-Agent':'Mozilla/5.0','Accept':'application/json'}
    if token:headers['Authorization']='Bearer '+token
    request=urllib.request.Request('https://civitai.com/api/v1/'+path,headers=headers)
    try:
        with urllib.request.build_opener(_NoRedirect).open(request,timeout=30) as response:
            data=json.loads(response.read(8*1024*1024))
        if not isinstance(data,dict):raise ValueError()
        return data
    except urllib.error.HTTPError as error:
        message='Civitai 访问受限；请确认 Colab Secrets 中 CIVITAI_API_TOKEN 的权限。' if error.code in (401,403) else 'Civitai 返回 HTTP '+str(error.code)+'，请检查链接和版本。'
        raise CivitaiLoraError(message) from None
    except (OSError,ValueError,urllib.error.URLError):
        raise CivitaiLoraError('无法读取 Civitai 模型信息，请稍后重试。') from None


def _base_family(value):
    base=re.sub('[^a-z0-9]','',str(value).lower())
    if 'krea2' in base:return 'Krea2'
    if 'anima' in base:return 'Anima'
    if any(key in base for key in ('sdxl','illustrious','noobai','pony')):return 'SDXL'
    if any(key in base for key in ('flux','sd1','sd2','sd3','qwen','zimage','wan')):return 'OtherKnown'
    return None


def resolve_civitai_lora(link,target,version_id=None,token=''):
    destination='drive';kind='lora';base_target=target
    if target.startswith('model:'):
        parts=target.split(':')
        if len(parts)!=3 or parts[1] not in ('local','drive'):raise CivitaiLoraError('模型下载位置无效。')
        _,destination,base_target=parts;kind='checkpoint'
    if base_target not in LORA_TARGETS:raise CivitaiLoraError('请选择目标模型。')
    parsed=parse_civitai_link(link)
    model=None;versions=[]
    if parsed['model_id']:
        model=civitai_json('models/'+str(parsed['model_id']),token)
        versions=model.get('modelVersions') or []
    chosen=version_id or parsed['version_id'] or (versions[0].get('id') if versions else None)
    if not isinstance(chosen,int) or chosen<=0:raise CivitaiLoraError('未找到可下载版本，请使用带 modelVersionId 的链接。')
    version=civitai_json('model-versions/'+str(chosen),token)
    model_info=version.get('model') or model or {}
    if model and version.get('modelId')!=model.get('id'):raise CivitaiLoraError('所选版本不属于链接中的模型。')
    allowed=('LORA','LOCON') if kind=='lora' else ('CHECKPOINT','DIFFUSIONMODEL','DIFFUSION MODEL')
    if str(model_info.get('type','')).upper() not in allowed:
        raise CivitaiLoraError('链接类型与所选下载类型不符，请切换主模型 / LoRA。')
    if not versions:versions=[version]
    files=[]
    for file in version.get('files',[]):
        name=str(file.get('name',''));sha=str((file.get('hashes') or {}).get('SHA256','')).lower()
        if file.get('type') not in (('Model','Other') if kind=='checkpoint' else ('Model',)) or not name.lower().endswith('.safetensors') or not re.fullmatch('[a-f0-9]{64}',sha):continue
        try:size=round(float(file.get('sizeKB',0))*1024)
        except (ValueError,TypeError,OverflowError):continue
        if size<=0:continue
        files.append(dict(id=file['id'],name=name,size=size,sha256=sha,primary=bool(file.get('primary')),
                          download_url=file.get('downloadUrl') or 'https://civitai.com/api/download/models/'+str(chosen)))
    if not files:raise CivitaiLoraError('该版本没有带 SHA256 的 safetensors 模型文件。')
    family=_base_family(version.get('baseModel'))
    compatible=family in (None,LORA_TARGETS[base_target]['family'])
    warning='' if family==LORA_TARGETS[base_target]['family'] else ('Civitai 未明确标注可识别的基础架构，请自行确认兼容性。' if family is None else '基础模型与所选目标不匹配，请改选目标模型或版本。')
    return dict(target=target,kind=kind,destination=destination,family=LORA_TARGETS[base_target]['family'],model_id=version.get('modelId'),model_name=model_info.get('name',''),
                version_id=chosen,version_name=version.get('name',''),base_model=version.get('baseModel',''),compatible=compatible,warning=warning,
                versions=[dict(id=v['id'],name=v.get('name',''),base_model=v.get('baseModel','')) for v in versions],files=files,
                trained_words=[w.strip() for w in version.get('trainedWords',[]) if isinstance(w,str) and 0<len(w.strip())<=512][:32],requested_file_id=parsed['file_id'])


def choose_civitai_file(info,file_id):
    if not info['compatible']:raise CivitaiLoraError(info['warning'])
    file=next((f for f in info['files'] if f['id']==file_id),None)
    if not file:raise CivitaiLoraError('请选择解析结果中的具体 safetensors 文件。')
    parsed=urllib.parse.urlsplit(file['download_url'])
    if parsed.scheme!='https' or parsed.hostname not in ('civitai.com','www.civitai.com') or parsed.path!='/api/download/models/'+str(info['version_id']):
        raise CivitaiLoraError('发布信息中的下载地址不是对应版本的 Civitai 官方链接。')
    query=dict(urllib.parse.parse_qsl(parsed.query));query.pop('token',None);query['fileId']=str(file_id)
    stem=re.sub(r'[^\w.()\[\] -]','_',file['name'].replace('\\','/').split('/')[-1][:-12]).strip('. ')[:110] or 'lora'
    filename=stem+'__v'+str(info['version_id'])+'_f'+str(file_id)+'.safetensors'
    kind=info.get('kind','lora')
    if kind=='checkpoint':
        filename=stem+('' if str(file_id) in stem else '_f'+str(file_id))+'.safetensors'
    return dict(name=filename,kind=kind,destination=info.get('destination','drive'),folder=(('checkpoints' if info['family']=='SDXL' else 'diffusion_models') if kind=='checkpoint' else info['family']),size=file['size'],sha256=file['sha256'],civitai_only=True,
                download_url=urllib.parse.urlunsplit(('https','civitai.com',parsed.path,urllib.parse.urlencode(query),'')),
                version_id=info['version_id'],file_id=file_id,model_id=info['model_id'],trained_words=info['trained_words'],model_name=info['model_name'])


"""Colab metadata/hash index: stores no tensors or credentials, never loads model weights."""
import hashlib
import json
from pathlib import Path
import struct
import threading
import time
_lora_index_lock=globals().get('_lora_index_lock',threading.RLock())


def read_lora_header(path):
    with path.open('rb') as stream:
        prefix=stream.read(8)
        if len(prefix)!=8:raise ValueError('文件头尚未完成')
        length=struct.unpack('<Q',prefix)[0]
        if not 2<=length<=16*1024*1024:
            if prefix.lstrip().lower().startswith((b'<!doctyp',b'<html',b'<?xml',b'{"error')):
                raise ValueError('文件内容像网页或错误响应，不是 safetensors 权重')
            raise ValueError('safetensors 文件头大小无效，可能下载错误或文件损坏')
        header=json.loads(stream.read(length))
    ends=[value['data_offsets'][1] for key,value in header.items() if key!='__metadata__']
    if not ends or path.stat().st_size!=8+length+max(ends):raise ValueError('权重仍在上传或文件不完整')
    metadata=header.get('__metadata__',{})
    return metadata if isinstance(metadata,dict) else {}


def _index_loras_unlocked(root, index_path):
    root=Path(root);index_path=Path(index_path)
    old={}
    try:
        old=json.loads(index_path.read_text(encoding='utf-8'))
        if not isinstance(old,dict):old={}
        previous=old.get('files',{})
    except (OSError,ValueError):previous={}
    files={};errors={}
    for path in sorted(root.rglob('*.safetensors')):
        try:
            path.resolve().relative_to(root.resolve())
            relative=path.relative_to(root).as_posix()
            stat=path.stat()
            prior=previous.get(relative,{})
            stamp=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
            if prior.get('size')==stamp['size'] and prior.get('mtime_ns')==stamp['mtime_ns'] and prior.get('sha256'):
                files[relative]=prior;continue
            metadata=read_lora_header(path)
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                while chunk:=stream.read(8*1024*1024):digest.update(chunk)
            after=path.stat()
            if after.st_size!=stat.st_size or after.st_mtime_ns!=stat.st_mtime_ns:continue
            fields=('modelspec.trigger_phrase','modelspec.trigger_words','trigger_words','triggerWords','trainedWords',
                    'activation_text','activation_tags','ss_activation_tag','ss_trigger_words','ss_tag_frequency')
            selected={key:metadata[key] for key in fields if key in metadata and len(str(metadata[key]))<=512*1024}
            files[relative]=dict(**stamp,sha256=digest.hexdigest(),metadata=selected)
            if prior.get('sha256')==digest.hexdigest() and isinstance(prior.get('civitai'),dict):
                files[relative]['civitai']=prior['civitai']
            print('LoRA 索引已更新：',relative,flush=True)
        except (OSError,ValueError,KeyError,TypeError,struct.error) as error:
            relative=path.relative_to(root).as_posix()
            message=str(error) if isinstance(error,ValueError) else '暂时无法读取完整的 safetensors 文件，请检查上传是否完成'
            errors[relative]=dict(message=message[:240],size=path.stat().st_size if path.exists() else 0)
            continue # An upload in progress is retried on the next scan.
    if files!=previous or errors!=old.get('errors',{}) or not index_path.exists():
        index_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=index_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(version=1,files=files,errors=errors,updated_at=time.time()),ensure_ascii=False),encoding='utf-8')
        temporary.replace(index_path)
    return len(files)


def index_loras_once(root,index_path):
    with _lora_index_lock:return _index_loras_unlocked(root,index_path)


def record_downloaded_lora(root,index_path,target,spec):
    """aria2 already checked SHA256; index the verified file without hashing it again."""
    with _lora_index_lock:
        try:data=json.loads(Path(index_path).read_text(encoding='utf-8'))
        except (OSError,ValueError):data=dict(version=1,files={},errors={})
        relative=Path(target).relative_to(root).as_posix();stat=Path(target).stat()
        metadata=read_lora_header(Path(target))
        fields=('modelspec.trigger_phrase','modelspec.trigger_words','trigger_words','triggerWords','trainedWords','activation_text','activation_tags','ss_activation_tag','ss_trigger_words','ss_tag_frequency')
        selected={key:metadata[key] for key in fields if key in metadata and len(str(metadata[key]))<=512*1024}
        data.setdefault('files',{})[relative]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=spec['sha256'],metadata=selected,
            civitai=dict(sha256=spec['sha256'],trainedWords=spec['trained_words'],modelId=spec['model_id'],versionId=spec['version_id']))
        data.setdefault('errors',{}).pop(relative,None);data['updated_at']=time.time();data['version']=1
        path=Path(index_path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8');temporary.replace(path)


def start_lora_indexer(root,index_path,interval=60):
    previous=globals().get('_comfy_lora_indexer_stop')
    if previous:previous.set()
    previous_thread=globals().get('_comfy_lora_indexer_thread')
    if previous_thread and previous_thread.is_alive():
        previous_thread.join(timeout=10)
        if previous_thread.is_alive():
            print('上一轮 LoRA 索引仍在读取新文件，请稍后再运行本单元格。',flush=True)
            return previous_thread
    count=index_loras_once(root,index_path)
    stop=threading.Event()
    globals()['_comfy_lora_indexer_stop']=stop
    def run():
        while not stop.wait(interval):
            try:index_loras_once(root,index_path)
            except Exception:pass
    thread=threading.Thread(target=run,name='comfy-lora-metadata-index',daemon=True)
    globals()['_comfy_lora_indexer_thread']=thread
    thread.start()
    print('LoRA 触发词索引就绪：',count,'个文件；每分钟检查新增或修改文件，已有文件不重复计算哈希。',flush=True)
    return thread




"""Authorized Civitai download queue consumed by the current Colab runtime."""
import json
import re
from pathlib import Path
import shutil
import threading
import time
import uuid


class LoraDownloadCancelled(Exception):pass


def _write_lora_job(path,data):
    data['updated_at']=time.time()
    if path.name=='lora_download_worker.json':
        data['protocol']=2
        data['civitai_token_loaded']=bool(globals().get('_COMFY_CIVITAI_TOKEN'))
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def _lora_token():
    if '_COMFY_CIVITAI_TOKEN' in globals():return globals()['_COMFY_CIVITAI_TOKEN']
    try:
        from google.colab import userdata
        return userdata.get('CIVITAI_API_TOKEN') or ''
    except Exception:return globals().get('CIVITAI_TOKEN','')


def resolve_civitai_download(spec,token):
    """Authenticate once at Civitai, then let aria2 use the signed CDN URL."""
    import urllib.parse
    import urllib.request
    import urllib.error
    token=str(token).strip()
    if token and (len(token)>2048 or any(character.isspace() for character in token)):
        raise CivitaiLoraError('CIVITAI_API_TOKEN 格式不正确，请填写 API Key 本身，不要填写链接或 SSH 私钥。')
    official=sources_for(spec,token)[0]
    current=official
    opener=urllib.request.build_opener(_NoRedirect)
    for _ in range(8):
        parsed=urllib.parse.urlsplit(current)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username:
            raise CivitaiLoraError('Civitai 返回了无效的下载重定向。')
        official_host=parsed.hostname in ('civitai.com','www.civitai.com')
        if official_host and parsed.path.startswith(('/login','/user/login','/api/auth')):
            raise CivitaiLoraError('Civitai 要求登录，下载鉴权未通过。请更新 Colab Secrets 中 CIVITAI_API_TOKEN 并重新运行第 4D 节，再重试；不要把 Token 发到聊天。')
        if not official_host and token and token in urllib.parse.unquote(current):
            raise CivitaiLoraError('下载重定向异常，已阻止将 Civitai API Token 发往其他主机。')
        headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity','Range':'bytes=0-0'}
        if official_host and token:headers['Authorization']='Bearer '+token
        try:
            with opener.open(urllib.request.Request(current,headers=headers),timeout=30) as response:
                kind=response.headers.get('Content-Type','').lower()
                if response.status not in (200,206):raise CivitaiLoraError('下载预检收到异常 HTTP 状态。')
                if any(t in kind for t in ('text/html','application/json','text/plain')):
                    raise CivitaiLoraError('下载地址返回网页或错误信息，没有返回模型文件。请检查 Civitai Token 和下载权限。')
                content_range=response.headers.get('Content-Range','')
                total=re.search(r'/(\d+)$',content_range)
                if total and int(total[1])!=spec['size']:
                    raise CivitaiLoraError('下载响应的文件大小与所选版本不一致，已停止下载。')
                return current
        except urllib.error.HTTPError as error:
            error.close()
            if error.code in (301,302,303,307,308):
                location=error.headers.get('Location')
                if not location:raise CivitaiLoraError('Civitai/CDN 重定向缺少 Location 响应头，暂时无法下载。') from None
                current=urllib.parse.urljoin(current,location)
                continue
            if error.code in (401,403):
                raise CivitaiLoraError('Civitai/CDN 拒绝下载（HTTP '+str(error.code)+'）。请确认 CIVITAI_API_TOKEN 有效且账号有此文件的下载权限。') from None
            if error.code==429:raise CivitaiLoraError('Civitai 暂时限流，请稍后重试。') from None
            raise CivitaiLoraError('下载预检失败：HTTP '+str(error.code)+'。') from None
        except (OSError,urllib.error.URLError):
            raise CivitaiLoraError('下载预检网络连接失败，请稍后重试。') from None
    raise CivitaiLoraError('下载重定向次数过多。')


def execute_lora_download(info,file_id,root,index_path,progress=None,cancelled=None):
    if info.get('kind')=='checkpoint':
        return execute_checkpoint_download(info,file_id,root,index_path,progress,cancelled)
    spec=choose_civitai_file(info,file_id)
    root=Path(root);target=root/spec['folder']/spec['name'];target.parent.mkdir(parents=True,exist_ok=True)
    target.resolve().relative_to(root.resolve())
    manifest_path=root.parent.parent/'manifests/lora-downloads.json';manifest_path.parent.mkdir(parents=True,exist_ok=True)
    try:manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError,ValueError):manifest={}
    if cancelled and cancelled():raise LoraDownloadCancelled()
    if verified_model(target,spec,manifest):
        save_manifest(manifest_path,manifest)
        record_downloaded_lora(root,index_path,target,spec)
        return dict(filename='Drive/'+target.relative_to(root).as_posix(),size=spec['size'],trigger_words=spec['trained_words'],sha256=spec['sha256'])
    if target.exists():raise CivitaiLoraError('同名文件已存在但校验失败，已保留原文件，请先另存或检查：'+spec['name'])
    if shutil.disk_usage(root).free<spec['size']+128*1024*1024:raise CivitaiLoraError('Drive 可用空间不足。')
    if not shutil.which('aria2c'):raise CivitaiLoraError('缺少 aria2，请先运行笔记本第 2 节安装依赖。')
    token=_lora_token()
    spec['_resolved_download_url']=resolve_civitai_download(spec,token)
    def report(state):
        if cancelled and cancelled():raise LoraDownloadCancelled()
        if progress:progress(state)
    with _lora_index_lock:
        download_with_aria2(spec,target,manifest_path,manifest,token,progress_callback=report)
        record_downloaded_lora(root,index_path,target,spec)
    return dict(filename='Drive/'+target.relative_to(root).as_posix(),size=spec['size'],trigger_words=spec['trained_words'],sha256=spec['sha256'])


def execute_checkpoint_download(info,file_id,root,index_path,progress=None,cancelled=None):
    spec=choose_civitai_file(info,file_id)
    persistent=Path(root).parent.parent
    models=(persistent/'models') if spec['destination']=='drive' else Path(globals().get('CACHE_MODELS','/content/comfy-model-cache'))
    target=models/spec['folder']/spec['name'];target.parent.mkdir(parents=True,exist_ok=True)
    target.resolve().relative_to(models.resolve())
    manifest_path=(persistent/'manifests/model-downloads.json') if spec['destination']=='drive' else models/'.downloaded_models.json'
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    try:manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError,ValueError):manifest={}
    if cancelled and cancelled():raise LoraDownloadCancelled()
    if not verified_model(target,spec,manifest):
        if target.exists():raise CivitaiLoraError('同名模型已存在但验证失败，已保留原文件：'+spec['name'])
        if shutil.disk_usage(models).free<spec['size']+1024**3:raise CivitaiLoraError('目标磁盘空间不足，请更换下载位置或清理空间。')
        if not shutil.which('aria2c'):raise CivitaiLoraError('缺少 aria2，请先运行第 2 节。')
        token=_lora_token();spec['_resolved_download_url']=resolve_civitai_download(spec,token)
        def report(state):
            if cancelled and cancelled():raise LoraDownloadCancelled()
            if progress:progress(state)
        download_with_aria2(spec,target,manifest_path,manifest,token,progress_callback=report)
    save_manifest(manifest_path,manifest)
    # Connect both storage choices even if this runtime started with the other source.
    alias='DownloadsDrive' if spec['destination']=='drive' else 'DownloadsLocal'
    link=Path('/content/ComfyUI/models')/spec['folder']/alias
    link.parent.mkdir(parents=True,exist_ok=True)
    if link.is_symlink():
        if link.resolve()!=(models/spec['folder']).resolve():raise CivitaiLoraError('下载目录链接冲突，原目录已保留。')
    elif link.exists():raise CivitaiLoraError('下载目录已被占用，原目录已保留。')
    else:link.symlink_to(models/spec['folder'],target_is_directory=True)
    return dict(filename=alias+'/'+spec['name'],path=str(target),size=spec['size'],sha256=spec['sha256'],kind='checkpoint')


def start_lora_download_worker(root,index_path):
    # Colab Secrets access belongs on the notebook thread, before background jobs.
    try:
        from google.colab import userdata
        globals()['_COMFY_CIVITAI_TOKEN']=(userdata.get('CIVITAI_API_TOKEN') or '').strip()
    except Exception:
        globals()['_COMFY_CIVITAI_TOKEN']=''
    print('Civitai Token：'+('已从当前 Secrets 读取（不显示内容）。' if globals()['_COMFY_CIVITAI_TOKEN'] else '未读取到；需要鉴权的下载会失败，请检查 Secrets 权限。'),flush=True)
    existing=globals().get('_comfy_lora_download_thread')
    if existing and existing.is_alive():
        print('复用当前 LoRA 下载接收程序。',flush=True);return existing
    root=Path(root);user_dir=Path(index_path).parent
    job_dir=user_dir/'lora_download_jobs';job_dir.mkdir(parents=True,exist_ok=True)
    worker_id=uuid.uuid4().hex
    def heartbeat():_write_lora_job(user_dir/'lora_download_worker.json',dict(worker_id=worker_id,protocol=1))
    def run():
        while True:
            heartbeat()
            for path in sorted(job_dir.glob('*.json')):
                if not re.fullmatch(r'[0-9a-f]{32}\.json',path.name):continue
                job=None
                try:
                    job=json.loads(path.read_text(encoding='utf-8'))
                    if not isinstance(job,dict):continue
                    if job.get('status') in ('success','failed','cancelled'):continue
                    if job.get('worker_id')==worker_id and job.get('status')!='pending':continue
                    cancel_path=path.with_suffix('.cancel.json')
                    if cancel_path.exists():job['status']='cancelled';_write_lora_job(path,job);continue
                    job.update(status='resolving',worker_id=worker_id);_write_lora_job(path,job)
                    def progress(state):
                        heartbeat()
                        status='verifying' if '校验' in state['status'] and state['completed']>=state['total'] else 'downloading'
                        job.update(status=status,progress=state);_write_lora_job(path,job)
                    link='https://civitai.com/api/download/models/'+str(int(job['version_id']))
                    info=resolve_civitai_lora(link,job['target'],token=_lora_token())
                    result=execute_lora_download(info,int(job['file_id']),root,index_path,progress,lambda:cancel_path.exists())
                    job.update(status='success',result=result,error=None);_write_lora_job(path,job)
                except LoraDownloadCancelled:
                    job.update(status='cancelled');_write_lora_job(path,job)
                except Exception as error:
                    if not isinstance(job,dict):continue
                    message=str(error) if isinstance(error,(CivitaiLoraError,RuntimeError)) else '下载接收程序发生错误，请检查网络、Drive 空间和文件权限。'
                    job.update(status='failed',error=message[:1000]);_write_lora_job(path,job)
            time.sleep(3)
    thread=threading.Thread(target=run,name='comfy-lora-download-worker',daemon=True)
    globals()['_comfy_lora_download_thread']=thread;thread.start()
    print('网页 LoRA 下载接收程序已启动；只处理你在网页确认提交的下载任务。',flush=True)
    return thread


"""Install editing adapters/nodes, preserving the Colab runtime and existing main models."""
import json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

EDIT_REPOS = {
    'comfyui-krea2edit':('https://github.com/lbouaraba/comfyui-krea2edit','13ef60f20828949becada815d3cbc371eb5ccaab'),
    'krea2-anypaint':('https://github.com/alexw5702-afk/krea2-anypaint','675be5a91eadbf8b8997b21c0e8e1848310b9571'),
    'ComfyUI-Cosmos-Reference':('https://github.com/Mirumo0u0/ComfyUI-Cosmos-Reference','b7fa5a822cd13fb305355ac17064140815ea60c6'),
}

def setup_edit_runtime(specs):
    workspace=Path(globals().get('WORKSPACE','/content/ComfyUI'))
    persistent=Path(globals().get('PERSISTENT','/content/drive/MyDrive/ComfyUI_Persistent'))
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('请先挂载 Drive。')
    if 'download_with_aria2' not in globals():raise RuntimeError('请先运行第 4D 节加载下载组件。')
    print('编辑升级：五个适配权重约 2.49GB；不重新下载四个主模型。',flush=True)
    for name,(url,commit) in EDIT_REPOS.items():
        destination=workspace/'custom_nodes'/name
        if not destination.exists():subprocess.run(['git','clone',url,str(destination)],check=True)
        status=subprocess.run(['git','-C',str(destination),'status','--porcelain'],check=True,capture_output=True,text=True).stdout
        if status.strip():raise RuntimeError('节点目录有本地修改，已保留：'+str(destination))
        subprocess.run(['git','-C',str(destination),'fetch','origin',commit],check=True)
        subprocess.run(['git','-C',str(destination),'checkout','--detach',commit],check=True)
    model_root=persistent/'models';manifest_path=persistent/'manifests/edit-models.json';manifest_path.parent.mkdir(parents=True,exist_ok=True)
    try:manifest=json.loads(manifest_path.read_text())
    except (OSError,ValueError):manifest={}
    token=globals().get('_COMFY_CIVITAI_TOKEN') or globals().get('CIVITAI_TOKEN','')
    if not token:
        from google.colab import userdata
        token=(userdata.get('CIVITAI_API_TOKEN') or '').strip()
    def install(spec):
        target=model_root/spec['folder']/spec['name'];target.parent.mkdir(parents=True,exist_ok=True)
        if verified_model(target,spec,manifest):save_manifest(manifest_path,manifest)
        else:
            if target.exists():raise RuntimeError('已有同名文件验证失败，已保留：'+str(target))
            current=dict(spec)
            if current.get('civitai_only'):current['_resolved_download_url']=resolve_civitai_download(current,token)
            download_with_aria2(current,target,manifest_path,manifest,token)
        local_folder=workspace/'models'/spec['folder'].split('/')[0];local_folder.mkdir(parents=True,exist_ok=True)
        link=local_folder/spec['name']
        if link.is_symlink():
            if link.resolve()!=target.resolve():raise RuntimeError('模型链接冲突：'+str(link))
        elif link.exists():raise RuntimeError('同名本地模型已存在，已保留：'+str(link))
        else:link.symlink_to(target)
        return spec['name']
    failures=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(install,spec):spec for spec in specs}
        for future in as_completed(pending):
            try:print('编辑权重就绪：',future.result(),flush=True)
            except Exception as error:failures.append(pending[future]['name']+': '+str(error))
    if failures:raise RuntimeError('编辑权重未全部完成，已有文件保留，重跑可续传：\n'+'\n'.join(failures))
    def api(path):
        with urllib.request.urlopen('http://127.0.0.1:8188'+path,timeout=10) as response:return json.load(response)
    required=['Krea2EditModelPatch','Krea2EditGroundedEncode','Krea2AnyPaintPrepare','Krea2AnyPaintEncode','Krea2AnyPaintModelPatch','ApplyCosmosReferenceLatent','AnimaLLLiteApply']
    try:oi=api('/object_info')
    except (OSError,ValueError):
        print('安装完成。ComfyUI 尚未启动，继续运行第 5 节即可。',flush=True);return
    if all(name in oi for name in required):print('编辑节点已在线，无需重启。',flush=True);return
    print('节点安装完成；等待现有任务结束后，仅重启 ComfyUI，保留当前运行时和模型文件。',flush=True)
    while True:
        queue=api('/queue')
        if not queue['queue_running'] and not queue['queue_pending']:break
        print('等待已有生图任务完成…',flush=True);time.sleep(5)
    pid_path=Path('/content/comfyui-drive.pid');command_path=Path('/content/comfyui-drive-command.json')
    if not pid_path.exists() or not command_path.exists():raise RuntimeError('找不到本笔记本的 ComfyUI 进程记录，未停止其他进程。请手动重启 ComfyUI 后继续。')
    pid=int(pid_path.read_text());command=json.loads(command_path.read_text())
    proc_cmd=Path('/proc')/str(pid)/'cmdline'
    if not proc_cmd.exists() or str(workspace/'main.py').encode() not in proc_cmd.read_bytes():raise RuntimeError('ComfyUI PID 不匹配，未停止其他进程。')
    queue=api('/queue')
    if queue['queue_running'] or queue['queue_pending']:raise RuntimeError('出现新的生图任务，请完成后重跑本单元格。')
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        try:api('/system_stats')
        except OSError:break
        time.sleep(.5)
    else:raise RuntimeError('ComfyUI 尚未退出，已停止重启操作，请检查日志。')
    log=Path('/content/comfyui-drive.log').open('ab',buffering=0)
    process=subprocess.Popen(command,cwd=workspace,stdout=log,stderr=log,start_new_session=True)
    globals()['_comfy_edit_process']=process;pid_path.write_text(str(process.pid))
    for _ in range(180):
        if process.poll() is not None:raise RuntimeError('ComfyUI 启动失败，请检查 /content/comfyui-drive.log。')
        try:
            oi=api('/object_info')
            missing=[name for name in required if name not in oi]
            if missing:raise RuntimeError('节点导入失败：'+', '.join(missing)+'。请检查日志。')
            print('编辑升级完成：Identity / Edit / 局部重绘节点均已在线；网页刷新即可。',flush=True);return
        except OSError:time.sleep(1)
    raise RuntimeError('ComfyUI 启动等待超时，请检查日志。')

EDIT_MODELS=[{'name': 'krea2_identity_edit_v2_r128.safetensors', 'folder': 'loras/Krea2', 'repo': 'conradlocke/krea2-identity-edit', 'remote': 'krea2_identity_edit_v2_r128.safetensors', 'size': 914188768, 'sha256': '2082515f1eb66532f5efb5343eb15048a29e3abe9f356e35e784cc60d00b45f7'}, {'name': 'krea2_general_edit_v2_r128.safetensors', 'folder': 'loras/Krea2', 'repo': 'conradlocke/krea2-identity-edit', 'remote': 'krea2_general_edit_v2_r128.safetensors', 'size': 914188760, 'sha256': 'fbca6ab74b9bef3db8de1d4a49b93d4fe7a778f5d4ae6995e0769ea8de736f00'}, {'name': 'krea2_anypaint_rank32.safetensors', 'folder': 'loras/Krea2', 'repo': 'yijunwang2/krea2-anypaint', 'remote': 'krea2_anypaint_rank32.safetensors', 'size': 228587752, 'sha256': '3a7d09f6b27f8ead160d340f2f59c11f4ee635c4a1ee87ffd8b1b9f9ba412f7a'}, {'name': 'anima-lllite-inpainting-v2.safetensors', 'folder': 'model_patches', 'repo': 'Comfy-Org/Anima-LLLite', 'remote': 'model_patches/anima-lllite-inpainting-v2.safetensors', 'size': 65772872, 'sha256': '5242e677d2be34ee70ca7c97c3b14ff5ee49838c03fc1e60ac4852a180db6ef5'}, {'name': 'AnimeEditV2.safetensors', 'folder': 'loras/Anima', 'version_id': 3089149, 'file_id': 2968664, 'size': 367104408, 'sha256': '1a3b14064831c144de293df208f67a1f3f32cf43cae8ce5df46f337b21765218', 'civitai_only': True, 'download_url': 'https://civitai.com/api/download/models/3089149?fileId=2968664'}]
setup_edit_runtime(EDIT_MODELS)
