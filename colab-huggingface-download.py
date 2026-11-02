if 'download_with_aria2' not in globals() or '_lora_index_lock' not in globals():
    raise RuntimeError('请先运行当前笔记本第 4D 节。无需重启或重新下载模型。')
"""Hugging Face model repositories and exact, commit-pinned safetensors files. Shared with Colab."""
import hashlib,json,re,urllib.parse,urllib.request,urllib.error

HF_TARGETS={'hassaku':'SDXL','anima':'Anima','krea_anime':'Krea2','lustify':'Krea2'}
class HuggingFaceDownloadError(ValueError):pass

def hf_path(value,empty=False):
    from pathlib import PurePosixPath
    if not isinstance(value,str) or '\\' in value or ':' in value or any(ord(c)<32 for c in value):raise HuggingFaceDownloadError('Hugging Face 路径无效。')
    parts=PurePosixPath(value).parts
    if (not parts and not empty) or value.startswith('/') or '..' in parts:raise HuggingFaceDownloadError('Hugging Face 路径无效。')
    return '/'.join(parts)

def parse_hf_link(link):
    try:p=urllib.parse.urlsplit(link.strip());port=p.port
    except (ValueError,AttributeError):raise HuggingFaceDownloadError('Hugging Face 链接格式无效。') from None
    if p.scheme!='https' or p.hostname not in ('huggingface.co','www.huggingface.co') or p.username or port not in (None,443):raise HuggingFaceDownloadError('请使用 https://huggingface.co/ 的模型仓库或文件链接。')
    if any(k.lower() in ('token','authorization','access_token') for k in urllib.parse.parse_qs(p.query)):raise HuggingFaceDownloadError('请移除链接中的 Token，改在 Colab Secrets 中设置 HF_TOKEN。')
    parts=p.path.strip('/').split('/')
    if len(parts)<2 or parts[0] in ('datasets','spaces','api'):raise HuggingFaceDownloadError('目前支持模型仓库，不支持 datasets 或 Spaces。')
    repo='/'.join(parts[:2])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*',repo):raise HuggingFaceDownloadError('Hugging Face 仓库名无效。')
    revision='main';remote='';directory=''
    if len(parts)>2:
        if len(parts)<4 or parts[2] not in ('blob','resolve','tree'):raise HuggingFaceDownloadError('请使用仓库首页、tree、blob 或 resolve 文件链接。')
        revision=hf_path(urllib.parse.unquote(parts[3]))
        path=hf_path(urllib.parse.unquote('/'.join(parts[4:])),empty=True)
        if parts[2]=='tree':directory=path
        else:remote=path
    return dict(repo=repo,revision=revision,remote=remote,directory=directory)

class HFNoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def hf_json(path,token=''):
    if token and (len(token)>2048 or any(c.isspace() for c in token)):raise HuggingFaceDownloadError('HF_TOKEN 格式无效，请填写 Token 本身。')
    headers={'User-Agent':'Comfy-model-downloader','Accept':'application/json'}
    if token:headers['Authorization']='Bearer '+token
    try:
        with urllib.request.build_opener(HFNoRedirect).open(urllib.request.Request('https://huggingface.co'+path,headers=headers),timeout=30) as r:return json.loads(r.read(16*1024*1024))
    except urllib.error.HTTPError as e:
        error=HuggingFaceDownloadError('HF 需要授权：请在 Colab Secrets 添加 HF_TOKEN，并确认已获仓库访问权限。' if e.code in (401,403,404) else 'Hugging Face 返回 HTTP '+str(e.code));error.status_code=e.code
        raise error from None
    except (OSError,ValueError):raise HuggingFaceDownloadError('无法读取 Hugging Face 仓库，请稍后重试。') from None

def hf_family(value):
    value=re.sub('[^a-z0-9]','',str(value).lower())
    if 'krea2' in value:return 'Krea2'
    if 'anima' in value:return 'Anima'
    if any(s in value for s in ('sdxl','stablediffusionxl','illustrious','noobai','pony')):return 'SDXL'
    return None

def resolve_hf_model(link,target,revision=None,token=''):
    parsed=parse_hf_link(link);rev=hf_path(revision or parsed['revision']);repo=parsed['repo']
    data=hf_json('/api/models/'+repo+'/revision/'+urllib.parse.quote(rev,safe='')+'?blobs=true',token)
    commit=data.get('sha','')
    if not re.fullmatch('[a-f0-9]{40,64}',commit):raise HuggingFaceDownloadError('仓库没有返回有效的固定提交版本。')
    kind='checkpoint' if target.startswith('model:') else 'lora';destination='drive';base=target
    if kind=='checkpoint':
        parts=target.split(':')
        if len(parts)!=3 or parts[1] not in ('local','drive'):raise HuggingFaceDownloadError('保存位置无效。')
        _,destination,base=parts
    if base!='auto' and base not in HF_TARGETS:raise HuggingFaceDownloadError('请选择兼容的 LoRA 目标模型。')
    card=data.get('cardData') or {};files=[]
    if not isinstance(card,dict):card={}
    for item in data.get('siblings') or []:
        name=item.get('rfilename','')
        if not name.lower().endswith('.safetensors') or re.search(r'-\d{5}-of-\d{5}\.safetensors$',name):continue
        hf_path(name)
        if parsed['remote'] and name!=parsed['remote']:continue
        if parsed['directory'] and not name.startswith(parsed['directory'].rstrip('/')+'/'):continue
        if kind=='checkpoint' and any(part in name.lower().split('/')[:-1] for part in ('text_encoders','text_encoder','text_encoder_2','vae','loras','model_patches','unet','transformer')):continue
        lfs=item.get('lfs') or {};sha=lfs.get('sha256') or lfs.get('oid') or '';size=item.get('size') or lfs.get('size')
        if not re.fullmatch('[a-f0-9]{64}',sha) or not isinstance(size,int) or size<=0:continue
        fid=int(hashlib.sha256((repo+'\0'+commit+'\0'+name).encode()).hexdigest()[:12],16)
        family=hf_family(str(card.get('base_model',''))+' '+repo+' '+name+' '+str(data.get('tags',[])))
        files.append(dict(id=fid,name=name,size=size,sha256=sha,primary=False,family=family))
    if not files:raise HuggingFaceDownloadError('没有找到带 SHA256 的单文件 safetensors 权重；目前不下载整套 Diffusers 分片，也不会把编码器或 VAE 当成主模型。')
    families={f['family'] for f in files if f['family']}
    family=next(iter(families)) if len(families)==1 else None
    if kind=='checkpoint' and base=='auto':
        base={'SDXL':'hassaku','Anima':'anima','Krea2':'krea_anime'}.get(family)
        if not base:raise HuggingFaceDownloadError('无法确认主模型属于当前支持的 SDXL / Illustrious、Anima 或 Krea2 架构。请使用具体模型文件链接。')
        target='model:'+destination+':'+base
    compatible=not family or family==HF_TARGETS.get(base)
    warning='' if family else '仓库未明确标注基础架构，请确认与所选目标兼容。'
    if not compatible:warning='基础架构与所选目标不匹配。'
    return dict(source='huggingface',target=target,kind=kind,destination=destination,family=family or HF_TARGETS[base],repo=repo,revision=commit,model_name=repo,version_name=commit[:12],version_id=commit,base_model=str(card.get('base_model') or family or ''),compatible=compatible,warning=warning,files=files,versions=[dict(id=commit,name=commit[:12])],requested_file_id=files[0]['id'] if parsed['remote'] else None,gated=bool(data.get('gated')),private=bool(data.get('private')))

def choose_hf_file(info,file_id):
    if not info.get('compatible'):raise HuggingFaceDownloadError(info.get('warning','基础架构不匹配。'))
    file=next((f for f in info['files'] if f['id']==file_id),None)
    if not file:raise HuggingFaceDownloadError('请选择解析列表中的具体文件。')
    if file.get('family') and file['family']!=info['family']:raise HuggingFaceDownloadError('此文件的基础架构与目标不匹配。')
    stem=re.sub(r'[^\w.()\[\] -]','_',file['name'].split('/')[-1][:-12]).strip('. ')[:100] or 'model'
    name=stem+'__hf_'+file['sha256'][:10]+'.safetensors'
    return dict(source='huggingface',name=name,repo=info['repo'],revision=info['revision'],remote=file['name'],size=file['size'],sha256=file['sha256'],kind=info['kind'],destination=info['destination'],family=info['family'],folder=('checkpoints' if info['family']=='SDXL' else 'diffusion_models') if info['kind']=='checkpoint' else info['family'],trained_words=[],model_name=info['model_name'])

def resolve_hf_download(spec,token=''):
    repo=parse_hf_link('https://huggingface.co/'+spec['repo'])['repo'];remote=hf_path(spec['remote']);rev=hf_path(spec['revision'])
    url='https://huggingface.co/'+repo+'/resolve/'+urllib.parse.quote(rev,safe='')+'/'+urllib.parse.quote(remote,safe='/')
    for _ in range(8):
        p=urllib.parse.urlsplit(url);host=p.hostname or ''
        allowed=host=='huggingface.co' or host.endswith('.huggingface.co') or host=='hf.co' or host.endswith('.hf.co')
        if p.scheme!='https' or not allowed or p.username or p.port not in (None,443):raise HuggingFaceDownloadError('HF 返回了非官方 CDN 地址，已停止。')
        headers={'User-Agent':'Comfy-model-downloader'}
        if token and host=='huggingface.co':headers['Authorization']='Bearer '+token
        try:
            with urllib.request.build_opener(HFNoRedirect).open(urllib.request.Request(url,headers=headers,method='HEAD'),timeout=30) as r:
                if token and host=='huggingface.co':
                    # Private weights normally redirect to a signed CDN URL. Never forward bearer tokens to aria2 redirects.
                    try:
                        with urllib.request.build_opener(HFNoRedirect).open(urllib.request.Request(url,method='HEAD'),timeout=20):pass
                    except urllib.error.HTTPError:raise HuggingFaceDownloadError('此授权文件没有返回可安全下载的 CDN 地址。') from None
                return url
        except urllib.error.HTTPError as e:
            if e.code in (301,302,303,307,308) and e.headers.get('Location'):
                url=urllib.parse.urljoin(url,e.headers['Location']);continue
            raise HuggingFaceDownloadError('Hugging Face 下载授权失败（HTTP '+str(e.code)+'）；请检查 HF_TOKEN 和仓库许可。') from None
    raise HuggingFaceDownloadError('HF 下载重定向过多。')

"""Drive-backed opt-in startup models; never execute data from download records."""
def startup_model_spec(spec):
    import re
    name=spec.get('name','');folder=spec.get('folder');size=spec.get('size');sha=str(spec.get('sha256','')).lower()
    if not isinstance(name,str) or not name.endswith('.safetensors') or '/' in name or '\\' in name or name.startswith('.'):
        raise ValueError('Invalid startup model filename')
    if folder not in ('checkpoints','diffusion_models') or not isinstance(size,int) or size<=0 or not re.fullmatch('[a-f0-9]{64}',sha):
        raise ValueError('Invalid startup model metadata')
    if spec.get('source')=='huggingface':
        from pathlib import PurePosixPath
        repo=spec.get('repo','');revision=spec.get('revision','');remote=spec.get('remote','')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*',repo) or not re.fullmatch('[a-f0-9]{40,64}',revision):raise ValueError('Invalid HF repository/revision')
        if not isinstance(remote,str) or not remote.endswith('.safetensors') or PurePosixPath(remote).is_absolute() or '..' in PurePosixPath(remote).parts or '\\' in remote or ':' in remote:raise ValueError('Invalid HF file path')
        return dict(source='huggingface',name=name,folder=folder,size=size,sha256=sha,repo=repo,revision=revision,remote=remote)
    if any(not isinstance(spec.get(key),int) or spec[key]<=0 for key in ('version_id','file_id')):raise ValueError('Invalid Civitai identifiers')
    return dict(name=name,folder=folder,size=size,sha256=sha,version_id=spec['version_id'],file_id=spec['file_id'])

def startup_model_identity(spec):
    return ('huggingface',spec.get('repo'),spec.get('revision'),spec.get('remote')) if spec.get('source')=='huggingface' else ('civitai',spec.get('version_id'),spec.get('file_id'))

def load_startup_models(models,user_directory):
    import json,re
    from pathlib import Path
    user_directory=Path(user_directory)
    disabled=[]
    for path in (user_directory/'disabled_models').glob('*.json'):
        try:
            item=json.loads(path.read_text(encoding='utf-8'))
            if item.get('folder') in ('checkpoints','diffusion_models') and isinstance(item.get('name'),str):disabled.append(item)
        except (OSError,ValueError):continue
    def blocked(spec,registered_at=0):
        for item in disabled:
            identity=[spec.get('version_id'),spec.get('file_id')]
            matches=(spec.get('folder')==item['folder'] and spec.get('name')==item['name']) or (identity[1] and identity in item.get('identities',[]))
            if matches and registered_at<=item.get('deleted_at',float('inf')):return True
        return False
    result=[spec for spec in models if not blocked(spec)]
    known={startup_model_identity(s) for s in result}
    names={(s.get('folder'),s.get('name')) for s in result}
    registry=user_directory/'startup_models'
    if not registry.is_dir():return result
    for path in sorted(registry.glob('*.json'))[:2000]:
        try:
            entry=json.loads(path.read_text(encoding='utf-8'));spec=startup_model_spec(entry['spec'])
            registered_at=path.stat().st_mtime
            if not entry.get('completed'):
                job_id=entry.get('job_id','')
                if not isinstance(job_id,str) or not re.fullmatch('[a-f0-9]{32}',job_id):continue
                job=json.loads((user_directory/('hf_download_jobs' if spec.get('source')=='huggingface' else 'lora_download_jobs')/(job_id+'.json')).read_text(encoding='utf-8'))
                if job.get('status')!='success' or job.get('model_deleted'):continue
                if spec.get('source')=='huggingface':
                    if any(job.get(k)!=spec[k] for k in ('repo','revision','remote')):continue
                elif job.get('file_id')!=spec['file_id'] or job.get('version_id')!=spec['version_id']:continue
                if (job.get('result') or {}).get('sha256','').lower()!=spec['sha256']:continue
                registered_at=max(registered_at,float(job.get('created_at',0)))
            if blocked(spec,registered_at):continue
            identity=startup_model_identity(spec);filename=(spec['folder'],spec['name'])
            if identity in known or filename in names:continue
            result.append(spec);known.add(identity);names.add(filename)
            print('记事本启动列表：',spec['name'],flush=True)
        except (OSError,ValueError,TypeError,KeyError):continue
    return result

def remember_direct_model(user_directory,spec):
    import json,uuid
    from pathlib import Path
    spec=startup_model_spec(spec);root=Path(user_directory)/'startup_models';root.mkdir(parents=True,exist_ok=True)
    target=root/(uuid.uuid4().hex+'.json');temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(version=1,spec=spec,completed=True),ensure_ascii=False),encoding='utf-8');temporary.replace(target)

"""Narrow Colab primary-model inventory/deletion worker. No arbitrary commands or paths."""
import hashlib,json,os,re,threading,time,urllib.request
from pathlib import Path,PurePosixPath

def primary_relative_name(name):
    name=str(name).replace('\\','/');parts=PurePosixPath(name).parts
    if not parts or PurePosixPath(name).is_absolute() or '..' in parts or ':' in name:raise ValueError('模型名称无效。')
    if parts[0] in ('DownloadsLocal','DownloadsDrive'):parts=parts[1:]
    if not parts or not parts[-1].endswith(('.safetensors','.ckpt','.gguf')):raise ValueError('只允许操作主模型权重文件。')
    return '/'.join(parts)

def primary_model_key(folder,name):
    if folder not in ('checkpoints','diffusion_models'):raise ValueError('模型类型无效。')
    return hashlib.sha256((folder+'\0'+primary_relative_name(name)).encode()).hexdigest()[:32]

def _primary_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temp.replace(path)

def primary_inventory(roots,aliases=None):
    groups={};aliases=aliases or {}
    allowed=[(Path(base)/folder).resolve() for base in roots.values() for folder in ('checkpoints','diffusion_models')]
    for storage,base in roots.items():
        for folder in ('checkpoints','diffusion_models'):
            directory=Path(base)/folder
            if not directory.is_dir():continue
            for path in directory.rglob('*'):
                if not path.is_file() or path.suffix not in ('.safetensors','.ckpt','.gguf'):continue
                resolved=path.resolve()
                if not any(resolved.is_relative_to(root) for root in allowed):continue
                try:name=primary_relative_name(path.relative_to(directory).as_posix())
                except ValueError:continue
                mid=primary_model_key(folder,name);entry=groups.setdefault(mid,dict(id=mid,folder=folder,name=name,aliases=[],files=[],links=[]))
                stat=path.stat()
                if not any(item['path']==str(resolved) for item in entry['files']):
                    place='drive' if resolved.is_relative_to(Path(roots['drive']).resolve()) else 'local'
                    entry['files'].append(dict(path=str(resolved),storage=place,size=stat.st_size,mtime_ns=stat.st_mtime_ns))
                if path.is_symlink():entry['links'].append(str(path))
    for mid,entry in groups.items():
        entry['aliases']=[name for name in aliases.get(entry['folder'],[]) if primary_model_key(entry['folder'],name)==mid]
        entry['files'].sort(key=lambda item:item['path'])
        entry['snapshot']=hashlib.sha256(json.dumps(entry['files'],sort_keys=True).encode()).hexdigest()
        entry['bytes']=sum(item['size'] for item in entry['files'])
    return sorted(groups.values(),key=lambda entry:(entry['folder'],entry['name']))

def primary_queue_keys(queue):
    keys=set()
    for item in queue.get('queue_running',[])+queue.get('queue_pending',[]):
        for node in item[2].values():
            kind=node.get('class_type');inputs=node.get('inputs',{})
            folder='checkpoints' if kind=='CheckpointLoaderSimple' else 'diffusion_models' if kind in ('UNETLoader','UnetLoaderGGUF','UnetLoaderGGUFAdvanced') else None
            if folder:
                try:keys.add(primary_model_key(folder,inputs.get('ckpt_name') or inputs.get('unet_name')))
                except ValueError:pass
    return keys

def delete_primary_model(request,roots,user_dir,queue_provider):
    mid=request.get('model_id','');expected=request.get('snapshot','')
    if not re.fullmatch('[a-f0-9]{32}',mid):raise ValueError('模型编号无效。')
    current=next((entry for entry in primary_inventory(roots) if entry['id']==mid),None)
    if current is None:raise ValueError('模型已不存在，请刷新列表。')
    if current['snapshot']!=expected:raise ValueError('模型文件已变化，未执行删除。请刷新列表重新确认。')
    if mid in primary_queue_keys(queue_provider()):raise ValueError('模型正在被运行或排队的任务使用，未执行删除。')
    user_dir=Path(user_dir);registered=[];identities=[]
    for path in [p for folder in ('lora_download_jobs','hf_download_jobs') for p in (user_dir/folder).glob('*.json')]:
        try:
            job=json.loads(path.read_text(encoding='utf-8'))
            if job.get('kind')!='checkpoint' or job.get('status') in ('success','failed','cancelled'):continue
            folder='checkpoints' if str(job.get('target','')).endswith(':hassaku') else 'diffusion_models'
            if primary_model_key(folder,job.get('filename',''))==mid:raise RuntimeError('此模型仍有下载任务，请先完成或取消下载。')
        except (OSError,ValueError,KeyError,TypeError):continue
    for path in (user_dir/'startup_models').glob('*.json'):
        try:
            record=json.loads(path.read_text(encoding='utf-8'));spec=record['spec']
            if primary_model_key(spec['folder'],spec['name'])==mid:
                registered.append(path);identities.append([spec.get('version_id'),spec.get('file_id')])
        except (OSError,ValueError,KeyError,TypeError):continue
    for spec in globals().get('MODELS',[]):
        try:
            if primary_model_key(spec['folder'],spec['name'])==mid:identities.append([spec.get('version_id'),spec.get('file_id')])
        except (ValueError,KeyError):pass
    # Recheck immediately before the first destructive operation.
    if mid in primary_queue_keys(queue_provider()):raise ValueError('出现新的使用任务，未执行删除。')
    for item in current['files']:
        path=Path(item['path']);stat=path.stat()
        if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:raise ValueError('模型文件已变化，未执行删除。')
    disabled=dict(version=1,model_id=mid,folder=current['folder'],name=current['name'],identities=identities,deleted_at=time.time())
    _primary_write(user_dir/'disabled_models'/(mid+'.json'),disabled)
    removed=[];errors=[]
    for item in current['files']:
        try:Path(item['path']).unlink();removed.append(item['path'])
        except OSError as error:errors.append(str(error))
    for link in current['links']:
        path=Path(link)
        if path.is_symlink() and not path.exists():path.unlink()
    for path in registered:path.unlink(missing_ok=True)
    for path in [p for folder in ('lora_download_jobs','hf_download_jobs') for p in (user_dir/folder).glob('*.json')]:
        if not re.fullmatch(r'[a-f0-9]{32}\.json',path.name):continue
        try:
            job=json.loads(path.read_text(encoding='utf-8'))
            if job.get('kind')!='checkpoint':continue
            target=str(job.get('target','')).split(':')
            folder='checkpoints' if target[-1]=='hassaku' else 'diffusion_models'
            if primary_model_key(folder,job.get('filename',''))==mid:
                job.update(model_deleted=True,add_to_notebook=False);_primary_write(path,job)
        except (OSError,ValueError,KeyError,TypeError):continue
    if isinstance(globals().get('MODELS'),list):
        globals()['MODELS'][:]=[s for s in globals()['MODELS'] if s.get('folder') not in ('checkpoints','diffusion_models') or primary_model_key(s['folder'],s['name'])!=mid]
    result=dict(removed=removed,notebook_removed=True,model_id=mid,name=current['name'])
    if errors:result.update(partial=True,error='部分文件删除失败：'+'；'.join(errors))
    return result

def start_primary_model_manager():
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('请先挂载 Google Drive。')
    existing=globals().get('_primary_model_manager_thread')
    if existing and existing.is_alive():print('主模型管理程序已在线。',flush=True);return
    persistent=Path(globals().get('PERSISTENT','/content/drive/MyDrive/ComfyUI_Persistent'))
    workspace=Path(globals().get('WORKSPACE','/content/ComfyUI'))
    roots=dict(local=Path(globals().get('CACHE_MODELS','/content/comfy-model-cache')),drive=persistent/'models',comfy=workspace/'models')
    user_dir=persistent/'user/default';jobs=user_dir/'model_management_jobs';jobs.mkdir(parents=True,exist_ok=True)
    def api(path):
        with urllib.request.urlopen('http://127.0.0.1:8188'+path,timeout=10) as response:return json.load(response)
    def run():
        while True:
            try:
                for path in sorted(jobs.glob('*.json')):
                    if not re.fullmatch(r'[a-f0-9]{32}\.json',path.name):continue
                    job=json.loads(path.read_text(encoding='utf-8'))
                    if job.get('status') not in ('pending','running'):continue
                    try:
                        job.update(status='running',updated_at=time.time());_primary_write(path,job)
                        if job.get('operation')!='delete_primary':raise ValueError('不支持的模型管理操作。')
                        result=delete_primary_model(job,roots,user_dir,lambda:api('/queue'))
                        job.update(status='failed' if result.get('partial') else 'success',result=result,error=result.get('error',''))
                    except Exception as error:job.update(status='failed',error=str(error))
                    job['updated_at']=time.time();_primary_write(path,job)
                oi=api('/object_info');aliases={}
                for folder,node,field in [('checkpoints','CheckpointLoaderSimple','ckpt_name'),('diffusion_models','UNETLoader','unet_name')]:
                    aliases[folder]=oi.get(node,{}).get('input',{}).get('required',{}).get(field,[[]])[0]
                inventory=primary_inventory(roots,aliases);busy=primary_queue_keys(api('/queue'))
                for item in inventory:item['in_use']=item['id'] in busy
                _primary_write(user_dir/'model_management_state.json',dict(protocol=1,updated_at=time.time(),models=inventory,error=''))
            except Exception as error:
                _primary_write(user_dir/'model_management_state.json',dict(protocol=1,updated_at=time.time(),models=[],error='主模型管理暂时不可用：'+str(error)))
            time.sleep(5)
    thread=threading.Thread(target=run,name='primary-model-manager',daemon=True);globals()['_primary_model_manager_thread']=thread;thread.start()
    print('主模型删除与启动列表管理程序已启动，不重启、不下载模型。',flush=True)

def sources_for(spec, civitai_token):
    if spec.get('source')=='huggingface' and not spec.get('_resolved_download_url'):
        return [resolve_hf_download(spec,globals().get('_COMFY_HF_TOKEN',''))]
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
"""HF download and private metadata workers; shared progress engine, independent Civitai queue."""
import json,re,shutil,threading,time,uuid
from pathlib import Path

def hf_job_write(path,value):
    value['updated_at']=time.time();path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temporary.replace(path)

def record_hf_lora(root,index_path,target,spec):
    with _lora_index_lock:
        try:data=json.loads(Path(index_path).read_text(encoding='utf-8'))
        except (OSError,ValueError):data=dict(version=1,files={},errors={})
        stat=target.stat();relative=target.relative_to(root).as_posix();metadata=read_lora_header(target)
        fields=('modelspec.trigger_phrase','modelspec.trigger_words','trigger_words','triggerWords','trainedWords','activation_text','activation_tags','ss_activation_tag','ss_trigger_words','ss_tag_frequency')
        data.setdefault('files',{})[relative]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=spec['sha256'],metadata={k:metadata[k] for k in fields if k in metadata and len(str(metadata[k]))<=512*1024},huggingface={k:spec[k] for k in ('repo','revision','remote')})
        data.setdefault('errors',{}).pop(relative,None);data.update(version=1,updated_at=time.time());hf_job_write(Path(index_path),data)

def execute_hf_download(info,file_id,root,index_path,progress=None,cancelled=None):
    spec=choose_hf_file(info,file_id);root=Path(root);persistent=root.parent.parent
    models=(persistent/'models') if spec['destination']=='drive' else Path(globals().get('CACHE_MODELS','/content/comfy-model-cache'))
    base=models if spec['kind']=='checkpoint' else root
    target=base/spec['folder']/spec['name'];target.parent.mkdir(parents=True,exist_ok=True);target.resolve().relative_to(base.resolve())
    manifest_path=persistent/'manifests/hf-downloads.json';manifest_path.parent.mkdir(parents=True,exist_ok=True)
    try:manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError,ValueError):manifest={}
    if cancelled and cancelled():raise LoraDownloadCancelled()
    if not verified_model(target,spec,manifest):
        if target.exists():raise HuggingFaceDownloadError('同名文件存在但校验失败，已保留：'+spec['name'])
        if shutil.disk_usage(base).free<spec['size']+128*1024**2:raise HuggingFaceDownloadError('目标空间不足。')
        if not shutil.which('aria2c'):raise HuggingFaceDownloadError('缺少 aria2，请先运行第 2 节。')
        spec['_resolved_download_url']=resolve_hf_download(spec,globals().get('_COMFY_HF_TOKEN',''))
        def report(state):
            if cancelled and cancelled():raise LoraDownloadCancelled()
            if progress:progress(state)
        download_with_aria2(spec,target,manifest_path,manifest,'',progress_callback=report)
    save_manifest(manifest_path,manifest)
    if spec['kind']=='lora':record_hf_lora(root,Path(index_path),target,spec);filename='Drive/'+target.relative_to(root).as_posix()
    else:
        alias='DownloadsDrive' if spec['destination']=='drive' else 'DownloadsLocal'
        link=Path(globals().get('WORKSPACE','/content/ComfyUI'))/'models'/spec['folder']/alias;link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink():
            if link.resolve()!=(models/spec['folder']).resolve():raise HuggingFaceDownloadError('模型目录链接冲突，已保留原目录。')
        elif link.exists():raise HuggingFaceDownloadError('模型目录已被占用，已保留原目录。')
        else:link.symlink_to(models/spec['folder'],target_is_directory=True)
        filename=alias+'/'+spec['name']
    return dict(filename=filename,size=spec['size'],sha256=spec['sha256'],source='huggingface',repo=spec['repo'],revision=spec['revision'],remote=spec['remote'],kind=spec['kind'])

def start_hf_download_worker(root,index_path):
    if 'download_with_aria2' not in globals() or '_lora_index_lock' not in globals():raise RuntimeError('请先运行第 4D 节加载下载和索引组件。')
    try:
        from google.colab import userdata
        globals()['_COMFY_HF_TOKEN']=(userdata.get('HF_TOKEN') or '').strip()
    except Exception:globals()['_COMFY_HF_TOKEN']=''
    print('HF_TOKEN：'+('已读取，不显示内容。' if globals()['_COMFY_HF_TOKEN'] else '未设置；公开仓库可用，授权仓库需添加 Token。'),flush=True)
    previous=globals().get('_hf_download_thread')
    if previous and previous.is_alive():print('HF 接收程序已在线。',flush=True);return
    root=Path(root);user=Path(index_path).parent;queue=user/'hf_download_jobs';queue.mkdir(parents=True,exist_ok=True)
    resolve_queue=user/'hf_resolve_jobs';resolve_queue.mkdir(parents=True,exist_ok=True)
    def heartbeat():hf_job_write(user/'hf_download_worker.json',dict(protocol=1,hf_token_loaded=bool(globals().get('_COMFY_HF_TOKEN'))))
    def resolve_jobs():
        while True:
            try:
                heartbeat()
                for path in sorted(resolve_queue.glob('*.json')):
                    if not re.fullmatch(r'[a-f0-9]{32}\.json',path.name):continue
                    job=json.loads(path.read_text())
                    if job.get('status')!='pending':continue
                    try:job.update(status='success',info=resolve_hf_model(job['url'],job['target'],job.get('revision'),globals().get('_COMFY_HF_TOKEN','')))
                    except Exception as e:job.update(status='failed',error=str(e)[:600] if isinstance(e,HuggingFaceDownloadError) else 'HF 解析失败，请检查网络与权限。')
                    hf_job_write(path,job)
            except Exception:pass
            time.sleep(2)
    def run():
        while True:
            for path in sorted(queue.glob('*.json')):
                if not re.fullmatch(r'[a-f0-9]{32}\.json',path.name):continue
                job=None
                try:
                    job=json.loads(path.read_text(encoding='utf-8'))
                    if job.get('status') in ('success','failed','cancelled'):continue
                    cancel=path.with_suffix('.cancel.json')
                    if cancel.exists():job['status']='cancelled';hf_job_write(path,job);continue
                    job.update(status='resolving');hf_job_write(path,job)
                    link='https://huggingface.co/'+job['repo']+'/blob/'+job['revision']+'/'+job['remote']
                    info=resolve_hf_model(link,job['target'],job['revision'],globals().get('_COMFY_HF_TOKEN',''))
                    spec=choose_hf_file(info,int(job['file_id']))
                    if spec['sha256']!=job['sha256'] or spec['size']!=job['size']:raise HuggingFaceDownloadError('文件元数据已变化，未下载。')
                    def progress(state):
                        job.update(status='verifying' if '校验' in state['status'] else 'downloading',progress=state);hf_job_write(path,job)
                    result=execute_hf_download(info,int(job['file_id']),root,index_path,progress,lambda:cancel.exists())
                    job.update(status='success',result=result,error=None);hf_job_write(path,job)
                except LoraDownloadCancelled:
                    if job:job.update(status='cancelled');hf_job_write(path,job)
                except Exception as e:
                    if job:job.update(status='failed',error=str(e)[:800] if isinstance(e,(HuggingFaceDownloadError,RuntimeError)) else 'HF 下载失败，请检查网络、空间及权限。');hf_job_write(path,job)
            time.sleep(2)
    metadata=threading.Thread(target=resolve_jobs,name='hf-metadata-worker',daemon=True);metadata.start()
    thread=threading.Thread(target=run,name='hf-download-worker',daemon=True);globals()['_hf_download_thread']=thread;thread.start()
    print('HF 主模型 / LoRA 下载程序已启动；不自动下载任何文件。',flush=True)

def resolve_download_link(link,target,version_id=None,token=''):
    if urllib.parse.urlsplit(link.strip()).hostname in ('huggingface.co','www.huggingface.co'):
        return resolve_hf_model(link,target,version_id,globals().get('_COMFY_HF_TOKEN',''))
    return resolve_civitai_lora(link,target,version_id,token)

def choose_download_file(info,file_id):return choose_hf_file(info,file_id) if info.get('source')=='huggingface' else choose_civitai_file(info,file_id)

def execute_download(info,file_id,root,index_path,progress=None,cancelled=None):
    if info.get('source')=='huggingface':return execute_hf_download(info,file_id,root,index_path,progress,cancelled)
    return execute_lora_download(info,file_id,root,index_path,progress,cancelled)

start_hf_download_worker(lora_drive,lora_index_path)
"""Interactive Civitai LoRA downloader; does nothing until the user confirms."""
import html
import ipywidgets as widgets
from IPython.display import display,clear_output

_dl_kind=widgets.Dropdown(options=[('LoRA / LoCon','lora'),('主模型','checkpoint')],description='下载类型：')
_dl_destination=widgets.Dropdown(options=[('Colab 本地','local'),('Google Drive','drive')],description='主模型保存：')
_dl_remember=widgets.Checkbox(value=False,description='加入记事本启动列表')
_dl_target=widgets.Dropdown(options=[(v['label'],k) for k,v in LORA_TARGETS.items()],description='目标模型：',style={'description_width':'initial'})
_dl_link=widgets.Text(value='',placeholder='Civitai 或 https://huggingface.co/作者/仓库',description='模型链接：',layout=widgets.Layout(width='95%'),style={'description_width':'initial'})
_dl_resolve=widgets.Button(description='解析链接',button_style='info')
_dl_version=widgets.Dropdown(options=[],description='版本：',layout=widgets.Layout(width='95%'))
_dl_file=widgets.Dropdown(options=[],description='权重文件：',layout=widgets.Layout(width='95%'),style={'description_width':'initial'})
_dl_confirm=widgets.Button(description='确认下载',button_style='success',disabled=True)
_dl_details=widgets.HTML();_dl_output=widgets.Output();_dl_state={'info':None,'busy':False}

def _show_download_preview(info):
    _dl_state['busy']=True;_dl_state['info']=info
    _dl_version.options=[(v['name']+' · '+str(v['id']),v['id']) for v in info['versions']]
    _dl_version.value=info['version_id']
    _dl_file.options=[(f['name']+' · '+f"{f['size']/1e6:.1f} MB",f['id']) for f in info['files']]
    requested=info.get('requested_file_id')
    if any(f['id']==requested for f in info['files']):_dl_file.value=requested
    else:_dl_file.value=next((f['id'] for f in info['files'] if f['primary']),info['files'][0]['id'])
    _dl_details.value='<p>'+html.escape(info['model_name']+' / '+info['version_name']+'；基础架构：'+str(info['base_model']))+'</p><p>'+html.escape(info['warning'])+'</p>'
    _dl_confirm.disabled=not info['compatible'];_dl_state['busy']=False

def _selected_download_target():
    return ('model:'+_dl_destination.value+':auto') if _dl_kind.value=='checkpoint' else _dl_target.value

def _resolve_lora_click(_):
    _dl_resolve.disabled=True;_dl_confirm.disabled=True
    with _dl_output:
        clear_output(wait=True)
        try:_show_download_preview(resolve_download_link(_dl_link.value,_selected_download_target(),token=_lora_token()))
        except (CivitaiLoraError,HuggingFaceDownloadError) as error:print(str(error))
    _dl_resolve.disabled=False

def _version_changed(change):
    if _dl_state['busy'] or change['new'] is None:return
    _dl_confirm.disabled=True
    with _dl_output:
        clear_output(wait=True)
        try:_show_download_preview(resolve_download_link(_dl_link.value,_selected_download_target(),version_id=change['new'],token=_lora_token()))
        except (CivitaiLoraError,HuggingFaceDownloadError) as error:print(str(error))

def _confirm_lora_click(_):
    info=_dl_state['info']
    if not info:return
    _dl_confirm.disabled=True;_dl_resolve.disabled=True
    with _dl_output:
        clear_output(wait=True)
        try:
            result=execute_download(info,_dl_file.value,lora_drive,lora_index_path)
            if info.get('kind')=='checkpoint' and _dl_remember.value:
                remember_direct_model(Path(lora_index_path).parent,choose_download_file(info,_dl_file.value))
                print('已加入记事本启动列表。')
            print('下载、SHA256 校验和索引已完成：',result['filename'])
            print('去网页刷新工作流，即可选择；触发词会自动获取。')
        except (CivitaiLoraError,HuggingFaceDownloadError,RuntimeError) as error:print(str(error))
    _dl_confirm.disabled=False;_dl_resolve.disabled=False

def _target_or_link_changed(_):
    _dl_remember.layout.display='' if _dl_kind.value=='checkpoint' else 'none'
    _dl_target.layout.display='none' if _dl_kind.value=='checkpoint' else ''
    _dl_destination.layout.display='' if _dl_kind.value=='checkpoint' else 'none'
    if not _dl_state['busy']:_dl_confirm.disabled=True;_dl_state['info']=None

_dl_resolve.on_click(_resolve_lora_click);_dl_confirm.on_click(_confirm_lora_click)
_dl_version.observe(_version_changed,names='value')
_dl_kind.observe(_target_or_link_changed,names='value');_dl_destination.observe(_target_or_link_changed,names='value');_dl_target.observe(_target_or_link_changed,names='value');_dl_link.observe(_target_or_link_changed,names='value')
_dl_destination.layout.display='none'
_dl_remember.layout.display='none'
display(widgets.VBox([_dl_kind,_dl_destination,_dl_remember,_dl_target,_dl_link,_dl_resolve,_dl_version,_dl_file,_dl_details,_dl_confirm,_dl_output]))
