"""Drive-backed opt-in startup models; never execute data from download records."""
def startup_model_spec(spec):
    import re
    name=spec.get('name','');folder=spec.get('folder');size=spec.get('size');sha=str(spec.get('sha256','')).lower()
    if not isinstance(name,str) or not name.endswith('.safetensors') or '/' in name or '\\' in name or name.startswith('.'):
        raise ValueError('Invalid startup model filename')
    if folder not in ('checkpoints','diffusion_models') or not isinstance(size,int) or size<=0 or not re.fullmatch('[a-f0-9]{64}',sha):
        raise ValueError('Invalid startup model metadata')
    if any(not isinstance(spec.get(key),int) or spec[key]<=0 for key in ('version_id','file_id')):raise ValueError('Invalid Civitai identifiers')
    return dict(name=name,folder=folder,size=size,sha256=sha,version_id=spec['version_id'],file_id=spec['file_id'])

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
    known={(s.get('version_id'),s.get('file_id')) for s in result if s.get('file_id')}
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
                job=json.loads((user_directory/'lora_download_jobs'/(job_id+'.json')).read_text(encoding='utf-8'))
                if job.get('status')!='success' or job.get('file_id')!=spec['file_id'] or job.get('version_id')!=spec['version_id']:continue
                if (job.get('result') or {}).get('sha256','').lower()!=spec['sha256']:continue
                registered_at=max(registered_at,float(job.get('created_at',0)))
            if blocked(spec,registered_at):continue
            identity=(spec['version_id'],spec['file_id']);filename=(spec['folder'],spec['name'])
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
    for path in (user_dir/'lora_download_jobs').glob('*.json'):
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
    for path in (user_dir/'lora_download_jobs').glob('*.json'):
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

# Exercise only tiny fake files in a temporary directory, never existing weights.
import tempfile
with tempfile.TemporaryDirectory(prefix='comfy-model-manager-test-') as _test_dir:
    _test_root=Path(_test_dir)
    _test_roots={key:_test_root/key for key in ('local','drive','comfy')}
    for _base in _test_roots.values():(_base/'checkpoints').mkdir(parents=True)
    (_test_roots['local']/'checkpoints/__management_selftest__.safetensors').write_bytes(b'test-only')
    (_test_roots['drive']/'checkpoints/__management_selftest__.safetensors').write_bytes(b'test-only')
    (_test_roots['local']/'checkpoints/keep.safetensors').write_bytes(b'keep')
    _test_model=next(item for item in primary_inventory(_test_roots) if item['name']=='__management_selftest__.safetensors')
    _test_result=delete_primary_model(dict(model_id=_test_model['id'],snapshot=_test_model['snapshot']),_test_roots,_test_root/'user',lambda:{})
    assert len(_test_result['removed'])==2
    assert (_test_roots['local']/'checkpoints/keep.safetensors').read_bytes()==b'keep'
print('主模型管理自检通过：仅操作了临时测试文件。',flush=True)
start_primary_model_manager()
