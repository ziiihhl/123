"""Narrow Colab worker: delete only image files named by a confirmed trash record."""
import hashlib,json,os,re,threading,time
from pathlib import Path,PurePosixPath

def image_delete_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temp.replace(path)

def image_delete_ref(image):
    if not isinstance(image,dict):raise ValueError('图片引用无效。')
    kind=image.get('type','output');name=image.get('filename') or image.get('comfy_filename')
    folder=image.get('subfolder') or ''
    if kind not in ('output','temp') or not isinstance(name,str) or not isinstance(folder,str):raise ValueError('只允许删除输出或预览图片。')
    if not name or '/' in name or '\\' in name or ':' in name or Path(name).suffix.lower() not in ('.png','.jpg','.jpeg','.webp','.gif','.bmp','.tif','.tiff','.avif'):raise ValueError('图片文件名无效。')
    folder=folder.replace('\\','/');parts=PurePosixPath(folder).parts
    if PurePosixPath(folder).is_absolute() or '..' in parts or ':' in folder:raise ValueError('图片子目录无效。')
    return dict(type=kind,filename=name,subfolder='/'.join(parts))

def image_delete_digest(images):
    return hashlib.sha256(json.dumps(images,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def delete_record_images(request,roots):
    if request.get('operation')!='delete_job_images':raise ValueError('不支持的图片操作。')
    refs=request.get('images')
    if not isinstance(refs,list) or len(refs)>1000:raise ValueError('图片清单无效。')
    refs=[image_delete_ref(image) for image in refs]
    if request.get('digest')!=image_delete_digest(refs):raise ValueError('图片清单校验失败，未执行删除。')
    targets=[]
    # Validate every target before deleting anything; never accept absolute paths from requests.
    for ref in refs:
        root=Path(roots[ref['type']]).resolve();path=root/ref['subfolder']/ref['filename']
        resolved=path.resolve()
        if not resolved.is_relative_to(root) or resolved==root:raise ValueError('图片路径超出输出目录，未执行删除。')
        if path.exists() and not path.is_file():raise ValueError('目标不是图片文件，未执行删除。')
        if str(resolved) not in [str(p) for p in targets]:targets.append(resolved)
    removed=[];missing=[];errors=[]
    for path in targets:
        try:
            if not path.exists():missing.append(str(path));continue
            path.unlink();removed.append(str(path))
        except OSError as error:errors.append(str(error))
    return dict(removed=removed,missing=missing,errors=errors,complete=not errors)

def start_image_delete_manager():
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('请先挂载 Google Drive。')
    thread=globals().get('_image_delete_manager_thread')
    if thread and thread.is_alive():print('回收站图片清理程序已在线。',flush=True);return
    persistent=Path(globals().get('PERSISTENT','/content/drive/MyDrive/ComfyUI_Persistent'))
    workspace=Path(globals().get('WORKSPACE','/content/ComfyUI'))
    roots={'output':persistent/'output','temp':workspace/'temp'}
    user=persistent/'user/default';queue=user/'image_delete_jobs';queue.mkdir(parents=True,exist_ok=True)
    def run():
        while True:
            try:
                for path in sorted(queue.glob('*.json')):
                    if not re.fullmatch(r'[a-f0-9]{32}\.json',path.name):continue
                    request=json.loads(path.read_text(encoding='utf-8'))
                    if request.get('status') not in ('pending','running'):continue
                    try:
                        request.update(status='running',updated_at=time.time());image_delete_write(path,request)
                        result=delete_record_images(request,roots)
                        request.update(status='success' if result['complete'] else 'failed',result=result,error='；'.join(result['errors']))
                    except Exception as error:request.update(status='failed',error=str(error))
                    request['updated_at']=time.time();image_delete_write(path,request)
                image_delete_write(user/'image_delete_state.json',dict(protocol=1,updated_at=time.time(),capabilities=['delete_job_images']))
            except Exception as error:
                try:image_delete_write(user/'image_delete_state.json',dict(protocol=1,updated_at=time.time(),error=str(error)))
                except OSError:pass
            time.sleep(3)
    thread=threading.Thread(target=run,name='image-delete-manager',daemon=True)
    globals()['_image_delete_manager_thread']=thread;thread.start()
    print('回收站图片清理程序已启动；仅在网页确认永久删除后处理对应图片，不重启或下载。',flush=True)

# Self-test touches only newly created temporary files, never existing output images.
import tempfile
with tempfile.TemporaryDirectory(prefix='comfy-image-cleanup-test-') as test_dir:
    test_root=Path(test_dir);test_output=test_root/'output';test_output.mkdir()
    (test_output/'delete-test.png').write_bytes(b'test-only')
    (test_output/'keep-test.png').write_bytes(b'keep')
    test_images=[dict(type='output',subfolder='',filename='delete-test.png')]
    result=delete_record_images(dict(operation='delete_job_images',images=test_images,digest=image_delete_digest(test_images)),{'output':test_output,'temp':test_root/'temp'})
    assert result['complete'] and len(result['removed'])==1
    assert (test_output/'keep-test.png').read_bytes()==b'keep'
print('图片清理自检通过：仅删除了临时测试图片。',flush=True)
start_image_delete_manager()
