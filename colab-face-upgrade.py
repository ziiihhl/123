"""Optional face detect/crop/refine pass, after hand repair and before model upscale.

Shared verbatim with the Colab notebook; only standard-library dependencies.
"""
import copy,math

FACE_MODEL='bbox/face_yolov8m.pt'
FACE_KEYS=('face_enabled','face_denoise','face_threshold')
FACE_TYPES={'provider':'UltralyticsDetectorProvider','detector':'BboxDetectorSEGS','detailer':'DetailerForEach','split':'ImpactImageBatchToImageList','merge':'ImageListToImageBatch'}

def face_widgets(node):
    values=node.get('widgets_values',[]);result={};index=0
    if node.get('type')=='KSampler' and not any(i.get('widget') for i in node.get('inputs',[])):
        names=['seed','steps','cfg','sampler_name','scheduler','denoise']
        for name in names:
            if index>=len(values):break
            result[name]=(index,values[index]);index+=1
            if name=='seed' and index<len(values) and values[index] in ('fixed','randomize','increment','decrement'):index+=1
        return result
    for item in node.get('inputs',[]):
        if not item.get('widget'):continue
        if index>=len(values):break
        result[item['name']]=(index,values[index]);index+=1
        if item['name'] in ('seed','noise_seed') and index<len(values) and values[index] in ('fixed','randomize','increment','decrement'):index+=1
    return result

def face_set(node,**values):
    fields=face_widgets(node)
    for key,value in values.items():
        if key not in fields:raise ValueError('面部修复节点缺少参数：'+key)
        node['widgets_values'][fields[key][0]]=value

def face_branch(raw):
    marker=(raw.get('extra') or {}).get('comfygen_face_detailer') or {}
    if marker.get('version')!=1:return {},{}
    nodes={n['id']:n for n in raw['nodes']};branch={role:nodes.get(marker.get(role)) for role in FACE_TYPES}
    if any(not node or node.get('type')!=FACE_TYPES[role] for role,node in branch.items()):raise ValueError('面部修复分支标记与节点不一致，请重新运行最新版第 5 节。')
    return marker,branch

def ensure_face_nodes(workflow):
    raw=copy.deepcopy(workflow);marker,branch=face_branch(raw)
    if marker:return raw,False
    hand=(raw.get('extra') or {}).get('comfygen_hand_detailer') or {}
    if hand.get('batch_version')!=1:return raw,False
    by_id={n['id']:n for n in raw['nodes']}
    if any(by_id.get(hand.get(role),{}).get('type')!=kind for role,kind in FACE_TYPES.items()):raise ValueError('面部修复需要完整的手部逐张处理基础分支。')
    oldids={hand[role] for role in FACE_TYPES};start=max(by_id)+1
    mapping={hand[role]:start+i for i,role in enumerate(FACE_TYPES)}
    links=raw['links'];outgoing=[link for link in links if link[1]==hand['merge'] and link[2]==0]
    if not outgoing:raise ValueError('找不到面部修复前的图像输出。')
    nextlink=max([l[0] for l in links]+[0])+1;newlinks=[];linkmap={}
    for link in list(links):
        if link[3] not in oldids:continue
        cloned=copy.deepcopy(link);linkmap[link[0]]=nextlink;cloned[0]=nextlink;nextlink+=1
        cloned[3]=mapping[link[3]]
        if link[3]==hand['split']:cloned[1:3]=[hand['merge'],0]
        elif link[1] in mapping:cloned[1]=mapping[link[1]]
        newlinks.append(cloned)
    for link in outgoing:link[1]=mapping[hand['merge']]
    for role in FACE_TYPES:
        node=copy.deepcopy(by_id[hand[role]]);node['id']=mapping[node['id']];node['order']=node['id']
        node['pos']=[node.get('pos',[0,0])[0],node.get('pos',[0,0])[1]+1050]
        node['title']={'provider':'面部 ADetailer · 检测模型','detector':'面部 ADetailer · 检测区域','detailer':'面部 ADetailer · 局部重绘','split':'面部 ADetailer · 逐张处理','merge':'面部 ADetailer · 合并批次'}[role]
        node['mode']=2 if role in ('provider','detector') else 4
        for item in node.get('inputs',[]):
            if item.get('link') is not None:item['link']=linkmap[item['link']]
        if role=='provider':node['widgets_values'][0]=FACE_MODEL
        if role=='detector':face_set(node,threshold=.3,dilation=8,crop_factor=2.0,drop_size=10,labels='all')
        if role=='detailer':face_set(node,guide_size=512,max_size=1024,denoise=.28,feather=12,wildcard='',cycle=1,noise_mask=True,force_inpaint=True)
        raw['nodes'].append(node)
    links.extend(newlinks)
    # Rebuild output bookkeeping so the saved UI graph has the same paths as the API graph.
    by_id={n['id']:n for n in raw['nodes']}
    for node in raw['nodes']:
        for output in node.get('outputs',[]):output['links']=[]
    for lid,src,slot,*_ in links:by_id[src]['outputs'][slot]['links'].append(lid)
    raw.setdefault('extra',{})['comfygen_face_detailer']=dict(version=1,**{role:mapping[hand[role]] for role in FACE_TYPES},model=FACE_MODEL,enabled=False,placement='after_hand_before_upscale')
    raw['last_node_id']=max(by_id);raw['last_link_id']=nextlink-1
    return raw,True

def describe_face(raw,oi):
    marker,branch=face_branch(raw)
    if not marker:return dict(supported=False,available=False,enabled=False,reason='当前工作流没有面部修复分支。')
    choices=oi.get('UltralyticsDetectorProvider',{}).get('input',{}).get('required',{}).get('model_name',[[]])[0]
    available=all(kind in oi for kind in FACE_TYPES.values()) and FACE_MODEL in choices
    return dict(supported=True,available=available,enabled=branch['detailer'].get('mode',0)==0,model=FACE_MODEL,denoise=face_widgets(branch['detailer'])['denoise'][1],threshold=face_widgets(branch['detector'])['threshold'][1],reason='' if available else '请运行面部修复升级单元格，安装检测权重后刷新工作流。',nodes={role:marker[role] for role in FACE_TYPES})

def face_for_description(raw):
    result=copy.deepcopy(raw);marker,branch=face_branch(result)
    if marker:
        for role,node in branch.items():node['mode']=2 if role in ('provider','detector') else 4
    return result

def configure_face(raw,values,oi):
    raw,_=ensure_face_nodes(raw);info=describe_face(raw,oi);enabled=values.get('face_enabled',info['enabled'])
    if enabled not in (True,False,0,1,'true','false'):raise ValueError('面部修复开关必须为 true 或 false。')
    enabled=enabled in (True,1,'true')
    if enabled and not info['available']:raise ValueError(info['reason'])
    if not info['supported']:return raw,dict(face_enabled=False)
    _,branch=face_branch(raw)
    for role,node in branch.items():node['mode']=0 if enabled else 2 if role in ('provider','detector') else 4
    used=dict(face_enabled=enabled)
    if enabled:
        def number(key,default,low,high):
            try:value=float(values.get(key,default))
            except (ValueError,TypeError):raise ValueError('面部修复参数无效：'+key) from None
            if not math.isfinite(value) or not low<=value<=high:raise ValueError('面部修复参数超出范围：'+key)
            return value
        denoise=number('face_denoise',.28,.05,.6);threshold=number('face_threshold',.3,.05,.95)
        face_set(branch['detailer'],denoise=denoise);face_set(branch['detector'],threshold=threshold)
        # Use the active main sampler's settings, not the old hand branch's Turbo defaults.
        sampler=next((n for n in raw['nodes'] if n['type']=='KSampler'),None)
        if sampler:
            defaults=face_widgets(sampler)
            selected={k:values.get(k,defaults.get(k,(None,None))[1]) for k in ('steps','cfg','sampler_name','scheduler')}
            selected={k:v for k,v in selected.items() if v not in (None,'')}
            if 'steps' in selected:selected['steps']=max(1,min(30,int(selected['steps'])))
            if 'cfg' in selected:selected['cfg']=float(selected['cfg'])
            face_set(branch['detailer'],**selected)
        used.update(face_denoise=denoise,face_threshold=threshold,face_model=FACE_MODEL)
    return raw,used

def sync_face_sampling(prompt,raw):
    marker,_=face_branch(raw)
    if not marker:return
    node=prompt.get(str(marker['detailer']))
    samplers=[n for n in prompt.values() if n['class_type']=='KSampler']
    if not node or len(samplers)!=1:return
    for key in ('seed','steps','cfg','sampler_name','scheduler'):
        if key in samplers[0]['inputs']:
            value=samplers[0]['inputs'][key]
            node['inputs'][key]=min(30,int(value)) if key=='steps' else value

def ensure_hand_batch_nodes(workflow):
    """Idempotently split decoded batches before hand detection, then merge results."""
    import copy
    raw=copy.deepcopy(workflow)
    marker=(raw.get('extra') or {}).get('comfygen_hand_detailer') or {}
    if marker.get('version')!=1:return raw,False
    nodes=raw.get('nodes',[]);links=raw.get('links',[])
    by_id={int(n['id']):n for n in nodes}
    types={'provider':'UltralyticsDetectorProvider','detector':'BboxDetectorSEGS','detailer':'DetailerForEach'}
    for role,kind in types.items():
        if by_id.get(marker.get(role),{}).get('type')!=kind:
            raise ValueError('手部分支标记与节点不一致：'+role)
    if marker.get('batch_version')==1:
        for role,kind in [('split','ImpactImageBatchToImageList'),('merge','ImageListToImageBatch')]:
            if by_id.get(marker.get(role),{}).get('type')!=kind:
                raise ValueError('手部批次分支标记与节点不一致：'+role)
        return raw,False
    detector=by_id[marker['detector']];detailer=by_id[marker['detailer']]
    def image_link(node):
        item=next((i for i in node.get('inputs',[]) if i.get('name')=='image'),None)
        link=next((l for l in links if item and l[0]==item.get('link')),None)
        if not link:raise ValueError('手部分支缺少图像输入连线。')
        return link
    detector_link=image_link(detector);detailer_link=image_link(detailer)
    if detector_link[1:3]!=detailer_link[1:3]:
        raise ValueError('手部检测与重绘必须使用同一图像来源。')
    source,slot=detector_link[1:3]
    output_links=[l for l in links if l[1]==detailer['id'] and l[2]==0]
    if not output_links:raise ValueError('手部重绘缺少图像输出连线。')
    split=max([int(n['id']) for n in nodes]+[int(raw.get('last_node_id',0))])+1;merge=split+1
    incoming=max([int(l[0]) for l in links]+[int(raw.get('last_link_id',0))])+1;outgoing=incoming+1
    active=detailer.get('mode',0)==0
    x,y=detailer.get('pos',[1500,850])
    def node(nid,kind,title,pos,input_name,input_link,outputs):
        return dict(id=nid,type=kind,title=title,pos=pos,size=[290,90],flags={},order=nid,mode=0 if active else 4,
                    inputs=[dict(name=input_name,type='IMAGE',link=input_link)],
                    outputs=[dict(name='IMAGE',type='IMAGE',links=outputs,slot_index=0)],
                    properties={'Node name for S&R':kind},widgets_values=[])
    nodes.extend([node(split,'ImpactImageBatchToImageList','手部处理 · 批次拆为逐张图像',[x-400,y+760],'image',incoming,[detector_link[0],detailer_link[0]]),
                  node(merge,'ImageListToImageBatch','手部处理 · 合并图像批次',[x+400,y+760],'images',outgoing,[l[0] for l in output_links])])
    original=by_id[source]['outputs'][slot]
    original['links']=[lid for lid in original.get('links',[]) if lid not in (detector_link[0],detailer_link[0])]+[incoming]
    detector_link[1:3]=[split,0];detailer_link[1:3]=[split,0]
    for link in output_links:link[1:3]=[merge,0]
    detailer['outputs'][0]['links']=[outgoing]
    split_input=0
    links.extend([[incoming,source,slot,split,split_input,'IMAGE'],[outgoing,detailer['id'],0,merge,0,'IMAGE']])
    marker.update(batch_version=1,split=split,merge=merge)
    raw['last_node_id']=merge;raw['last_link_id']=outgoing
    return raw,True

"""Install one pinned 52 MB face detector in Drive, expose it to current ComfyUI."""
from pathlib import Path
import hashlib,json,time,urllib.request
from tqdm.auto import tqdm

def install_face_detector():
    workspace=Path(globals().get('WORKSPACE','/content/ComfyUI'))
    persistent=Path(globals().get('PERSISTENT','/content/drive/MyDrive/ComfyUI_Persistent'))
    if not persistent.is_dir():raise RuntimeError('请先运行第 1 节挂载现有 Drive。')
    target=persistent/'models/ultralytics/bbox/face_yolov8m.pt';target.parent.mkdir(parents=True,exist_ok=True)
    expected='717923c19b3f4bbf5250b728f1fa6b2cb72a33aed1d236ea9caf0e21ad943e5f'
    def digest(path):
        value=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(4*1024**2):value.update(chunk)
        return value.hexdigest()
    if target.exists():
        if target.stat().st_size!=52026019 or digest(target)!=expected:raise RuntimeError('已有面部检测权重不匹配，已保留原文件。')
        print('面部检测模型已在 Drive 保存，跳过下载。',flush=True)
    else:
        url='https://huggingface.co/Bingsu/adetailer/resolve/53cc19de382014514d9d4038601d261a7faa9b7b/face_yolov8m.pt'
        temporary=target.with_suffix('.pt.part');value=hashlib.sha256();count=0
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Comfy-face-upgrade'}),timeout=90) as response,temporary.open('wb') as writer,tqdm(total=52026019,unit='B',unit_scale=True,desc='face_yolov8m.pt') as bar:
            while chunk:=response.read(1024**2):
                writer.write(chunk);value.update(chunk);count+=len(chunk);bar.update(len(chunk))
        if count!=52026019 or value.hexdigest()!=expected:raise RuntimeError('面部检测模型校验失败，未启用。重新运行可重试。')
        temporary.replace(target)
    local=workspace/'models/ultralytics/bbox/face_yolov8m.pt';local.parent.mkdir(parents=True,exist_ok=True)
    if local.exists() or local.is_symlink():
        if local.resolve()!=target.resolve() and (not local.is_file() or digest(local)!=expected):raise RuntimeError('本地同名检测权重冲突，已保留。')
    else:local.symlink_to(target)
    user=persistent/'user/default';user.mkdir(parents=True,exist_ok=True)
    # Do not broaden PyTorch loading permissions: the official YOLO model uses the existing Subpack loader.
    (user/'face_detailer_state.json').write_text(json.dumps(dict(version=1,installed=True,model='bbox/face_yolov8m.pt',sha256=expected,size=52026019,updated_at=time.time())),encoding='utf-8')
    print('面部检测权重已校验并连接；无需重启 ComfyUI。',flush=True)

install_face_detector()


import json,time
from pathlib import Path
user=Path(globals().get('PERSISTENT','/content/drive/MyDrive/ComfyUI_Persistent'))/'user/default'
folder=user/'workflows';backup=user.parent.parent/'backups'/('face-detailer-'+time.strftime('%Y%m%d-%H%M%S'))
count=0
for name in ('01_Hassaku_Illustrious.json','02_Anima_Turbo.json','03_Krea2_Anime.json','04_Lustify_Krea2.json'):
    path=folder/name
    if not path.is_file():continue
    original=path.read_text(encoding='utf-8');raw=json.loads(original)
    raw,_=ensure_hand_batch_nodes(raw);raw,changed=ensure_face_nodes(raw)
    if changed:
        backup.mkdir(parents=True,exist_ok=True);(backup/name).write_text(original,encoding='utf-8')
        temp=path.with_suffix('.face.tmp');temp.write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    if 'WORKFLOWS' in globals():WORKFLOWS[name]=raw
    print(name+'：面部 ADetailer 分支已就绪（默认关闭）。',flush=True);count+=1
if count!=4:raise RuntimeError('未找到全部四份现有工作流，请运行最新版第 5 节后重试。已完成的工作流保留。')
print('四份工作流已更新；网页刷新后在“处理”里开启面部修复。',flush=True)
