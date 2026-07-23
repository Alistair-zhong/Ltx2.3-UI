const $ = (id) => document.getElementById(id);
const n = (id) => Number($(id).value);
const state = { currentJob: null, poll: null, uploadTarget: null, activeModelKey: null };
const JOB_POLL_INTERVAL_MS = 5000;
const HEALTH_POLL_INTERVAL_MS = 10000;

function toast(message) {
  const el = $('toast'); el.textContent = message; el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2600);
}
async function api(url, options = {}) {
  let response;
  try {
    response = await fetch(url, { headers: options.body instanceof FormData ? {} : {'Content-Type':'application/json'}, ...options });
  } catch (error) {
    throw new Error(`无法连接服务端 ${location.origin}${url}。请确认页面通过 ltx23-ui 的 7860 端口打开，且服务进程仍在运行。`);
  }
  if (!response.ok) { let data; try { data = await response.json(); } catch { data = {}; } const detail=Array.isArray(data.detail)?data.detail.map(item=>`${(item.loc||[]).slice(1).join('.')}: ${item.msg}`).join('；'):data.detail;throw new Error(detail || `请求失败 ${response.status}`); }
  return response.json();
}
function modelData() {
  return {
    checkpoint_path:$('checkpoint').value.trim(), gemma_root:$('gemma').value.trim(),
    distilled_lora:{path:$('distilledPath').value.trim(), strength:n('distilledStrength')},
    spatial_upsampler_path:$('upsampler').value.trim(),
    loras:[...document.querySelectorAll('.user-lora')].map(row => ({path:row.querySelector('.lora-path').value.trim(), strength:Number(row.querySelector('.lora-strength').value)})).filter(x=>x.path),
    quantization:$('quantization').value, offload:$('offload').value,
    compile_mode:$('compileMode').value, max_batch_size:n('maxBatch')
  };
}
function requestData() {
  return {model:modelData(), generation:{
    prompt:$('prompt').value.trim(), negative_prompt:$('negativePrompt').value.trim(),
    audio_path:$('audioPath').value.trim(), audio_start_time:n('audioStart'),
    audio_max_duration:$('audioDuration').value ? n('audioDuration') : null,
    images:[...document.querySelectorAll('.condition-row')].map(row => ({path:row.querySelector('.image-path').value.trim(), frame_idx:Number(row.querySelector('.image-frame').value), strength:Number(row.querySelector('.image-strength').value), crf:33})).filter(x=>x.path),
    height:n('height'), width:n('width'), num_frames:n('numFrames'), frame_rate:n('fps'),
    num_inference_steps:n('steps'), seed:n('seed'), output_path:$('outputPath').value.trim(),
    enhance_prompt:$('enhancePrompt').checked, profile:$('profile').checked,
    guidance:{cfg_scale:n('cfg'),stg_scale:n('stg'),rescale_scale:n('rescale'),a2v_scale:n('a2v'),skip_step:n('skipStep'),stg_blocks:$('stgBlocks').value.split(',').map(x=>Number(x.trim())).filter(Number.isFinite)}
  }};
}
function addImage(data={path:'',frame_idx:0,strength:1}) {
  const row=document.createElement('div'); row.className='condition-row';
  row.innerHTML=`<label class="field"><span>图片路径</span><div class="file-row"><input class="image-path" value="${escapeHtml(data.path)}" placeholder="图片路径"><button type="button" class="upload row-upload">上传</button></div></label><label class="field"><span>目标帧</span><input class="image-frame" type="number" min="0" value="${data.frame_idx}"></label><label class="field"><span>强度</span><input class="image-strength" type="number" min="0" max="1" step="0.05" value="${data.strength}"></label><button type="button" class="icon-button">×</button>`;
  row.querySelector('.icon-button').onclick=()=>{row.remove();scheduleValidate()}; row.querySelector('.row-upload').onclick=()=>openUpload(row.querySelector('.image-path'));
  row.querySelectorAll('input').forEach(x=>x.addEventListener('input',scheduleValidate)); $('imageList').appendChild(row);
}
function addLora(data={path:'/home/us5090/workspace/niro-workspace/LTX-2/models/lora_weights_step_01750.safetensors',strength:.8}) {
  const row=document.createElement('div'); row.className='lora-row user-lora';
  row.innerHTML=`<label class="field"><span>LoRA 路径</span><input class="lora-path" value="${escapeHtml(data.path)}"></label><label class="field"><span>强度</span><input class="lora-strength" type="number" min="-4" max="4" step="0.05" value="${data.strength}"></label><button type="button" class="icon-button">×</button>`;
  row.querySelector('.icon-button').onclick=()=>{row.remove();scheduleValidate()}; row.querySelectorAll('input').forEach(x=>x.addEventListener('input',scheduleValidate)); $('loraList').appendChild(row);
}
function escapeHtml(value){const d=document.createElement('div');d.textContent=value;return d.innerHTML.replaceAll('"','&quot;')}
function updateSummary() {
  const frames=n('numFrames'), fps=n('fps')||1, duration=frames/fps;
  $('frameSummary').textContent=`${frames} 帧`; $('durationSummary').textContent=`${duration.toFixed(2)} 秒 · ${(frames-1)%8===0?'满足 8k+1':'帧数无效'}`;
  $('specResolution').textContent=`${n('width')} × ${n('height')}`; $('specDuration').textContent=`${duration.toFixed(2)}s`;
  $('specFrames').textContent=`${frames} @ ${fps}`; $('specSeed').textContent=$('seed').value;
}
async function autoFrames() {
  if (!$('autoFrames').checked || !n('audioDuration') || !n('fps')) return;
  try { const data=await api('/api/frames',{method:'POST',body:JSON.stringify({duration:n('audioDuration'),fps:n('fps')})}); $('numFrames').value=data.num_frames; updateSummary(); scheduleValidate(); } catch(e) { toast(e.message); }
}
async function probeAudio() {
  if (!$('audioPath').value.trim()) return toast('请先填写或上传音频');
  try { const data=await api('/api/probe',{method:'POST',body:JSON.stringify({path:$('audioPath').value.trim(),fps:n('fps'),start_time:n('audioStart'),max_duration:$('audioDuration').value?n('audioDuration'):null})});
    $('audioMeta').textContent=`文件 ${data.source_duration}s · 使用 ${data.selected_duration}s · 推荐 ${data.num_frames} 帧`;
    if($('autoFrames').checked){$('numFrames').value=data.num_frames;updateSummary()} scheduleValidate();
  } catch(e){toast(e.message)}
}
let validationTimer;
function scheduleValidate(){updateSummary();clearTimeout(validationTimer);validationTimer=setTimeout(()=>validate(false),450)}
async function validate(notify=true){
  try{const data=await api('/api/validate',{method:'POST',body:JSON.stringify(requestData())});renderValidation(data);if(notify)toast(data.valid?'参数检查通过':'请修正参数错误');return data}
  catch(e){$('validationPill').textContent='参数不完整';$('validationPill').className='pill bad';$('issues').innerHTML=`<div class="issue error">${escapeHtml(e.message)}</div>`;if(notify)toast(e.message);return null}
}
function renderValidation(data){
  const errors=data.issues.filter(x=>x.level==='error').length,warns=data.issues.filter(x=>x.level==='warning').length;
  $('validationPill').textContent=errors?'检查失败':warns?'有提示':'参数有效';$('validationPill').className=`pill ${errors?'bad':warns?'warn':'good'}`;
  $('issues').innerHTML=data.issues.slice(0,4).map(x=>`<div class="issue ${x.level}">${escapeHtml(x.message)}</div>`).join('');
  $('runButton').disabled=!data.valid;$('runHint').textContent=data.requires_reload?'将加载新模型配置':'将复用已加载模型';
  $('reloadBadge').textContent=data.requires_reload?'下一任务重载':'模型可复用';$('reloadBadge').style.color=data.requires_reload?'var(--warm)':'var(--accent)';
}
async function submit(event){event.preventDefault();const result=await validate(false);if(!result?.valid)return toast('参数未通过检查');
  try{const job=await api('/api/jobs',{method:'POST',body:JSON.stringify(requestData())});state.currentJob=job.id;showProgress(job);toast(`任务 ${job.id} 已加入队列`);startPolling();loadJobs()}
  catch(e){toast(e.message)}
}
function renderProfile(profile){
  if(!profile){$('profileBox').classList.add('hidden');return}
  const phases=(profile.phases||[]).slice(0,4);
  const recommendations=(profile.recommendations||[]).slice(0,2);
  $('profileBox').classList.remove('hidden');
  $('profileBox').innerHTML=`<div class="profile-head"><b>性能报告 · ${profile.total_seconds??'失败'}${profile.total_seconds?'s':''}</b><a href="/api/jobs/${profile.job_id}/profile" target="_blank">JSON</a></div>${phases.map(x=>`<div class="profile-row"><span>${escapeHtml(x.label)}</span><b>${x.seconds}s · ${x.percent}%</b></div>`).join('')}${recommendations.map(x=>`<div class="profile-advice">建议 · ${escapeHtml(x)}</div>`).join('')}<small>${profile.error?escapeHtml(profile.error):`${profile.cold_start?'首次加载/编译冷运行':'模型复用热运行'} · compile ${escapeHtml(profile.compile_mode)}`}</small>`;
}
function showProgress(job){$('progressBox').classList.remove('hidden');$('progressMessage').textContent=job.message;$('progressValue').textContent=`${job.progress}%`;$('progressBar').style.width=`${job.progress}%`;renderProfile(job.profile)}
function startPolling(){clearInterval(state.poll);state.poll=setInterval(async()=>{if(!state.currentJob)return;try{const job=await api(`/api/jobs/${state.currentJob}`);showProgress(job);if(['completed','failed','cancelled'].includes(job.state)){clearInterval(state.poll);loadJobs();health();if(job.state==='completed'){$('preview').innerHTML=`<video controls autoplay src="/api/jobs/${job.id}/video?t=${Date.now()}"></video>`;toast('视频生成完成')}else toast(job.error?.split('\n')[0]||job.message)}}catch{}},JOB_POLL_INTERVAL_MS)}
async function loadJobs(){try{const jobs=await api('/api/jobs');$('jobList').innerHTML=jobs.length?jobs.map(job=>`<div class="job-card"><div class="top"><b>#${job.id}</b><span class="state">${job.state} · ${job.progress}%</span></div><p>${escapeHtml(job.prompt)}</p><small>${new Date(job.created_at).toLocaleString()} · Seed ${job.seed}${job.profile?.total_seconds?` · ${job.profile.total_seconds}s`:''}</small><div class="job-actions">${job.state==='completed'?`<a href="/api/jobs/${job.id}/video" target="_blank">播放 / 下载</a>`:''}${job.profile?`<a href="/api/jobs/${job.id}/profile" target="_blank">性能报告</a>`:''}${job.state==='queued'?`<button onclick="cancelJob('${job.id}')">取消</button>`:''}</div></div>`).join(''):'<div class="empty">还没有生成任务</div>'}catch(e){toast(e.message)}}
async function cancelJob(id){try{await api(`/api/jobs/${id}/cancel`,{method:'POST'});loadJobs()}catch(e){toast(e.message)}}window.cancelJob=cancelJob;
async function health(){try{const data=await api('/api/health');state.health=data;$('statusDot').className='online';const base=data.model_loaded?'模型已加载 · 可复用':data.upload_ready?'服务在线 · 模型未加载':'服务在线 · 上传目录不可写';$('modelStatus').textContent=`${base} · v${data.version||'unknown'}`}catch{$('statusDot').className='';$('modelStatus').textContent='服务离线'}}
async function loadDefaults(){try{const data=await api('/api/defaults');if(!$('negativePrompt').value)$('negativePrompt').value=data.negative_prompt}catch{}}
function openUpload(input){state.uploadTarget=input;$('hiddenUpload').accept=input.id==='audioPath'?'audio/*':'image/*';$('hiddenUpload').click()}
function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', '/api/upload');
    request.responseType = 'json';
    request.upload.onprogress = event => {
      if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100));
    };
    request.onload = () => {
      const data = request.response || {};
      if (request.status >= 200 && request.status < 300) resolve(data);
      else reject(new Error(data.detail || `上传失败：HTTP ${request.status}`));
    };
    request.onerror = () => reject(new Error(`上传连接被中断。请检查 ${location.origin}/api/health 是否能打开，并查看 ltx23-ui 终端日志。`));
    request.onabort = () => reject(new Error('上传已取消'));
    const body = new FormData(); body.append('file', file); request.send(body);
  });
}
async function handleUpload(){const file=$('hiddenUpload').files[0];if(!file||!state.uploadTarget)return;try{if(state.health&&!state.health.upload_ready)throw new Error(`服务端上传目录不可写：${state.health.upload_dir}`);const max=state.health?.max_upload_bytes;if(max&&file.size>max)throw new Error(`文件过大，最大允许 ${Math.round(max/1024/1024)} MB`);toast(`正在上传 ${file.name}…`);const data=await uploadFile(file,percent=>toast(`正在上传 ${file.name} · ${percent}%`));state.uploadTarget.value=data.path;state.uploadTarget.dispatchEvent(new Event('input'));toast(`已上传 ${data.name}`)}catch(e){toast(e.message)}finally{$('hiddenUpload').value=''}}
function savePreset(){const name=prompt('预设名称');if(!name)return;const presets=JSON.parse(localStorage.getItem('ltx-presets')||'{}');presets[name]=requestData();localStorage.setItem('ltx-presets',JSON.stringify(presets));refreshPresets();toast('预设已保存')}
function refreshPresets(){const presets=JSON.parse(localStorage.getItem('ltx-presets')||'{}');$('presetSelect').innerHTML='<option value="">配置预设</option>'+Object.keys(presets).map(x=>`<option>${escapeHtml(x)}</option>`).join('')}
function loadPreset(name){const p=JSON.parse(localStorage.getItem('ltx-presets')||'{}')[name];if(!p)return;const m=p.model,g=p.generation;const map={checkpoint:m.checkpoint_path,gemma:m.gemma_root,upsampler:m.spatial_upsampler_path,distilledPath:m.distilled_lora.path,distilledStrength:m.distilled_lora.strength,quantization:m.quantization,offload:m.offload,compileMode:m.compile_mode??'reduce-overhead',maxBatch:m.max_batch_size,prompt:g.prompt,negativePrompt:g.negative_prompt,audioPath:g.audio_path,audioStart:g.audio_start_time,audioDuration:g.audio_max_duration,height:g.height,width:g.width,numFrames:g.num_frames,fps:g.frame_rate,steps:g.num_inference_steps,seed:g.seed,outputPath:g.output_path,cfg:g.guidance.cfg_scale,stg:g.guidance.stg_scale,rescale:g.guidance.rescale_scale,a2v:g.guidance.a2v_scale,skipStep:g.guidance.skip_step,stgBlocks:g.guidance.stg_blocks.join(',')};Object.entries(map).forEach(([id,v])=>{if($(id))$(id).value=v??''});$('enhancePrompt').checked=g.enhance_prompt;$('profile').checked=g.profile??true;$('loraList').innerHTML='';m.loras.forEach(addLora);$('imageList').innerHTML='';g.images.forEach(addImage);scheduleValidate();toast(`已加载预设 ${name}`)}
document.querySelectorAll('.nav-item').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('.nav-item,.tab-page').forEach(x=>x.classList.remove('active'));btn.classList.add('active');$(`tab-${btn.dataset.tab}`).classList.add('active');if(btn.dataset.tab==='queue')loadJobs()});
$('generationForm').addEventListener('submit',submit);$('addImage').onclick=()=>addImage();$('addLora').onclick=()=>addLora();$('probeAudio').onclick=probeAudio;$('validateButton').onclick=()=>validate(true);$('refreshJobs').onclick=loadJobs;$('savePreset').onclick=savePreset;$('presetSelect').onchange=e=>loadPreset(e.target.value);$('hiddenUpload').onchange=handleUpload;document.querySelectorAll('[data-upload-for]').forEach(b=>b.onclick=()=>openUpload($(b.dataset.uploadFor)));$('unloadModel').onclick=async()=>{try{await api('/api/model/unload',{method:'POST'});health();scheduleValidate();toast('模型已从内存卸载')}catch(e){toast(e.message)}};
['prompt','audioPath','audioStart','audioDuration','fps','numFrames','width','height','steps','seed','outputPath','checkpoint','gemma','upsampler','distilledPath','distilledStrength','quantization','offload','compileMode','maxBatch','cfg','stg','a2v','rescale','skipStep','stgBlocks','negativePrompt'].forEach(id=>$(id).addEventListener('input',()=>{if(id==='prompt')$('promptCount').textContent=$('prompt').value.length;if(['audioDuration','fps'].includes(id))autoFrames();else scheduleValidate()}));
addImage({path:'/home/us5090/workspace/niro-workspace/LTX-2/test/1.jpg',frame_idx:0,strength:1});addLora();refreshPresets();updateSummary();loadDefaults().then(()=>validate(false));health();setInterval(health,HEALTH_POLL_INTERVAL_MS);
