const form = document.getElementById('uploadForm');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const outVideo = document.getElementById('outVideo');
const jsonLink = document.getElementById('jsonLink');
const submitBtn = document.getElementById('submitBtn');

const modelPreset = document.getElementById('model_preset');
const customModelWrap = document.getElementById('customModelWrap');
const modelPathInput = document.getElementById('model_path');
const useGpuEl = document.getElementById('use_gpu');

function updateModelUi() {
  if (!modelPreset) return;
  const v = modelPreset.value;
  if (v === '__custom__') {
    customModelWrap.classList.remove('hidden');
  } else {
    customModelWrap.classList.add('hidden');
  }
}

if (modelPreset) {
  modelPreset.addEventListener('change', updateModelUi);
  updateModelUi();
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();

    const pct = Math.round((job.progress || 0) * 100);
    setStatus(`Job ${job.id}: ${job.state} (${pct}%)${job.error ? ` | ${job.error}` : ''}`);

    if (job.state === 'done') {
      const vUrl = `/api/jobs/${jobId}/result/video`;
      outVideo.src = vUrl;
      jsonLink.href = `/api/jobs/${jobId}/result/json`;
      resultEl.classList.remove('hidden');
      submitBtn.disabled = false;
      return;
    }

    if (job.state === 'error') {
      submitBtn.disabled = false;
      return;
    }

    await sleep(1000);
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById('video');
  if (!fileInput.files || !fileInput.files[0]) {
    setStatus('Please select a video.');
    return;
  }

  submitBtn.disabled = true;
  resultEl.classList.add('hidden');
  outVideo.removeAttribute('src');

  const fd = new FormData();
  fd.append('video', fileInput.files[0]);

  let modelPath = '';
  if (modelPreset && modelPreset.value === '__custom__') {
    modelPath = (modelPathInput && modelPathInput.value) ? modelPathInput.value : '';
  } else if (modelPreset) {
    modelPath = modelPreset.value;
  }
  fd.append('model_path', modelPath);

  const useGpu = useGpuEl ? !!useGpuEl.checked : false;
  fd.append('use_gpu', useGpu ? 'true' : 'false');

  fd.append('conf', document.getElementById('conf').value || '0.25');
  fd.append('iou', document.getElementById('iou').value || '0.45');

  setStatus('Uploading...');

  const res = await fetch('/api/jobs', {
    method: 'POST',
    body: fd,
  });

  if (!res.ok) {
    const text = await res.text();
    setStatus(`Upload failed: ${text}`);
    submitBtn.disabled = false;
    return;
  }

  const data = await res.json();
  setStatus(`Uploaded. Job ${data.job_id} queued...`);

  await pollJob(data.job_id);
});
