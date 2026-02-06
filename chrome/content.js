let isAnalyzing = false;

async function getBackendUrl() {
  const result = await chrome.storage.sync.get(['backendUrl']);
  return result.backendUrl || 'https://fakefinder.vercel.app';
}

function createOverlay(image, result) {
  const overlay = document.createElement('div');
  overlay.className = 'fakefinder-overlay';
  
  const label = result.label === 'fake' ? 'FAKE' : 'REAL';
  const color = result.label === 'fake' ? '#f44336' : '#4CAF50';
  const confidence = (result.confidence * 100).toFixed(1);
  
  overlay.innerHTML = `
    <div class="fakefinder-result" style="border-color: ${color}">
      <div class="fakefinder-label" style="color: ${color}">
        ${label} ${confidence}%
      </div>
      <div class="fakefinder-details">
        <div>Real: ${(result.real_confidence * 100).toFixed(1)}%</div>
        <div>Fake: ${(result.fake_confidence * 100).toFixed(1)}%</div>
      </div>
      <button class="fakefinder-close">×</button>
    </div>
  `;
  
  const rect = image.getBoundingClientRect();
  overlay.style.top = `${rect.top + window.scrollY}px`;
  overlay.style.left = `${rect.left + window.scrollX}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  
  document.body.appendChild(overlay);
  
  overlay.querySelector('.fakefinder-close').addEventListener('click', () => {
    overlay.remove();
  });
  
  setTimeout(() => {
    overlay.classList.add('show');
  }, 10);
  
  return overlay;
}

async function analyzeImage(imageUrl) {
  if (isAnalyzing) return;
  isAnalyzing = true;
  
  try {
    const backendUrl = await getBackendUrl();
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    
    const formData = new FormData();
    formData.append('file', blob, 'image.jpg');
    
    const apiResponse = await fetch(`${backendUrl}/api/inference`, {
      method: 'POST',
      body: formData
    });
    
    if (!apiResponse.ok) {
      throw new Error(`Erreur API: ${apiResponse.status}`);
    }
    
    const result = await apiResponse.json();
    return result;
  } catch (error) {
    console.error('Erreur analyse:', error);
    throw error;
  } finally {
    isAnalyzing = false;
  }
}

function showLoading(image) {
  const loading = document.createElement('div');
  loading.className = 'fakefinder-loading';
  const rect = image.getBoundingClientRect();
  loading.style.top = `${rect.top + window.scrollY + rect.height / 2}px`;
  loading.style.left = `${rect.left + window.scrollX + rect.width / 2}px`;
  loading.textContent = 'Analyse en cours...';
  document.body.appendChild(loading);
  return loading;
}

document.addEventListener('click', async (e) => {
  if (isAnalyzing) return;
  
  let img = e.target;
  if (img.tagName !== 'IMG') {
    img = img.closest('img');
  }
  
  if (!img || img.classList.contains('fakefinder-processed')) return;
  
  e.preventDefault();
  e.stopPropagation();
  
  const imageUrl = img.src || img.dataset.src || img.currentSrc;
  if (!imageUrl || imageUrl.startsWith('data:')) {
    alert('Impossible d\'analyser cette image');
    return;
  }
  
  img.classList.add('fakefinder-processed');
  const loading = showLoading(img);
  
  try {
    const result = await analyzeImage(imageUrl);
    
    if (result.image) {
      const resultImg = new Image();
      resultImg.src = result.image;
      resultImg.onload = () => {
        img.src = result.image;
        loading.remove();
        createOverlay(img, result);
      };
    } else {
      loading.remove();
      createOverlay(img, result);
    }
  } catch (error) {
    loading.remove();
    alert(`Erreur: ${error.message}`);
    img.classList.remove('fakefinder-processed');
  }
}, true);
