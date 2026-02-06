document.addEventListener('DOMContentLoaded', async () => {
  const backendUrlInput = document.getElementById('backendUrl');
  const saveButton = document.getElementById('saveSettings');
  const statusDiv = document.getElementById('status');

  // Charger les paramètres sauvegardés
  const result = await chrome.storage.sync.get(['backendUrl']);
  if (result.backendUrl) {
    backendUrlInput.value = result.backendUrl;
  } else {
    backendUrlInput.value = "https://fakefinder.vercel.app";
  }

  saveButton.addEventListener('click', async () => {
    const url = backendUrlInput.value.trim();
    if (!url) {
      showStatus('Veuillez entrer une URL', 'error');
      return;
    }

    await chrome.storage.sync.set({ backendUrl: url });
    showStatus('Paramètres enregistrés !', 'success');
  });
});

function showStatus(message, type) {
  const statusDiv = document.getElementById('status');
  statusDiv.textContent = message;
  statusDiv.className = `status ${type}`;
  setTimeout(() => {
    statusDiv.style.display = 'none';
  }, 3000);
}
