import { initModel, runInference } from "./inference.js";

document.addEventListener("DOMContentLoaded", async () => {
  const imageInput = document.getElementById("imageInput");
  const uploadArea = document.getElementById("uploadArea");
  const previewSection = document.getElementById("previewSection");
  const previewImage = document.getElementById("previewImage");
  const resultInfo = document.getElementById("resultInfo");
  const loading = document.getElementById("loading");
  const statusDiv = document.getElementById("status");
  const resetButton = document.getElementById("resetButton");

  try {
    await initModel();
  } catch (error) {
    showStatus(`Modèle: ${error.message}`, "error");
    console.error("Model init error:", error);
  }

  // Drag & drop
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith("image/")) {
      handleImageFile(files[0]);
    }
  });

  imageInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleImageFile(e.target.files[0]);
    }
  });

  async function handleImageFile(file) {
    if (!file.type.startsWith("image/")) {
      showStatus("Veuillez sélectionner une image", "error");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewSection.style.display = "block";
      uploadArea.style.display = "none";
      loading.style.display = "flex";
      resultInfo.innerHTML = "";
    };
    reader.readAsDataURL(file);

    await analyzeImage(file);
  }

  async function analyzeImage(file) {
    try {
      const result = await runInference(file);

      loading.style.display = "none";

      if (result.image) {
        previewImage.src = result.image;
      }

      const label = result.label === "fake" ? "FAKE" : "REAL";
      const color = result.label === "fake" ? "#f44336" : "#4CAF50";
      const confidence = (result.confidence * 100).toFixed(1);

      resultInfo.innerHTML = `
        <div class="result-label" style="color: ${color}; border-color: ${color}">
          ${label} ${confidence}%
        </div>
        <div class="result-details">
          <div>Real: ${(result.real_confidence * 100).toFixed(1)}%</div>
          <div>Fake: ${(result.fake_confidence * 100).toFixed(1)}%</div>
        </div>
      `;
      resultInfo.style.display = "block";
    } catch (error) {
      loading.style.display = "none";
      showStatus(`Erreur: ${error.message}`, "error");
      console.error("Erreur analyse:", error);
    }
  }

  function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = "block";
    setTimeout(() => {
      statusDiv.style.display = "none";
    }, 5000);
  }

  function reset() {
    previewSection.style.display = "none";
    uploadArea.style.display = "flex";
    previewImage.removeAttribute("src");
    resultInfo.innerHTML = "";
    resultInfo.style.display = "none";
    imageInput.value = "";
    loading.style.display = "none";
    statusDiv.style.display = "none";
  }

  if (resetButton) {
    resetButton.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      reset();
    });
  }

  uploadArea.addEventListener("click", () => {
    if (previewSection.style.display === "block") {
      reset();
    }
  });
});
