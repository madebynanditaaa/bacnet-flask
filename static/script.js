/*
  BacNet Frontend Script
  - Handles image preview, API request to /predict, dynamic DOM updates, and Chart.js rendering
  - Update colors and layout in style.css; backend base URL assumed to be relative '/predict'
*/

// Elements
const imageInput = document.getElementById('imageInput');
const classifyBtn = document.getElementById('classifyBtn');
const predictedClassEl = document.getElementById('predictedClass');
const confidenceEl = document.getElementById('confidence');
const errorBox = document.getElementById('errorBox');
const loadingEl = document.getElementById('loading');
const historyBody = document.getElementById('historyBody');
const previewImg = document.getElementById('previewImg');
const previewPlaceholder = document.getElementById('previewPlaceholder');

// In-memory history (frontend-only)
const history = [];

// Chart.js instance holder
let probChartInstance = null;

// Helper: get selected model value from radio inputs
function getSelectedModel() {
  const input = document.querySelector('input[name="model"]:checked');
  return input ? input.value : null;
}

// Image preview handler
imageInput.addEventListener('change', () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    clearPreview();
    return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block';
    previewPlaceholder.style.display = 'none';
  };
  reader.readAsDataURL(file);
});

// Clear preview (used if needed)
function clearPreview() {
  previewImg.src = '';
  previewImg.style.display = 'none';
  previewPlaceholder.style.display = 'grid';
}

// Button click: classify image
classifyBtn.addEventListener('click', async () => {
  hideError();

  const file = imageInput.files && imageInput.files[0];
  const model = getSelectedModel();

  if (!file) return showError('Please select an image to classify.');
  if (!model) return showError('Please choose a model.');

  setLoading(true);
  setButtonLoading(true);
  try {
    const form = new FormData();
    // The backend should expect 'file' and 'model' fields
    form.append('file', file);
    form.append('model', model);

    const res = await fetch('/predict', {
      method: 'POST',
      body: form
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Request failed (${res.status}): ${text}`);
    }

    /**
     Expected response structure:
     {
       predicted_class: string,
       confidence: number,
       probabilities: Record<string, number>,
       history: Array<{ model: string, class: string, confidence: number }>
     }
    */
    const data = await res.json();
    updateResults(data);
  } catch (err) {
    console.error(err);
    showError('Classification failed. Please try again or check the server logs.');
  } finally {
    setLoading(false);
    setButtonLoading(false);
  }
});

// Update Results UI
function updateResults(payload) {
  const { predicted_class, confidence, probabilities, history: serverHistory } = payload || {};

  // Predicted class & confidence
  predictedClassEl.textContent = predicted_class ?? '—';
  confidenceEl.textContent = isFinite(confidence) ? `${Number(confidence).toFixed(2)}%` : '—';
  predictedClassEl.classList.remove('muted');
  confidenceEl.classList.remove('muted');

  // Render chart
  if (probabilities && typeof probabilities === 'object') {
    renderChart(probabilities);
  }

  // Append to history (frontend) using the latest prediction
  const modelUsed = getSelectedModel();
  if (predicted_class && isFinite(confidence)) {
    history.push({ model: modelUsed, class: predicted_class, confidence: Number(confidence) });
    appendHistoryRow({ model: modelUsed, class: predicted_class, confidence: Number(confidence) });
  }

  // Optionally use server-provided history if needed:
  // if (Array.isArray(serverHistory)) { /* could sync/compare */ }
}

// Append a row to the history table
function appendHistoryRow(entry) {
  const tr = document.createElement('tr');
  const tdModel = document.createElement('td');
  const tdClass = document.createElement('td');
  const tdConf = document.createElement('td');

  tdModel.textContent = entry.model;
  tdClass.textContent = entry.class;
  tdConf.textContent = `${entry.confidence.toFixed(2)}`;

  tr.appendChild(tdModel);
  tr.appendChild(tdClass);
  tr.appendChild(tdConf);
  historyBody.prepend(tr); // newest on top
}

// Error helpers
function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
function hideError() { errorBox.hidden = true; errorBox.textContent = ''; }

// Loading state helpers
function setLoading(isLoading) {
  loadingEl.hidden = !isLoading;
}
function setButtonLoading(isLoading) {
  classifyBtn.disabled = isLoading;
  classifyBtn.textContent = isLoading ? 'Classifying…' : 'Classify';
}

// Chart.js renderer
function renderChart(probabilities) {
  const ctx = document.getElementById('probChart');
  if (!ctx) return;

  // Destroy previous chart if any
  if (probChartInstance) {
    probChartInstance.destroy();
    probChartInstance = null;
  }

  // Prepare labels and values from probabilities object
  const labels = Object.keys(probabilities);
  const values = labels.map(k => Number(probabilities[k]));

  // Sort by descending probability for better readability
  const combined = labels.map((l, i) => ({ label: l, value: values[i] }));
  combined.sort((a, b) => b.value - a.value);

  const sortedLabels = combined.map(item => item.label);
  const sortedValues = combined.map(item => item.value);

  probChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sortedLabels,
      datasets: [
        {
          label: 'Probability (%)',
          data: sortedValues,
          backgroundColor: 'rgba(16, 185, 129, 0.25)',
          borderColor: 'rgba(16, 185, 129, 1)',
          borderWidth: 2,
          borderRadius: 8,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: context => `${context.raw.toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#6b7280' },
          grid: { display: false }
        },
        y: {
          beginAtZero: true,
          ticks: { color: '#6b7280', callback: value => `${value}%` },
          grid: { color: 'rgba(229, 231, 235, 0.7)' }
        }
      },
      animation: {
        duration: 600,
        easing: 'easeOutQuart'
      }
    }
  });
}


