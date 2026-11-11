/**
 * HarvestIQ – Frontend Application Controller
 * Handles backend connectivity, real-time agro-climatic validation,
 * quick presets, form dispatch, and live machine learning recommendation rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM References
  const form = document.getElementById("crop-form");
  const submitBtn = document.getElementById("submit-btn");
  const resetBtn = document.getElementById("reset-btn");
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  const errorContainer = document.getElementById("error-container");
  const errorList = document.getElementById("error-list");
  const resultContainer = document.getElementById("result-container");
  const presetsContainer = document.getElementById("presets-container");
  const backendNotice = document.getElementById("backend-notice");
  const backendNoticeText = document.getElementById("backend-notice-text");
  const docsLink = document.querySelector(".docs-link");

  // Keep docs link synchronized with resolved API base URL
  if (docsLink) {
    docsLink.href = `${CONFIG.API_BASE_URL}/docs`;
  }

  // Track active inputs
  const inputs = {
    nitrogen: document.getElementById("input-nitrogen"),
    phosphorus: document.getElementById("input-phosphorus"),
    potassium: document.getElementById("input-potassium"),
    temperature: document.getElementById("input-temperature"),
    humidity: document.getElementById("input-humidity"),
    ph: document.getElementById("input-ph"),
    rainfall: document.getElementById("input-rainfall"),
  };

  const sliders = {
    nitrogen: document.getElementById("slider-nitrogen"),
    phosphorus: document.getElementById("slider-phosphorus"),
    potassium: document.getElementById("slider-potassium"),
    temperature: document.getElementById("slider-temperature"),
    humidity: document.getElementById("slider-humidity"),
    ph: document.getElementById("slider-ph"),
    rainfall: document.getElementById("slider-rainfall"),
  };

  // Cache last submitted payload for retry functionality
  let lastSubmittedPayload = null;

  // -------------------------------------------------------------------------
  // 1. Health Monitoring & Live Backend Check
  // -------------------------------------------------------------------------
  async function checkBackendHealth() {
    try {
      const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.HEALTH}`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      statusPill.classList.add("online");

      const modelLabel = data.model_name ? ` (${data.model_name})` : "";
      statusText.textContent = `Backend: Online${modelLabel}`;

      if (!data.model_loaded) {
        backendNotice.style.display = "flex";
        backendNoticeText.innerHTML =
          "Backend is online, but the trained model artifact is not loaded. Live predictions are suspended until the model is available.";
      } else {
        backendNotice.style.display = "none";
      }
    } catch (err) {
      statusPill.classList.remove("online");
      statusText.textContent = "Backend: Offline";
      backendNotice.style.display = "flex";
      backendNoticeText.innerHTML = `
        Cannot connect to FastAPI at <code>${CONFIG.API_BASE_URL}</code>.<br>
        If running locally, start the server: <code>python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload</code><br>
        If running on Render / production, set your API URL:
        <div style="margin-top: 8px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
          <input type="text" id="api-url-input" value="${CONFIG.API_BASE_URL}" placeholder="https://harvestiq-api.onrender.com" style="flex: 1; min-width: 220px; padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3); color: #fff; font-size: 13px;" />
          <button type="button" class="btn-retry" id="save-api-url-btn" style="margin: 0; padding: 4px 10px;">Set URL</button>
          <button type="button" class="btn-retry" id="retry-health-btn" style="margin: 0; padding: 4px 10px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16"/>
            </svg>
            Retry
          </button>
        </div>
      `;

      const retryBtn = document.getElementById("retry-health-btn");
      if (retryBtn) {
        retryBtn.addEventListener("click", () => {
          statusText.textContent = "Backend: Reconnecting...";
          checkBackendHealth();
        });
      }

      const saveBtn = document.getElementById("save-api-url-btn");
      if (saveBtn) {
        saveBtn.addEventListener("click", () => {
          const input = document.getElementById("api-url-input");
          if (input && input.value.trim()) {
            CONFIG.setApiBaseUrl(input.value.trim());
          }
        });
      }
    }
  }

  // Initial check + poll every 10 seconds
  checkBackendHealth();
  setInterval(checkBackendHealth, 10000);

  // Allow clicking on status pill to manually refresh connection
  statusPill.style.cursor = "pointer";
  statusPill.title = "Click to refresh backend connection status";
  statusPill.addEventListener("click", () => {
    statusText.textContent = "Checking...";
    checkBackendHealth();
  });

  // -------------------------------------------------------------------------
  // 2. Synchronize Sliders with Numeric Inputs
  // -------------------------------------------------------------------------
  Object.keys(inputs).forEach((key) => {
    const input = inputs[key];
    const slider = sliders[key];
    if (!input || !slider) return;

    slider.addEventListener("input", (e) => {
      input.value = e.target.value;
      validateSingleField(key);
    });

    input.addEventListener("input", (e) => {
      slider.value = e.target.value;
      validateSingleField(key);
    });
  });

  function validateSingleField(key) {
    const input = inputs[key];
    const bounds = CONFIG.PARAMETERS[key];
    if (!input || !bounds) return true;

    const val = parseFloat(input.value);
    const isInvalid = isNaN(val) || val < bounds.min || val > bounds.max;

    if (isInvalid) {
      input.classList.add("invalid");
    } else {
      input.classList.remove("invalid");
    }
    return !isInvalid;
  }

  // -------------------------------------------------------------------------
  // 3. Quick Preset Buttons
  // -------------------------------------------------------------------------
  function renderPresets() {
    if (!presetsContainer || !CONFIG.PRESETS) return;
    presetsContainer.innerHTML = "";

    CONFIG.PRESETS.forEach((preset) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-btn";
      btn.textContent = preset.name;
      btn.title = preset.desc;

      btn.addEventListener("click", () => {
        document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        applyPreset(preset.values);
      });

      presetsContainer.appendChild(btn);
    });
  }

  function applyPreset(values) {
    Object.keys(values).forEach((key) => {
      if (inputs[key] && sliders[key]) {
        inputs[key].value = values[key];
        sliders[key].value = values[key];
        inputs[key].classList.remove("invalid");
      }
    });
    hideErrors();
  }

  renderPresets();

  // -------------------------------------------------------------------------
  // 4. Form Submission & Real ML API Dispatch
  // -------------------------------------------------------------------------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideErrors();

    // Collect and convert all 7 values into numerical types
    const payload = {
      nitrogen: parseFloat(inputs.nitrogen.value),
      phosphorus: parseFloat(inputs.phosphorus.value),
      potassium: parseFloat(inputs.potassium.value),
      temperature: parseFloat(inputs.temperature.value),
      humidity: parseFloat(inputs.humidity.value),
      ph: parseFloat(inputs.ph.value),
      rainfall: parseFloat(inputs.rainfall.value),
    };

    lastSubmittedPayload = payload;

    // Client-side pre-validation
    let hasClientErrors = false;
    const clientErrorMsgs = [];

    Object.keys(payload).forEach((key) => {
      const bounds = CONFIG.PARAMETERS[key];
      const val = payload[key];
      if (isNaN(val)) {
        hasClientErrors = true;
        inputs[key].classList.add("invalid");
        clientErrorMsgs.push(`${bounds.label}: value is required and must be a valid number.`);
      } else if (val < bounds.min || val > bounds.max) {
        hasClientErrors = true;
        inputs[key].classList.add("invalid");
        clientErrorMsgs.push(
          `${bounds.label}: ${val} ${bounds.unit} is outside valid physiological range (${bounds.min} – ${bounds.max} ${bounds.unit})`
        );
      }
    });

    if (hasClientErrors) {
      showErrors(clientErrorMsgs);
      return;
    }

    await dispatchPrediction(payload);
  });

  async function dispatchPrediction(payload) {
    // Set loading state in UI
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
      <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
        <path d="M12 2a10 10 0 0 1 10 10" />
      </svg>
      Evaluating Parameters...
    `;

    resultContainer.className = "recommendation-result";
    resultContainer.innerHTML = `
      <div class="loading-card">
        <div class="loading-spinner-hero"></div>
        <h4 style="color: #fff; margin-bottom: 8px;">Running Machine Learning Inference</h4>
        <p style="color: var(--text-muted); font-size: 0.88rem;">
          Passing 7 agro-climatic parameters into the trained Random Forest model pipeline...
        </p>
      </div>
    `;

    try {
      const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.RECOMMEND}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.status === 422) {
        // HTTP 422 Input Validation Error
        handleValidationErrors(data);
        restoreEmptyPlaceholder();
      } else if (response.status === 503) {
        // Model unavailable
        showErrors([
          data.message || "Model artifact not available. Please ensure model training has been completed.",
        ]);
        restoreEmptyPlaceholder();
      } else if (!response.ok || !data.success) {
        // General server error
        showErrors([data.message || `Server responded with HTTP ${response.status}`]);
        restoreEmptyPlaceholder();
      } else {
        // Real ML Prediction Success
        renderRecommendationResult(data);
      }
    } catch (err) {
      showConnectionError();
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
        </svg>
        Generate Crop Recommendation
      `;
    }
  }

  // -------------------------------------------------------------------------
  // 5. Error Handlers & Connection State
  // -------------------------------------------------------------------------
  function handleValidationErrors(data) {
    const messages = [];

    if (data.errors && Array.isArray(data.errors)) {
      data.errors.forEach((err) => {
        messages.push(err.message || `${err.field}: invalid value`);
        if (inputs[err.field]) {
          inputs[err.field].classList.add("invalid");
        }
      });
    } else if (data.detail && Array.isArray(data.detail)) {
      data.detail.forEach((err) => {
        const fieldName = err.loc ? err.loc[err.loc.length - 1] : "field";
        messages.push(`${fieldName}: ${err.msg}`);
        if (inputs[fieldName]) {
          inputs[fieldName].classList.add("invalid");
        }
      });
    } else {
      messages.push(data.detail || data.message || "Validation error occurred.");
    }

    showErrors(messages);
  }

  function showErrors(messages) {
    errorList.innerHTML = "";
    messages.forEach((msg) => {
      const li = document.createElement("li");
      li.textContent = msg;
      errorList.appendChild(li);
    });
    errorContainer.style.display = "flex";
  }

  function showConnectionError() {
    errorList.innerHTML = "";
    const li = document.createElement("li");
    li.innerHTML = `
      <strong>Backend Unreachable:</strong> Could not connect to the HarvestIQ API at <code>${CONFIG.API_BASE_URL}</code>.<br>
      Please start your backend server using: <code>python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload</code>
      <br>
      <button type="button" class="btn-retry" id="retry-prediction-btn">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16"/>
        </svg>
        Retry Prediction
      </button>
    `;
    errorList.appendChild(li);
    errorContainer.style.display = "flex";

    const retryBtn = document.getElementById("retry-prediction-btn");
    if (retryBtn && lastSubmittedPayload) {
      retryBtn.addEventListener("click", () => {
        hideErrors();
        dispatchPrediction(lastSubmittedPayload);
      });
    }

    restoreEmptyPlaceholder();
  }

  function hideErrors() {
    errorContainer.style.display = "none";
    errorList.innerHTML = "";
    Object.values(inputs).forEach((inp) => inp.classList.remove("invalid"));
  }

  function restoreEmptyPlaceholder() {
    resultContainer.className = "recommendation-result";
    resultContainer.innerHTML = `
      <div class="result-placeholder">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="M12 8v4M12 16h.01"/>
        </svg>
        <p>Awaiting agro-climatic parameters.<br>Adjust the soil nutrients and climate factors on the left and click <strong>Generate Crop Recommendation</strong>.</p>
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // 6. Real ML Prediction Result Renderer
  // -------------------------------------------------------------------------
  function renderRecommendationResult(data) {
    hideErrors();

    if (!data.recommended_crop) {
      restoreEmptyPlaceholder();
      return;
    }

    const primaryCrop = data.recommended_crop;
    const confidenceVal = data.confidence !== null && data.confidence !== undefined ? data.confidence : 0;
    const modelName = data.model_name || "RandomForestClassifier";
    const modelVersion = data.model_version || "1.0.0";
    const scoreType = data.score_type || "uncalibrated model score";
    const timestampFormatted = data.timestamp
      ? new Date(data.timestamp).toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "medium",
        })
      : new Date().toLocaleTimeString();

    // 1. Build Top-3 Ranked Candidates Table
    let rankedCandidatesHtml = "";
    if (data.recommendations && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
      const rows = data.recommendations
        .map((item) => {
          const isWinner = item.rank === 1;
          const conf = item.confidence_percentage !== undefined ? item.confidence_percentage : (item.score * 100).toFixed(1);
          return `
            <div class="ranked-item-row ${isWinner ? "is-winner" : ""}">
              <div class="rank-badge">${item.rank}</div>
              <div class="ranked-crop-name">${item.crop}</div>
              <div class="ranked-progress-bg">
                <div class="ranked-progress-fill" style="width: ${conf}%;"></div>
              </div>
              <div class="ranked-score-val">${conf}%</div>
            </div>
          `;
        })
        .join("");

      rankedCandidatesHtml = `
        <div class="recommendations-ranked-container">
          <div class="ranked-list-title">
            <span>Top-3 Model Ranked Candidates</span>
            <span style="font-size: 0.72rem; text-transform: none; color: var(--text-muted);">Ensemble Voting</span>
          </div>
          <div class="ranked-items-list">
            ${rows}
          </div>
        </div>
      `;
    }

    // 2. Build Validated Feature Chips
    let echoHtml = "";
    if (data.validated_input) {
      echoHtml = `
        <div class="echo-table-container">
          <div class="echo-title">Verified Agro-Climatic Feature Vector</div>
          <div class="echo-grid">
            <div class="echo-chip"><div class="echo-chip-label">N</div><div class="echo-chip-value">${data.validated_input.nitrogen}</div></div>
            <div class="echo-chip"><div class="echo-chip-label">P</div><div class="echo-chip-value">${data.validated_input.phosphorus}</div></div>
            <div class="echo-chip"><div class="echo-chip-label">K</div><div class="echo-chip-value">${data.validated_input.potassium}</div></div>
            <div class="echo-chip"><div class="echo-chip-label">pH</div><div class="echo-chip-value">${data.validated_input.ph}</div></div>
            <div class="echo-chip"><div class="echo-chip-label">Temp</div><div class="echo-chip-value">${data.validated_input.temperature}°C</div></div>
            <div class="echo-chip"><div class="echo-chip-label">Humidity</div><div class="echo-chip-value">${data.validated_input.humidity}%</div></div>
            <div class="echo-chip" style="grid-column: span 2;"><div class="echo-chip-label">Precipitation</div><div class="echo-chip-value">${data.validated_input.rainfall} mm</div></div>
          </div>
        </div>
      `;
    }

    // 3. Assemble Complete Results Card
    resultContainer.className = "recommendation-result has-crop";
    resultContainer.innerHTML = `
      <div class="crop-badge-hero">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        Optimal Crop Viability Identified
      </div>

      <h2 class="crop-name-hero">${primaryCrop}</h2>

      <div class="confidence-bar-wrapper">
        <div class="confidence-header">
          <span>Primary Prediction Confidence</span>
          <strong>${confidenceVal}%</strong>
        </div>
        <div class="confidence-bar-bg">
          <div class="confidence-bar-fill" style="width: ${confidenceVal}%;"></div>
        </div>
      </div>

      ${rankedCandidatesHtml}

      <div class="uncalibrated-callout">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-400)" stroke-width="2" style="flex-shrink: 0; margin-top: 2px;">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="16" x2="12" y2="12"/>
          <line x1="12" y1="8" x2="12.01" y2="8"/>
        </svg>
        <div>
          <strong>Score Type: ${scoreType}</strong> – The confidence score reflects the raw voting proportion across the 100 constituent decision trees in the Random Forest ensemble (votes / 100).
        </div>
      </div>

      ${echoHtml}

      <div class="model-meta-footer">
        <div class="meta-pill-group">
          <span class="meta-pill">${modelName}</span>
          <span class="meta-pill">v${modelVersion}</span>
          <span class="meta-pill">22 Crops Active</span>
        </div>
        <div style="font-family: monospace; font-size: 0.72rem;">
          ${timestampFormatted}
        </div>
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // 7. Reset Button Handler
  // -------------------------------------------------------------------------
  resetBtn.addEventListener("click", () => {
    hideErrors();
    document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
    restoreEmptyPlaceholder();
  });
});
