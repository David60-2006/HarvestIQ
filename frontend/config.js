/**
 * HarvestIQ – Frontend Configuration & Constants
 *
 * Dynamically resolves API base URL from window global, localStorage,
 * or defaults to localhost for development.
 */

function resolveApiBaseUrl() {
  // 1. Injected global override (useful in automated tests or script tags)
  if (typeof window !== "undefined" && window.__HARVESTIQ_API_URL__) {
    return window.__HARVESTIQ_API_URL__.replace(/\/+$/, "");
  }

  // 2. User/Evaluator localStorage override (allows pointing to Render backend from UI)
  if (typeof window !== "undefined" && window.localStorage) {
    try {
      const stored = localStorage.getItem("HARVESTIQ_API_URL");
      if (stored && stored.trim()) {
        return stored.trim().replace(/\/+$/, "");
      }
    } catch (_) {
      // LocalStorage access may be restricted in sandboxed iframes
    }
  }

  // 3. Fallback: local FastAPI development server
  return "http://127.0.0.1:8000";
}

const CONFIG = {
  // FastAPI Backend Base URL (dynamic resolution)
  API_BASE_URL: resolveApiBaseUrl(),

  // Runtime helper to reconfigure backend URL without redeploying code
  setApiBaseUrl: function(url) {
    if (typeof window !== "undefined" && window.localStorage) {
      if (!url || !url.trim()) {
        localStorage.removeItem("HARVESTIQ_API_URL");
      } else {
        localStorage.setItem("HARVESTIQ_API_URL", url.trim().replace(/\/+$/, ""));
      }
      window.location.reload();
    }
  },

  ENDPOINTS: {
    HEALTH: "/health",
    MODEL_INFO: "/model-info",
    RECOMMEND: "/recommend",
    PARAMETERS: "/parameters",
  },

  // Agro-climatic parameter boundaries and metadata
  PARAMETERS: {
    nitrogen: {
      min: 0,
      max: 200,
      defaultVal: 90,
      step: 1,
      unit: "kg/ha",
      label: "Nitrogen (N)",
      hint: "Essential for vegetative leaf growth & protein synthesis",
    },
    phosphorus: {
      min: 0,
      max: 200,
      defaultVal: 42,
      step: 1,
      unit: "kg/ha",
      label: "Phosphorus (P)",
      hint: "Crucial for root development, flowering & seed formation",
    },
    potassium: {
      min: 0,
      max: 300,
      defaultVal: 43,
      step: 1,
      unit: "kg/ha",
      label: "Potassium (K)",
      hint: "Enhances drought resistance, disease tolerance & enzyme activation",
    },
    temperature: {
      min: 0,
      max: 50,
      defaultVal: 20.8,
      step: 0.1,
      unit: "°C",
      label: "Temperature",
      hint: "Average ambient temperature during active growing season",
    },
    humidity: {
      min: 0,
      max: 100,
      defaultVal: 82.0,
      step: 0.5,
      unit: "%",
      label: "Humidity",
      hint: "Relative atmospheric humidity in the crop canopy",
    },
    ph: {
      min: 3.0,
      max: 10.0,
      defaultVal: 6.5,
      step: 0.1,
      unit: "pH",
      label: "Soil pH",
      hint: "Acidity (3–6), neutral (6–7.5), or alkaline (7.5–10)",
    },
    rainfall: {
      min: 0,
      max: 400,
      defaultVal: 202.9,
      step: 0.5,
      unit: "mm",
      label: "Rainfall",
      hint: "Expected seasonal precipitation / irrigation equivalent",
    },
  },

  // Quick preset profiles for testing
  PRESETS: [
    {
      id: "rice",
      name: "Rice / High Moisture",
      desc: "Warm, tropical conditions with high rainfall & humidity",
      values: {
        nitrogen: 90,
        phosphorus: 42,
        potassium: 43,
        temperature: 20.9,
        humidity: 82.0,
        ph: 6.5,
        rainfall: 202.9,
      },
    },
    {
      id: "maize",
      name: "Maize / Grain Corn",
      desc: "Moderate climate with balanced soil nutrients",
      values: {
        nitrogen: 80,
        phosphorus: 48,
        potassium: 20,
        temperature: 23.5,
        humidity: 65.0,
        ph: 6.2,
        rainfall: 95.0,
      },
    },
    {
      id: "chickpea",
      name: "Chickpea / Legume",
      desc: "Arid/semi-arid dry season, low nitrogen requirement",
      values: {
        nitrogen: 40,
        phosphorus: 68,
        potassium: 79,
        temperature: 18.9,
        humidity: 16.9,
        ph: 7.3,
        rainfall: 80.1,
      },
    },
    {
      id: "cotton",
      name: "Cotton / Cash Crop",
      desc: "Warm summer conditions, high potassium demand",
      values: {
        nitrogen: 120,
        phosphorus: 45,
        potassium: 20,
        temperature: 24.0,
        humidity: 80.0,
        ph: 6.8,
        rainfall: 80.0,
      },
    },
  ],
};

// Export configuration globally for browser access
if (typeof window !== "undefined") {
  window.CONFIG = CONFIG;
}
