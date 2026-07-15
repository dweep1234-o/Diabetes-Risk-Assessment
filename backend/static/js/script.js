const form = document.getElementById("risk-form");
const submitBtn = document.getElementById("submit-btn");
const resetBtn = document.getElementById("reset-btn");
const errorBox = document.getElementById("error-box");
const emptyState = document.getElementById("empty-state");
const resultMeta = document.getElementById("result-meta");
const riskLabel = document.getElementById("risk-label");
const gaugeValue = document.getElementById("gauge-value");
const gaugeArc = document.getElementById("gauge-arc");
const gaugeNeedle = document.getElementById("gauge-needle");
const metaModel = document.getElementById("meta-model");
const metaCategory = document.getElementById("meta-category");

const ARC_LENGTH = 251.2; // circumference of the semicircle path

function setGauge(probability) {
  // probability: 0..1
  const offset = ARC_LENGTH * (1 - probability);
  gaugeArc.style.strokeDashoffset = offset;

  // needle sweeps from -90deg (0%) to +90deg (100%)
  const angle = -90 + probability * 180;
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;

  gaugeValue.textContent = `${Math.round(probability * 100)}%`;
}

function labelFor(probability) {
  if (probability < 0.33) return { text: "Low Risk", cls: "low" };
  if (probability < 0.66) return { text: "Moderate Risk", cls: "mid" };
  return { text: "High Risk", cls: "high" };
}

function showError(messages) {
  errorBox.hidden = false;
  errorBox.innerHTML = Array.isArray(messages)
    ? messages.map((m) => `&bull; ${m}`).join("<br>")
    : messages;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = value;
  }

  submitBtn.disabled = true;
  submitBtn.querySelector("span").textContent = "Assessing...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.details || data.error || "Something went wrong.");
      return;
    }

    emptyState.hidden = true;
    resultMeta.hidden = false;

    setGauge(data.probability);
    const { text, cls } = labelFor(data.probability);
    riskLabel.textContent = text;
    riskLabel.className = `risk-label ${cls}`;

    metaModel.textContent = data.model_name.replaceAll("_", " ");
    metaCategory.textContent = data.risk_label;
  } catch (err) {
    showError("Could not reach the prediction service. Please try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector("span").textContent = "Assess risk";
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  clearError();
  emptyState.hidden = false;
  resultMeta.hidden = true;
  riskLabel.textContent = "Enter details and assess to see a result";
  riskLabel.className = "risk-label";
  gaugeValue.textContent = "—";
  gaugeArc.style.strokeDashoffset = ARC_LENGTH;
  gaugeNeedle.style.transform = "rotate(-90deg)";
});
