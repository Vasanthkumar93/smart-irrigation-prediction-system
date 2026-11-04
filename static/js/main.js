/**
 * Smart Irrigation Prediction System - Interactive Frontend Logic
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Mobile Menu Toggle
  const menuToggle = document.getElementById('menuToggle');
  const navMenu = document.getElementById('navMenu');
  if (menuToggle && navMenu) {
    menuToggle.addEventListener('click', () => {
      navMenu.classList.toggle('open');
    });
  }

  // 2. Range Slider & Number Input Synchronization
  const syncPairs = [
    { range: 'sm_range', input: 'soil_moisture' },
    { range: 'temp_range', input: 'temperature' },
    { range: 'hum_range', input: 'humidity' },
    { range: 'rain_range', input: 'rainfall' }
  ];

  syncPairs.forEach(pair => {
    const rangeEl = document.getElementById(pair.range);
    const inputEl = document.getElementById(pair.input);

    if (rangeEl && inputEl) {
      rangeEl.addEventListener('input', (e) => {
        inputEl.value = parseFloat(e.target.value).toFixed(1);
      });

      inputEl.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (!isNaN(val)) {
          rangeEl.value = val;
        }
      });
    }
  });

  // 3. Quick Scenario Presets
  const presets = {
    drought: { sm: 11.5, temp: 42.0, hum: 18.0, rain: 0.0 },
    summer: { sm: 18.2, temp: 36.5, hum: 30.0, rain: 0.0 },
    spring: { sm: 42.0, temp: 27.5, hum: 62.0, rain: 1.5 },
    monsoon: { sm: 65.0, temp: 22.0, hum: 85.0, rain: 24.0 }
  };

  document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', function () {
      const scenario = this.getAttribute('data-preset');
      const data = presets[scenario];
      if (data) {
        setFieldValue('soil_moisture', 'sm_range', data.sm);
        setFieldValue('temperature', 'temp_range', data.temp);
        setFieldValue('humidity', 'hum_range', data.hum);
        setFieldValue('rainfall', 'rain_range', data.rain);
      }
    });
  });

  function setFieldValue(inputId, rangeId, value) {
    const inputEl = document.getElementById(inputId);
    const rangeEl = document.getElementById(rangeId);
    if (inputEl) inputEl.value = value;
    if (rangeEl) rangeEl.value = value;
  }

  // 4. Alert Auto-Dismiss
  document.querySelectorAll('.alert-close').forEach(btn => {
    btn.addEventListener('click', function () {
      const alert = this.closest('.alert');
      if (alert) {
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 250);
      }
    });
  });

  // 5. CSV Export for Prediction History
  const exportBtn = document.getElementById('exportHistoryBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', function () {
      exportTableToCSV('smart_irrigation_prediction_history.csv');
    });
  }

  function exportTableToCSV(filename) {
    const table = document.getElementById('historyTable');
    if (!table) return;

    let csv = [];
    const rows = table.querySelectorAll('tr');

    for (let i = 0; i < rows.length; i++) {
      const row = [], cols = rows[i].querySelectorAll('td, th');
      for (let j = 0; j < cols.length; j++) {
        // Strip inner newlines and double quotes
        let text = cols[j].innerText.replace(/"/g, '""').trim();
        row.push('"' + text + '"');
      }
      csv.push(row.join(','));
    }

    const csvFile = new Blob([csv.join('\n')], { type: 'text/csv' });
    const downloadLink = document.createElement('a');
    downloadLink.download = filename;
    downloadLink.href = window.URL.createObjectURL(csvFile);
    downloadLink.style.display = 'none';
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
  }
});
