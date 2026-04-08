<template>
  <div class="page">
    <header class="navbar">
      <div class="nav-left">
        <div class="logo">RhizoVisionAnalytics</div>
      </div>

      <nav class="nav-right">
        <button
          class="nav-btn"
          :class="{ active: currentPage === 'home' }"
          @click="currentPage = 'home'"
        >
          Home
        </button>

        <button
          class="nav-btn"
          :class="{ active: currentPage === 'run' }"
          @click="currentPage = 'run'"
        >
          Run
        </button>

        <button
          class="nav-btn"
          :class="{ active: currentPage === 'history' }"
          @click="currentPage = 'history'"
        >
          History
        </button>
      </nav>
    </header>

    <main class="container">
      <section v-if="currentPage === 'home'" class="card">
        <p class="eyebrow">SOIL ROOT IMAGING PLATFORM</p>
        <h1>Welcome to RhizoVisionAnalytics</h1>
        <p class="subtitle">
          This platform supports image upload, reconstruction parameter setting,
          result review, and future integration with the reconstruction algorithm.
        </p>
      </section>

      <section v-if="currentPage === 'run'" class="card">
        <h2>Run Reconstruction</h2>

        <div class="upload-box">
          <label>Choose Image</label>
          <input type="file" @change="handleFileChange" />
          <p class="file-name" v-if="selectedFile">Selected: {{ selectedFile.name }}</p>
          <p class="file-name muted" v-else>No file selected</p>
        </div>

        <div class="form-grid">
          <div class="form-group">
            <label>Reconstruction Width</label>
            <input v-model="form.reconstruction_width" type="number" />
          </div>

          <div class="form-group">
            <label>Iterations</label>
            <input v-model="form.iterations" type="number" />
          </div>

          <div class="form-group">
            <label>Sourceloc</label>
            <input v-model="form.sourceloc" type="number" />
          </div>

          <div class="form-group">
            <label>Detectors</label>
            <input v-model="form.detectors" type="number" />
          </div>

          <div class="form-group full-width">
            <label>Fan Angle Degrees</label>
            <input v-model="form.fan_angle_degrees" type="number" step="0.1" />
          </div>
        </div>

        <button class="primary-btn" @click="submitForm">Run Reconstruction</button>

        <div class="card inner-card error-card" v-if="errorMessage">
          <h3>Error</h3>
          <p class="error-text">{{ errorMessage }}</p>
        </div>

        <div class="card inner-card response-card" v-if="responseData">
          <h3>Backend Response</h3>
          <pre>{{ responseData }}</pre>
        </div>
      </section>

      <section v-if="currentPage === 'history'" class="card">
        <h2>History</h2>
        <p class="subtitle">
          This page shows previous submission records from the current browser.
        </p>

        <div v-if="historyRecords.length === 0" class="history-placeholder">
          <p>No history records yet.</p>
        </div>

        <div v-else class="history-list">
          <div
            v-for="(item, index) in historyRecords"
            :key="index"
            class="history-item"
        >
            <div class="history-top">
              <strong>{{ item.filename }}</strong>
              <span class="history-time">{{ item.time }}</span>
            </div>

            <div class="history-grid">
              <p><strong>Width:</strong> {{ item.params.reconstruction_width }}</p>
              <p><strong>Iterations:</strong> {{ item.params.iterations }}</p>
              <p><strong>Sourceloc:</strong> {{ item.params.sourceloc }}</p>
              <p><strong>Detectors:</strong> {{ item.params.detectors }}</p>
              <p><strong>Fan Angle:</strong> {{ item.params.fan_angle_degrees }}</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const currentPage = ref('home')
const selectedFile = ref(null)
const responseData = ref(null)
const errorMessage = ref('')
const historyRecords = ref([])

const form = ref({
  reconstruction_width: 64,
  iterations: 100,
  sourceloc: 30,
  detectors: 55,
  fan_angle_degrees: 80
})

const loadHistory = () => {
  const saved = localStorage.getItem('rhizo_history')
  historyRecords.value = saved ? JSON.parse(saved) : []
}

const saveHistoryRecord = (filename, params) => {
  const newRecord = {
    filename,
    time: new Date().toLocaleString(),
    params
  }

  historyRecords.value.unshift(newRecord)
  localStorage.setItem('rhizo_history', JSON.stringify(historyRecords.value))
}

const handleFileChange = (e) => {
  selectedFile.value = e.target.files[0]
}

onMounted(() => {
  loadHistory()
})

const submitForm = async () => {
  errorMessage.value = ''
  responseData.value = null

  if (!selectedFile.value) {
    errorMessage.value = 'Please select an image before running.'
    return
  }

  const formData = new FormData()
  formData.append('image', selectedFile.value)
  formData.append('reconstruction_width', form.value.reconstruction_width)
  formData.append('iterations', form.value.iterations)
  formData.append('sourceloc', form.value.sourceloc)
  formData.append('detectors', form.value.detectors)
  formData.append('fan_angle_degrees', form.value.fan_angle_degrees)

  try {
  const res = await axios.post('http://localhost:8000/run', formData)
  responseData.value = res.data

  saveHistoryRecord(selectedFile.value.name, {
    reconstruction_width: form.value.reconstruction_width,
    iterations: form.value.iterations,
    sourceloc: form.value.sourceloc,
    detectors: form.value.detectors,
    fan_angle_degrees: form.value.fan_angle_degrees
  })

  currentPage.value = 'run'
} catch (err) {
    errorMessage.value = err.message
    currentPage.value = 'run'
  }
}
</script>

<style>
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: linear-gradient(135deg, #eef4f1 0%, #f8fbff 100%);
  color: #1f2937;
}

.page {
  min-height: 100vh;
}

.navbar {
  width: 100%;
  background: white;
  border-bottom: 1px solid #e5e7eb;
  padding: 16px 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
}

.logo {
  font-size: 22px;
  font-weight: 700;
  color: #2e7d5b;
}

.nav-right {
  display: flex;
  gap: 10px;
}

.nav-btn {
  border: none;
  background: transparent;
  padding: 10px 14px;
  border-radius: 10px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 600;
  color: #374151;
}

.nav-btn:hover {
  background: #f3f4f6;
}

.nav-btn.active {
  background: #2e7d5b;
  color: white;
}

.container {
  max-width: 960px;
  margin: 0 auto;
  padding: 32px 20px;
}

.card {
  background: white;
  border-radius: 18px;
  padding: 24px;
  margin-top: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
}

.inner-card {
  margin-top: 24px;
}

.eyebrow {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 2px;
  color: #2e7d5b;
  margin-bottom: 10px;
}

h1 {
  margin: 0;
  font-size: 40px;
  line-height: 1.2;
}

h2 {
  margin-top: 0;
  margin-bottom: 18px;
  font-size: 26px;
}

h3 {
  margin-top: 0;
  margin-bottom: 12px;
}

.subtitle {
  margin-top: 12px;
  font-size: 16px;
  color: #4b5563;
  line-height: 1.6;
}

.upload-box {
  padding: 16px;
  border: 1px solid #dbe4dc;
  border-radius: 14px;
  background: #f8fcf9;
  margin-bottom: 20px;
}

.file-name {
  margin-top: 12px;
  font-size: 14px;
}

.muted {
  color: #6b7280;
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
}

.full-width {
  grid-column: 1 / -1;
}

label {
  margin-bottom: 8px;
  font-weight: 700;
  color: #374151;
}

input {
  width: 100%;
  padding: 12px 14px;
  border: 1px solid #cfd8d3;
  border-radius: 12px;
  font-size: 15px;
  background: white;
}

input:focus {
  outline: none;
  border-color: #2e7d5b;
  box-shadow: 0 0 0 3px rgba(46, 125, 91, 0.12);
}

.primary-btn {
  margin-top: 22px;
  padding: 13px 20px;
  border: none;
  border-radius: 12px;
  background: #2e7d5b;
  color: white;
  font-size: 15px;
  font-weight: 700;
  cursor: pointer;
}

.primary-btn:hover {
  opacity: 0.95;
}

.error-card {
  border-left: 5px solid #d93025;
}

.error-text {
  color: #d93025;
  font-weight: 700;
}

.response-card pre {
  margin: 0;
  padding: 16px;
  border-radius: 12px;
  background: #f4f7f6;
  overflow-x: auto;
  font-size: 14px;
  line-height: 1.6;
}

.history-placeholder {
  margin-top: 16px;
  padding: 24px;
  border: 1px dashed #cbd5e1;
  border-radius: 14px;
  background: #f8fafc;
  color: #6b7280;
  text-align: center;
}

.history-list {
  margin-top: 18px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.history-item {
  border: 1px solid #dbe4dc;
  border-radius: 14px;
  padding: 16px;
  background: #f9fcfa;
}

.history-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  gap: 12px;
}

.history-time {
  font-size: 13px;
  color: #6b7280;
}

.history-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 16px;
}

.history-grid p {
  margin: 0;
  font-size: 14px;
}

@media (max-width: 700px) {
  .navbar {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }

  .form-grid {
    grid-template-columns: 1fr;
  }

  .full-width {
    grid-column: auto;
  }

  h1 {
    font-size: 30px;
  }
  .history-top {
    flex-direction: column;
    align-items: flex-start;
  }

  .history-grid {
    grid-template-columns: 1fr;
  }
}
</style>