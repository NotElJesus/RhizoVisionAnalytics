<template>
  <div class="page">
    <div class="container">
      <div class="hero">
        <p class="eyebrow">SOIL ROOT IMAGING PLATFORM</p>
        <h1>Soil Reconstruction Web GUI</h1>
        <p class="subtitle">
          Upload an image, set reconstruction parameters, and submit the job to the backend.
        </p>
      </div>

      <div class="card">
        <h2>Upload Image</h2>
        <div class="upload-box">
          <label class="upload-label">Choose Image File</label>
          <input type="file" @change="handleFileChange" />
          <p class="file-name" v-if="selectedFile">Selected: {{ selectedFile.name }}</p>
          <p class="file-name muted" v-else>No file selected</p>
        </div>
      </div>

      <div class="card">
        <h2>Reconstruction Parameters</h2>
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
      </div>

      <div class="card error-card" v-if="errorMessage">
        <h2>Error</h2>
        <p class="error-text">{{ errorMessage }}</p>
      </div>

      <div class="card response-card" v-if="responseData">
        <h2>Backend Response</h2>
        <pre>{{ responseData }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const selectedFile = ref(null)
const responseData = ref(null)
const errorMessage = ref('')

const form = ref({
  reconstruction_width: 64,
  iterations: 100,
  sourceloc: 30,
  detectors: 55,
  fan_angle_degrees: 80
})

const handleFileChange = (e) => {
  selectedFile.value = e.target.files[0]
}

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
  } catch (err) {
    errorMessage.value = err.message
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
  padding: 40px 20px;
}

.container {
  max-width: 900px;
  margin: 0 auto;
}

.hero {
  margin-bottom: 24px;
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
  font-size: 44px;
  line-height: 1.15;
}

.subtitle {
  margin-top: 12px;
  font-size: 17px;
  color: #4b5563;
}

.card {
  background: white;
  border-radius: 18px;
  padding: 24px;
  margin-top: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
}

h2 {
  margin-top: 0;
  margin-bottom: 18px;
  font-size: 22px;
}

.upload-box {
  padding: 16px;
  border: 1px solid #dbe4dc;
  border-radius: 14px;
  background: #f8fcf9;
}

.upload-label {
  display: block;
  margin-bottom: 10px;
  font-weight: 700;
}

.file-name {
  margin-top: 12px;
  font-size: 14px;
  color: #1f2937;
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
  background: #ffffff;
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
  transition: 0.2s ease;
}

.primary-btn:hover {
  transform: translateY(-1px);
  opacity: 0.96;
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

@media (max-width: 700px) {
  h1 {
    font-size: 32px;
  }

  .form-grid {
    grid-template-columns: 1fr;
  }

  .full-width {
    grid-column: auto;
  }
}
</style>