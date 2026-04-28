<template>
  <div class="page">
    <header class="navbar">
      <div class="brand">
        <span class="brand-mark">RVA</span>
        <span>RhizoVisionAnalytics</span>
      </div>

      <nav class="nav-right">
        <button
          v-for="page in pages"
          :key="page.id"
          class="nav-btn"
          :class="{ active: currentPage === page.id }"
          type="button"
          @click="currentPage = page.id"
        >
          {{ page.label }}
        </button>
      </nav>
    </header>

    <main class="container">
      <section v-if="currentPage === 'home'" class="hero">
        <p class="eyebrow">SOIL ROOT IMAGING PLATFORM</p>
        <h1>RhizoVisionAnalytics</h1>
        <p class="subtitle">
          Upload a source image or audio scan files, run the reconstruction pipeline, and inspect the generated result.
        </p>

        <button class="primary-btn" type="button" @click="currentPage = 'run'">
          Start Reconstruction
        </button>
      </section>

      <section v-if="currentPage === 'run'" class="workspace">
        <div class="section-heading">
          <div>
            <p class="eyebrow">RECONSTRUCTION</p>
            <h2>Run Pipeline</h2>
          </div>
          <span class="status-pill" :class="{ active: isRunning }">
            {{ isRunning ? 'Running' : 'Ready' }}
          </span>
        </div>

        <div class="run-grid">
          <form class="panel" @submit.prevent="submitForm">
            <div class="mode-switch">
              <button
                class="mode-btn"
                :class="{ active: runMode === 'image' }"
                type="button"
                @click="runMode = 'image'"
              >
                Image Simulation
              </button>
              <button
                class="mode-btn"
                :class="{ active: runMode === 'audio' }"
                type="button"
                @click="runMode = 'audio'"
              >
                Audio Scan
              </button>
            </div>

            <label v-if="runMode === 'image'" class="upload-box">
              <span>Choose Image</span>
              <input type="file" accept="image/*" @change="handleImageChange" />
              <strong>{{ selectedImage ? selectedImage.name : 'No file selected' }}</strong>
            </label>

            <div v-else class="audio-upload-grid">
              <label class="upload-box">
                <span>Baseline / Touching WAV</span>
                <input type="file" accept="audio/*,.wav" @change="handleBaselineChange" />
                <strong>{{ baselineFile ? baselineFile.name : 'No baseline selected' }}</strong>
              </label>

              <label class="upload-box">
                <span>Measurement WAV Files</span>
                <input type="file" accept="audio/*,.wav" multiple @change="handleMeasurementsChange" />
                <strong>{{ measurementFiles.length }} files selected</strong>
              </label>
            </div>

            <div class="form-grid">
              <label class="form-group">
                <span>Reconstruction Width</span>
                <input v-model.number="form.reconstruction_width" min="1" type="number" />
              </label>

              <label class="form-group">
                <span>Iterations</span>
                <input v-model.number="form.iterations" min="0" type="number" />
              </label>

              <label class="form-group">
                <span>Source Location</span>
                <input v-model.number="form.sourceloc" min="1" type="number" />
              </label>

              <label class="form-group">
                <span>Detectors</span>
                <input v-model.number="form.detectors" min="1" type="number" />
              </label>

              <label class="form-group full-width">
                <span>Fan Angle Degrees</span>
                <input v-model.number="form.fan_angle_degrees" min="1" step="0.1" type="number" />
              </label>

              <template v-if="runMode === 'audio'">
                <label class="form-group">
                  <span>Rotations</span>
                  <input v-model.number="audioForm.rotations" min="1" type="number" />
                </label>

                <label class="form-group">
                  <span>Target Frequency</span>
                  <input v-model.number="audioForm.desired_freq" min="1" step="0.1" type="number" />
                </label>

                <label class="form-group">
                  <span>Kernel Size</span>
                  <input v-model.number="audioForm.kernel_size" min="1" step="2" type="number" />
                </label>

                <label class="check-row">
                  <input v-model="audioForm.use_window" type="checkbox" />
                  <span>Use Hanning window</span>
                </label>

                <label class="check-row">
                  <input v-model="audioForm.scale_by_rows" type="checkbox" />
                  <span>Scale by rows</span>
                </label>
              </template>
            </div>

            <button class="primary-btn wide" type="submit" :disabled="isRunning">
              {{ isRunning ? 'Running...' : 'Run Reconstruction' }}
            </button>
          </form>

          <aside class="panel result-panel">
            <div v-if="errorMessage" class="alert error">
              <strong>Error</strong>
              <p>{{ errorMessage }}</p>
            </div>

            <div v-else-if="outputImageUrl" class="result-view">
              <img :src="outputImageUrl" alt="Reconstruction Output" class="result-image" />
              <a :href="outputImageUrl" target="_blank" rel="noopener noreferrer" class="secondary-btn">
                Open Full Image
              </a>
              <div v-if="sinogramImageUrl" class="sinogram-preview">
                <img :src="sinogramImageUrl" alt="Sinogram Output" class="sinogram-image" />
                <a :href="sinogramImageUrl" target="_blank" rel="noopener noreferrer" class="secondary-btn">
                  Open Sinogram
                </a>
              </div>
            </div>

            <div v-else class="empty-state">
              <span class="empty-icon">□</span>
              <p>Result preview will appear here.</p>
            </div>
          </aside>
        </div>

        <details v-if="responseData" class="response-panel">
          <summary>Backend Response</summary>
          <pre>{{ responseText }}</pre>
        </details>
      </section>

      <section v-if="currentPage === 'history'" class="workspace">
        <div class="section-heading">
          <div>
            <p class="eyebrow">LOCAL RECORDS</p>
            <h2>History</h2>
          </div>
          <button class="secondary-btn" type="button" @click="clearHistory" :disabled="historyRecords.length === 0">
            Clear
          </button>
        </div>

        <div v-if="historyRecords.length === 0" class="panel empty-state">
          <p>No history records yet.</p>
        </div>

        <div v-else class="history-list">
          <article v-for="(item, index) in historyRecords" :key="index" class="history-item">
            <div class="history-top">
              <strong>{{ item.filename }}</strong>
              <span>{{ item.time }}</span>
            </div>

            <div class="history-grid">
              <p><strong>Width:</strong> {{ item.params.reconstruction_width }}</p>
              <p><strong>Iterations:</strong> {{ item.params.iterations }}</p>
              <p><strong>Source:</strong> {{ item.params.sourceloc }}</p>
              <p><strong>Detectors:</strong> {{ item.params.detectors }}</p>
              <p><strong>Fan Angle:</strong> {{ item.params.fan_angle_degrees }}</p>
              <p v-if="item.params.rotations"><strong>Rotations:</strong> {{ item.params.rotations }}</p>
              <p v-if="item.params.desired_freq"><strong>Freq:</strong> {{ item.params.desired_freq }}</p>
            </div>

            <a v-if="item.output_url" :href="item.output_url" target="_blank" rel="noopener noreferrer">
              Open result
            </a>
          </article>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const pages = [
  { id: 'home', label: 'Home' },
  { id: 'run', label: 'Run' },
  { id: 'history', label: 'History' }
]

const currentPage = ref('home')
const runMode = ref('image')
const selectedImage = ref(null)
const baselineFile = ref(null)
const measurementFiles = ref([])
const responseData = ref(null)
const errorMessage = ref('')
const historyRecords = ref([])
const outputImageUrl = ref('')
const sinogramImageUrl = ref('')
const isRunning = ref(false)

const form = ref({
  reconstruction_width: 64,
  iterations: 100,
  sourceloc: 30,
  detectors: 55,
  fan_angle_degrees: 80
})

const audioForm = ref({
  rotations: 6,
  desired_freq: 2500,
  kernel_size: 11,
  use_window: true,
  scale_by_rows: false
})

const responseText = computed(() => {
  return responseData.value ? JSON.stringify(responseData.value, null, 2) : ''
})

const loadHistory = () => {
  const saved = localStorage.getItem('rhizo_history')
  historyRecords.value = saved ? JSON.parse(saved) : []
}

const persistHistory = () => {
  localStorage.setItem('rhizo_history', JSON.stringify(historyRecords.value))
}

const saveHistoryRecord = (result) => {
  historyRecords.value.unshift({
    filename: result.filename,
    mode: runMode.value,
    time: new Date().toLocaleString(),
    params: result.params,
    output_url: result.output_url,
    sinogram_url: result.sinogram_url
  })
  persistHistory()
}

const clearHistory = () => {
  historyRecords.value = []
  persistHistory()
}

const handleImageChange = (event) => {
  selectedImage.value = event.target.files?.[0] || null
}

const handleBaselineChange = (event) => {
  baselineFile.value = event.target.files?.[0] || null
}

const handleMeasurementsChange = (event) => {
  measurementFiles.value = Array.from(event.target.files || [])
}

const submitForm = async () => {
  errorMessage.value = ''
  responseData.value = null
  outputImageUrl.value = ''
  sinogramImageUrl.value = ''

  // Route the shared form controls to the selected reconstruction pipeline.
  if (runMode.value === 'audio') {
    await submitAudioForm()
    return
  }

  if (!selectedImage.value) {
    errorMessage.value = 'Please select an image before running.'
    return
  }

  const formData = new FormData()
  formData.append('image', selectedImage.value)
  formData.append('reconstruction_width', form.value.reconstruction_width)
  formData.append('iterations', form.value.iterations)
  formData.append('sourceloc', form.value.sourceloc)
  formData.append('detectors', form.value.detectors)
  formData.append('fan_angle_degrees', form.value.fan_angle_degrees)

  isRunning.value = true

  try {
    const res = await axios.post(`${API_BASE_URL}/run`, formData)
    responseData.value = res.data
    outputImageUrl.value = res.data.output_url || ''
    sinogramImageUrl.value = res.data.sinogram_url || ''
    saveHistoryRecord(res.data)
  } catch (err) {
    const detail = err.response?.data?.detail || err.response?.data?.message || err.message
    errorMessage.value = detail
  } finally {
    isRunning.value = false
  }
}

const submitAudioForm = async () => {
  if (!baselineFile.value) {
    errorMessage.value = 'Please select a baseline WAV before running.'
    return
  }

  if (measurementFiles.value.length === 0) {
    errorMessage.value = 'Please select measurement WAV files before running.'
    return
  }

  const expectedFiles = audioForm.value.rotations * form.value.detectors
  if (measurementFiles.value.length !== expectedFiles) {
    errorMessage.value = `Expected ${expectedFiles} measurement files, got ${measurementFiles.value.length}.`
    return
  }

  const formData = new FormData()
  formData.append('baseline', baselineFile.value)
  // Keep the browser upload order deterministic for scan rows and detectors.
  measurementFiles.value
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
    .forEach((file) => formData.append('measurements', file))
  formData.append('reconstruction_width', form.value.reconstruction_width)
  formData.append('iterations', form.value.iterations)
  formData.append('sourceloc', form.value.sourceloc)
  formData.append('detectors', form.value.detectors)
  formData.append('fan_angle_degrees', form.value.fan_angle_degrees)
  formData.append('rotations', audioForm.value.rotations)
  formData.append('desired_freq', audioForm.value.desired_freq)
  formData.append('kernel_size', audioForm.value.kernel_size)
  formData.append('use_window', audioForm.value.use_window)
  formData.append('scale_by_rows', audioForm.value.scale_by_rows)

  isRunning.value = true

  try {
    const res = await axios.post(`${API_BASE_URL}/run-audio`, formData)
    responseData.value = res.data
    outputImageUrl.value = res.data.output_url || ''
    sinogramImageUrl.value = res.data.sinogram_url || ''
    saveHistoryRecord(res.data)
  } catch (err) {
    const detail = err.response?.data?.detail || err.response?.data?.message || err.message
    errorMessage.value = detail
  } finally {
    isRunning.value = false
  }
}

onMounted(loadHistory)
</script>

<style scoped>
* {
  box-sizing: border-box;
}

.page {
  min-height: 100vh;
  color: #18231f;
  background: linear-gradient(135deg, #f1f6ef 0%, #f8fbff 100%);
}

.navbar {
  width: 100%;
  background: rgba(255, 255, 255, 0.92);
  border-bottom: 1px solid #dfe6df;
  padding: 14px 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: 10;
  backdrop-filter: blur(10px);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 800;
}

.brand-mark {
  display: inline-grid;
  place-items: center;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: #275d4d;
  color: white;
  font-size: 13px;
  letter-spacing: 0;
}

.nav-right {
  display: flex;
  gap: 8px;
}

.nav-btn,
.primary-btn,
.secondary-btn {
  border: 0;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 750;
}

.nav-btn {
  background: transparent;
  padding: 9px 13px;
  color: #31453d;
}

.nav-btn:hover,
.nav-btn.active {
  background: #275d4d;
  color: white;
}

.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 34px 20px 52px;
}

.hero,
.workspace {
  background: white;
  border: 1px solid #dfe6df;
  border-radius: 8px;
  padding: 28px;
  box-shadow: 0 14px 36px rgba(24, 35, 31, 0.08);
}

.hero {
  min-height: 360px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.eyebrow {
  margin: 0 0 8px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0;
  color: #2f735e;
}

h1,
h2 {
  margin: 0;
  letter-spacing: 0;
}

h1 {
  font-size: 48px;
  line-height: 1.08;
}

h2 {
  font-size: 28px;
}

.subtitle {
  max-width: 620px;
  margin: 14px 0 0;
  color: #50645b;
  line-height: 1.6;
}

.section-heading,
.history-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.status-pill {
  border: 1px solid #cdd9d2;
  border-radius: 999px;
  padding: 7px 11px;
  color: #50645b;
  font-size: 13px;
  font-weight: 800;
}

.status-pill.active {
  color: #114c3d;
  background: #dff1e7;
}

.run-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 18px;
  margin-top: 22px;
}

.mode-switch {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 16px;
}

.mode-btn {
  border: 1px solid #cdd9d2;
  border-radius: 8px;
  background: white;
  color: #31453d;
  cursor: pointer;
  font-weight: 800;
  padding: 10px 12px;
}

.mode-btn.active {
  border-color: #275d4d;
  background: #275d4d;
  color: white;
}

.panel,
.history-item,
.response-panel {
  border: 1px solid #dfe6df;
  border-radius: 8px;
  background: #fbfdfb;
  padding: 18px;
}

.upload-box,
.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-weight: 750;
}

.upload-box {
  border: 1px dashed #aebfb4;
  border-radius: 8px;
  padding: 16px;
  background: #f5faf6;
}

.upload-box input {
  max-width: 100%;
}

.audio-upload-grid {
  display: grid;
  gap: 12px;
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 15px;
  margin-top: 16px;
}

.full-width {
  grid-column: 1 / -1;
}

input {
  width: 100%;
  border: 1px solid #cbd8cf;
  border-radius: 8px;
  background: white;
  padding: 11px 12px;
  color: #18231f;
}

input[type='checkbox'] {
  width: auto;
}

.check-row {
  display: flex;
  align-items: center;
  gap: 9px;
  min-height: 45px;
  color: #31453d;
  font-weight: 750;
}

input:focus {
  outline: 3px solid rgba(47, 115, 94, 0.16);
  border-color: #2f735e;
}

.primary-btn {
  margin-top: 22px;
  padding: 12px 18px;
  background: #275d4d;
  color: white;
}

.primary-btn:disabled,
.secondary-btn:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.wide {
  width: 100%;
}

.secondary-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 40px;
  padding: 9px 14px;
  background: #e8f0eb;
  color: #1f4f42;
  text-decoration: none;
}

.result-panel {
  min-height: 360px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.result-view {
  width: 100%;
  text-align: center;
}

.result-image {
  display: block;
  width: min(100%, 260px);
  aspect-ratio: 1;
  object-fit: contain;
  margin: 0 auto 16px;
  border: 1px solid #d7e2da;
  border-radius: 8px;
  background: white;
}

.sinogram-preview {
  margin-top: 18px;
}

.sinogram-image {
  display: block;
  width: min(100%, 260px);
  max-height: 150px;
  object-fit: contain;
  margin: 0 auto 12px;
  border: 1px solid #d7e2da;
  border-radius: 8px;
  background: white;
}

.empty-state {
  width: 100%;
  color: #62766d;
  text-align: center;
}

.empty-icon {
  display: block;
  margin-bottom: 8px;
  color: #8da196;
  font-size: 42px;
  line-height: 1;
}

.alert {
  width: 100%;
  border-radius: 8px;
  padding: 14px;
}

.alert.error {
  background: #fff2f0;
  border: 1px solid #f1b5ae;
  color: #9a2d20;
}

.alert p {
  margin: 8px 0 0;
}

.response-panel {
  margin-top: 18px;
}

.response-panel summary {
  cursor: pointer;
  font-weight: 800;
}

.response-panel pre {
  max-height: 280px;
  overflow: auto;
  margin: 14px 0 0;
  white-space: pre-wrap;
  color: #273a33;
}

.history-list {
  display: grid;
  gap: 14px;
  margin-top: 20px;
}

.history-item a {
  display: inline-block;
  margin-top: 12px;
  color: #1f6d55;
  font-weight: 800;
}

.history-top span {
  color: #64776e;
  font-size: 13px;
}

.history-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.history-grid p {
  margin: 0;
}

@media (max-width: 860px) {
  .navbar,
  .section-heading,
  .history-top {
    align-items: flex-start;
    flex-direction: column;
  }

  .run-grid,
  .form-grid,
  .history-grid,
  .mode-switch {
    grid-template-columns: 1fr;
  }

  h1 {
    font-size: 36px;
  }
}
</style>
