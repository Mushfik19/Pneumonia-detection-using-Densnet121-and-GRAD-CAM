import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const HEALTH_TIMEOUT_MS = 15000
const PREDICTION_TIMEOUT_MS = 120000

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: PREDICTION_TIMEOUT_MS,
})

function normalizeApiError(error) {
  if (error.response?.data?.detail) {
    const detail = error.response.data.detail
    return typeof detail === 'string'
      ? detail
      : detail.message || 'Prediction request failed.'
  }

  if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
    return 'The AI server is waking up or processing the X-ray. Please wait and try again.'
  }

  if (error.message === 'Network Error') {
    return 'Cannot reach backend service. Ensure the API server is running.'
  }

  return 'Unexpected server error occurred.'
}

export async function checkHealth() {
  const response = await apiClient.get('/health', { timeout: HEALTH_TIMEOUT_MS })
  return response.data
}

export async function predictXray(file, onUploadProgress) {
  const formData = new FormData()
  formData.append('image', file)

  try {
    const response = await apiClient.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (event) => {
        if (!event.total || !onUploadProgress) {
          return
        }

        const progress = Math.round((event.loaded * 100) / event.total)
        onUploadProgress(progress)
      },
    })

    return response.data
  } catch (error) {
    throw new Error(normalizeApiError(error))
  }
}
