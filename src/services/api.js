import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
})

function normalizeApiError(error) {
  if (error.response?.data?.detail) {
    const detail = error.response.data.detail
    return typeof detail === 'string'
      ? detail
      : detail.message || 'Prediction request failed.'
  }

  if (error.code === 'ECONNABORTED') {
    return 'Request timed out. Please try again.'
  }

  if (error.message === 'Network Error') {
    return 'Cannot reach backend service. Ensure the API server is running.'
  }

  return 'Unexpected server error occurred.'
}

export async function checkHealth() {
  const response = await apiClient.get('/health')
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
