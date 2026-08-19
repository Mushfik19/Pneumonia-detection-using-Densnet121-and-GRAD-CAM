import { useEffect, useMemo, useState } from 'react'
import { Header } from '../components/Header'
import { UploadCard } from '../components/UploadCard'
import { ResultCard } from '../components/ResultCard'
import { GradCamPanel } from '../components/GradCamPanel'
import { HistoryPanel } from '../components/HistoryPanel'
import { ToastRegion } from '../components/ToastRegion'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { Footer } from '../components/Footer'
import { useLocalHistory } from '../hooks/useLocalHistory'
import { checkHealth, predictXray } from '../services/api'
import { validateImageFile } from '../utils/fileValidation'

function makeToast(type, message) {
  return {
    id: `${Date.now()}_${Math.random().toString(16).slice(2)}`,
    type,
    message,
  }
}

function safeRevoke(url) {
  if (url && url.startsWith('blob:')) {
    URL.revokeObjectURL(url)
  }
}

function fileToThumbnailDataUrl(file) {
  return new Promise((resolve, reject) => {
    const sourceUrl = URL.createObjectURL(file)
    const image = new Image()

    image.onload = () => {
      const maxSide = 240
      const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight))
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale))
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale))
      canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height)
      URL.revokeObjectURL(sourceUrl)
      resolve(canvas.toDataURL('image/jpeg', 0.72))
    }

    image.onerror = () => {
      URL.revokeObjectURL(sourceUrl)
      reject(new Error('Cannot create image thumbnail for history.'))
    }

    image.src = sourceUrl
  })
}

export function DashboardPage() {
  const [healthStatus, setHealthStatus] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [toasts, setToasts] = useState([])
  const { history, addEntry, clearHistory } = useLocalHistory()

  const disclaimer = useMemo(
    () =>
      'Research and Educational Use Only. PneumoVision AI is not a medical device and does not provide a medical diagnosis. Results should not replace evaluation by a qualified healthcare professional.',
    [],
  )

  useEffect(() => {
    let active = true

    async function loadHealth() {
      try {
        const data = await checkHealth()
        if (active) {
          setHealthStatus(data)
        }
      } catch {
        if (active) {
          setHealthStatus({ server: 'offline', model_loaded: false })
        }
      }
    }

    loadHealth()
    const timer = setInterval(loadHealth, 20000)

    return () => {
      active = false
      clearInterval(timer)
    }
  }, [])

  function addToast(type, message) {
    const toast = makeToast(type, message)
    setToasts((prev) => [...prev, toast])
    setTimeout(() => {
      setToasts((prev) => prev.filter((item) => item.id !== toast.id))
    }, 4000)
  }

  function applySelectedFile(file) {
    const validation = validateImageFile(file)

    if (!validation.valid) {
      addToast('error', validation.error)
      return
    }

    const nextPreview = URL.createObjectURL(file)

    safeRevoke(previewUrl)

    setSelectedFile(file)
    setPreviewUrl(nextPreview)
    setResult(null)
    setUploadProgress(0)
  }

  function handleSelectFile(event) {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    applySelectedFile(file)
  }

  function handleDropFile(event) {
    event.preventDefault()
    if (loading) {
      return
    }

    const file = event.dataTransfer?.files?.[0]
    if (!file) {
      return
    }

    applySelectedFile(file)
  }

  function handleClearFile() {
    safeRevoke(previewUrl)

    setSelectedFile(null)
    setPreviewUrl('')
    setResult(null)
    setUploadProgress(0)
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      addToast('error', 'Select an X-ray image before running analysis.')
      return
    }

    setLoading(true)
    setUploadProgress(0)

    try {
      const response = await predictXray(selectedFile, setUploadProgress)
      setResult(response)
      const thumbnail = await fileToThumbnailDataUrl(selectedFile)

      addEntry({
        id: `${Date.now()}_${Math.random().toString(16).slice(2)}`,
        thumbnail,
        predictionLabel: response.prediction,
        confidence: response.confidence,
        timestamp: new Date().toISOString(),
        result: response,
      })

      addToast('success', 'Analysis completed successfully.')
    } catch (error) {
      addToast('error', error.message)
    } finally {
      setLoading(false)
    }
  }

  function handleSelectHistoryItem(item) {
    setResult(item.result)
    setPreviewUrl(item.thumbnail)
    setSelectedFile(null)
    addToast('info', 'Loaded a previous analysis result.')
  }

  return (
    <>
    <div className="app-shell" id="top">
      <Header healthStatus={healthStatus} />

      <section className="hero-section">
        <p className="kicker">Intelligent Pneumonia Screening from Chest X-Rays</p>
        <p className="hero-text">
          DenseNet121-assisted interpretability workflow for educational and
          research-oriented chest radiograph triage simulation.
        </p>
        <p className="disclaimer" role="note">
          {disclaimer}
        </p>
      </section>

      <section className="analysis-grid" id="analyze">
        <UploadCard
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          onSelectFile={handleSelectFile}
          onDropFile={handleDropFile}
          onClearFile={handleClearFile}
          busy={loading}
        />

        <ErrorBoundary title="Prediction result unavailable" id="model">
          <ResultCard result={result} loading={loading} />
        </ErrorBoundary>
      </section>

      <div className="cta-row">
        <button
          type="button"
          className="primary-btn"
          onClick={handleAnalyze}
          disabled={loading}
        >
          {loading ? 'Analyzing X-ray...' : 'Analyze X-Ray'}
        </button>

        {loading ? (
          <p className="upload-progress" aria-live="polite">
            Upload Progress: {uploadProgress}%
          </p>
        ) : null}
      </div>

      <section className="bottom-grid">
        <ErrorBoundary title="Model attention visualization unavailable">
          <GradCamPanel result={result} originalPreview={previewUrl} />
        </ErrorBoundary>
        <HistoryPanel
          history={history}
          onClear={clearHistory}
          onSelect={handleSelectHistoryItem}
        />
      </section>

      <ToastRegion toasts={toasts} />
    </div>
      <Footer />
    </>
  )
}
