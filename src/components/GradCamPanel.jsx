import { Columns2, Download, Image as ImageIcon, Info, Layers3 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

function downloadDataUrl(dataUrl, filename) {
  const anchor = document.createElement('a')
  anchor.href = dataUrl
  anchor.download = filename
  anchor.click()
}

export function GradCamPanel({ result, originalPreview }) {
  const [tab, setTab] = useState('gradcam')

  const gradcamDataUrl = useMemo(() => {
    const raw = result?.gradcam_image_base64 || result?.gradcam
    if (!raw) {
      return null
    }
    return `data:image/png;base64,${raw}`
  }, [result])

  // Automatically show the Grad-CAM visualization when a new prediction arrives
  useEffect(() => {
    if (gradcamDataUrl) {
      setTab('gradcam')
    } else if (originalPreview) {
      setTab('original')
    }
  }, [gradcamDataUrl, originalPreview])

  return (
    <section className="panel gradcam-panel" aria-labelledby="gradcam-heading">
      <div className="panel-head">
        <h2 id="gradcam-heading">Explainable AI / Grad-CAM Attention</h2>
        <p>
          Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the
          spatial regions in the chest radiograph that most strongly influenced
          the DenseNet121 model&apos;s decision.
        </p>
      </div>

      {!result ? (
        <div className="empty-state">
          <Layers3 size={24} aria-hidden="true" />
          <p>Grad-CAM attention heatmap appears here after running an analysis.</p>
        </div>
      ) : !result.gradcam_available || !gradcamDataUrl ? (
        <div className="empty-state warning-state">
          <Layers3 size={24} aria-hidden="true" />
          <p>
            Grad-CAM explainability is temporarily unavailable for this prediction.
            The core prediction result above remains valid.
          </p>
        </div>
      ) : (
        <>
          <div className="tabs" role="tablist" aria-label="Image visualization modes">
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'gradcam'}
              className={tab === 'gradcam' ? 'active' : ''}
              onClick={() => setTab('gradcam')}
            >
              <Layers3 size={15} aria-hidden="true" /> Grad-CAM Overlay
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'split'}
              className={tab === 'split' ? 'active' : ''}
              onClick={() => setTab('split')}
            >
              <Columns2 size={15} aria-hidden="true" /> Side-by-Side
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'original'}
              className={tab === 'original' ? 'active' : ''}
              onClick={() => setTab('original')}
            >
              <ImageIcon size={15} aria-hidden="true" /> Original X-Ray
            </button>
          </div>

          <div className="gradcam-stage">
            {tab === 'gradcam' ? (
              <div className="single-view">
                <img src={gradcamDataUrl} alt="Grad-CAM attention overlay heatmap on chest X-ray" />
                <span className="view-badge">Grad-CAM Overlay</span>
              </div>
            ) : tab === 'original' ? (
              <div className="single-view">
                <img src={originalPreview} alt="Original uploaded chest X-ray" />
                <span className="view-badge">Original X-Ray</span>
              </div>
            ) : (
              <div className="split-view">
                <div className="split-col">
                  <span className="split-label">Original X-Ray</span>
                  <img src={originalPreview} alt="Original uploaded chest radiograph" />
                </div>
                <div className="split-col">
                  <span className="split-label">Grad-CAM Attention Overlay</span>
                  <img src={gradcamDataUrl} alt="Grad-CAM heat overlay on chest radiograph" />
                </div>
              </div>
            )}
          </div>

          <div className="gradcam-caption" role="note">
            <Info size={16} aria-hidden="true" className="info-icon" />
            <p>
              <strong>Explainability Note:</strong> Grad-CAM highlights image regions
              that influenced the model&apos;s prediction. It is provided for model
              interpretability and should not be considered a medical diagnosis.
            </p>
          </div>

          <button
            type="button"
            className="secondary-btn"
            onClick={() => downloadDataUrl(gradcamDataUrl, 'pneumovision-gradcam-overlay.png')}
          >
            <Download size={16} aria-hidden="true" /> Download Visualization
          </button>
        </>
      )}
    </section>
  )
}
