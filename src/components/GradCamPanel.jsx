import { Download, Image, Layers3 } from 'lucide-react'
import { useMemo, useState } from 'react'

function downloadDataUrl(dataUrl, filename) {
  const anchor = document.createElement('a')
  anchor.href = dataUrl
  anchor.download = filename
  anchor.click()
}

export function GradCamPanel({ result, originalPreview }) {
  const [tab, setTab] = useState('original')

  const gradcamDataUrl = useMemo(() => {
    if (!result?.gradcam_image_base64) {
      return null
    }
    return `data:image/png;base64,${result.gradcam_image_base64}`
  }, [result])

  return (
    <section className="panel gradcam-panel" aria-labelledby="gradcam-heading">
      <div className="panel-head">
        <h2 id="gradcam-heading">Model Attention Visualization</h2>
        <p>
          Highlighted regions represent areas that contributed more strongly to
          the model prediction. This visualization is provided for model
          interpretability and should not be interpreted as a clinical diagnosis.
        </p>
      </div>

      {!result ? (
        <div className="empty-state">
          <Layers3 size={24} aria-hidden="true" />
          <p>Grad-CAM appears after a successful prediction.</p>
        </div>
      ) : !result.gradcam_available || !gradcamDataUrl ? (
        <div className="empty-state warning-state">
          <Layers3 size={24} aria-hidden="true" />
          <p>
            Grad-CAM is unavailable for this prediction. The selected model may
            not expose a compatible convolutional layer.
          </p>
        </div>
      ) : (
        <>
          <div className="tabs" role="tablist" aria-label="Image visualization modes">
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'original'}
              className={tab === 'original' ? 'active' : ''}
              onClick={() => setTab('original')}
            >
              <Image size={15} aria-hidden="true" /> Original X-Ray
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'gradcam'}
              className={tab === 'gradcam' ? 'active' : ''}
              onClick={() => setTab('gradcam')}
            >
              <Layers3 size={15} aria-hidden="true" /> Grad-CAM
            </button>
          </div>

          <div className="gradcam-stage">
            {tab === 'original' ? (
              <img src={originalPreview} alt="Original uploaded X-ray" />
            ) : (
              <img src={gradcamDataUrl} alt="Grad-CAM attention overlay" />
            )}
          </div>

          <button
            type="button"
            className="secondary-btn"
            onClick={() => downloadDataUrl(gradcamDataUrl, 'pneumovision-gradcam.png')}
          >
            <Download size={16} aria-hidden="true" /> Download Visualization
          </button>
        </>
      )}
    </section>
  )
}
