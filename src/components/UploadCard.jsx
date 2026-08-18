import { ImagePlus, UploadCloud, X } from 'lucide-react'
import { formatFileSize } from '../utils/fileValidation'

export function UploadCard({
  selectedFile,
  previewUrl,
  onSelectFile,
  onDropFile,
  onClearFile,
  busy,
}) {
  return (
    <section className="panel upload-panel" aria-labelledby="upload-heading">
      <div className="panel-head">
        <h2 id="upload-heading">Chest X-Ray Upload</h2>
        <p>Upload a frontal chest radiograph in JPG or PNG format.</p>
      </div>

      <label
        className={`dropzone ${busy ? 'is-busy' : ''}`}
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDropFile}
      >
        <input
          type="file"
          accept=".jpg,.jpeg,.png,image/jpeg,image/png"
          onChange={onSelectFile}
          disabled={busy}
          aria-label="Upload chest X-ray image"
        />
        <UploadCloud size={28} aria-hidden="true" />
        <p>
          Drag and drop an X-ray image here, or <span>browse files</span>
        </p>
        <small>Maximum file size: 8 MB</small>
      </label>

      {selectedFile && previewUrl ? (
        <div className="preview-card" aria-live="polite">
          <div className="preview-meta">
            <ImagePlus size={16} aria-hidden="true" />
            <div>
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">{formatFileSize(selectedFile.size)}</p>
            </div>
          </div>
          <button
            type="button"
            className="ghost-btn"
            onClick={onClearFile}
            disabled={busy}
          >
            <X size={16} aria-hidden="true" />
            Clear
          </button>
          <img src={previewUrl} alt="Uploaded chest X-ray preview" />
        </div>
      ) : (
        <div className="empty-preview" aria-live="polite">
          <p>No image selected yet.</p>
        </div>
      )}
    </section>
  )
}
