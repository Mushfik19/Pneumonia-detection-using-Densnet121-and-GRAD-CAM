import { Clock3, Trash2 } from 'lucide-react'

function toHumanTime(timestamp) {
  try {
    return new Date(timestamp).toLocaleString()
  } catch {
    return timestamp
  }
}

export function HistoryPanel({ history, onSelect, onClear }) {
  return (
    <section className="panel history-panel" aria-labelledby="history-heading">
      <div className="panel-head row">
        <div>
          <h2 id="history-heading">Session History</h2>
          <p>Stored locally in your browser for the current session.</p>
        </div>
        <button
          type="button"
          className="ghost-btn"
          onClick={onClear}
          disabled={history.length === 0}
        >
          <Trash2 size={14} aria-hidden="true" /> Clear
        </button>
      </div>

      {history.length === 0 ? (
        <div className="empty-state">
          <Clock3 size={20} aria-hidden="true" />
          <p>No previous analyses yet.</p>
        </div>
      ) : (
        <ul className="history-list">
          {history.map((item) => (
            <li key={item.id}>
              <button type="button" onClick={() => onSelect(item)}>
                <img src={item.thumbnail} alt="Previous X-ray thumbnail" />
                <div>
                  <strong>{item.predictionLabel}</strong>
                  <p>{(item.confidence * 100).toFixed(1)}% confidence</p>
                  <small>{toHumanTime(item.timestamp)}</small>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
