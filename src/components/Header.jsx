import { ActivitySquare, ShieldAlert, Wifi } from 'lucide-react'

export function Header({ healthStatus }) {
  const online = healthStatus?.status === 'ok' || healthStatus?.server === 'ok'
  const modelLoaded = healthStatus?.model_loaded

  return (
    <header className="app-header">
      <div className="brand-wrap">
        <div className="brand-icon" aria-hidden="true">
          <ActivitySquare size={24} />
        </div>
        <div>
          <p className="eyebrow">AI-Powered Chest X-Ray Analysis</p>
          <h1>PneumoVision AI</h1>
        </div>
      </div>

      <div className="status-group" role="status" aria-live="polite">
        <span className={`status-pill ${online ? 'online' : 'offline'}`}>
          <Wifi size={14} aria-hidden="true" />
          {online ? 'Backend Online' : 'Backend Offline'}
        </span>
        <span className={`status-pill ${modelLoaded ? 'ready' : 'warning'}`}>
          <ShieldAlert size={14} aria-hidden="true" />
          {modelLoaded ? 'Model Ready' : 'Model Not Loaded'}
        </span>
      </div>
    </header>
  )
}
