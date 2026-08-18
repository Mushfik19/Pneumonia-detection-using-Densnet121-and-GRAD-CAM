import { ActivitySquare, LogOut, ShieldAlert, UserRound, Wifi } from 'lucide-react'

export function Header({ healthStatus, user, onLogin, onLogout }) {
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
        {user ? (
          <>
            <span className="user-pill"><span className="avatar">{user.initials}</span>{user.name}</span>
            <button type="button" className="header-action logout-action" onClick={onLogout}><LogOut size={14} /> Logout</button>
          </>
        ) : (
          <button type="button" className="header-action" onClick={onLogin}><UserRound size={14} /> Login</button>
        )}
      </div>
    </header>
  )
}
