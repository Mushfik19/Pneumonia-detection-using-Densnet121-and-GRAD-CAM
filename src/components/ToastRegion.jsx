import { AlertCircle, CheckCircle2, Info } from 'lucide-react'

const ICONS = {
  error: AlertCircle,
  success: CheckCircle2,
  info: Info,
}

export function ToastRegion({ toasts }) {
  return (
    <div className="toast-region" aria-live="polite" aria-atomic="true">
      {toasts.map((toast) => {
        const Icon = ICONS[toast.type] || Info
        return (
          <article key={toast.id} className={`toast toast-${toast.type}`}>
            <Icon size={18} aria-hidden="true" />
            <p>{toast.message}</p>
          </article>
        )
      })}
    </div>
  )
}
