import { Eye, EyeOff, LockKeyhole, Mail, X } from 'lucide-react'
import { useState } from 'react'

export function LoginModal({ onClose, onSignIn }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [remember, setRemember] = useState(true)
  const [error, setError] = useState('')

  function handleSubmit(event) {
    event.preventDefault()
    if (!/^\S+@\S+\.\S+$/.test(email) || password.length < 6) {
      setError('Enter a valid email and a password with at least 6 characters.')
      return
    }

    // This project has no identity provider. This creates a local browser session only.
    onSignIn({ email, remember })
    onClose()
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="login-modal" role="dialog" aria-modal="true" aria-labelledby="login-heading" onMouseDown={(event) => event.stopPropagation()}>
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close login dialog">
          <X size={18} />
        </button>
        <p className="login-kicker">PneumoVision AI</p>
        <h2 id="login-heading">Welcome Back</h2>
        <p className="login-copy">Sign in to personalize this local research workspace.</p>
        <p className="local-auth-note">Local development session only — no credentials are sent to or verified by a server.</p>

        <form onSubmit={handleSubmit} noValidate>
          <label>
            Email
            <span className="field-wrap"><Mail size={16} /><input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" placeholder="you@example.com" /></span>
          </label>
          <label>
            Password
            <span className="field-wrap"><LockKeyhole size={16} /><input type={showPassword ? 'text' : 'password'} value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" placeholder="At least 6 characters" /><button type="button" className="password-toggle" onClick={() => setShowPassword((value) => !value)} aria-label={showPassword ? 'Hide password' : 'Show password'}>{showPassword ? <EyeOff size={16} /> : <Eye size={16} />}</button></span>
          </label>
          {error ? <p className="form-error" role="alert">{error}</p> : null}
          <div className="login-options"><label className="remember-option"><input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} /> Remember me for this browser session</label><button type="button" className="text-btn" onClick={() => setError('Password recovery needs a real identity provider and is not configured in local development.')}>Forgot password?</button></div>
          <button type="submit" className="primary-btn login-submit">Sign In</button>
        </form>
        <p className="create-account">New to PneumoVision? <button type="button" className="text-btn" onClick={() => setError('Account creation needs a real identity provider and is not configured in local development.')}>Create account</button></p>
      </section>
    </div>
  )
}
