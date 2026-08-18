import { useEffect, useState } from 'react'

const STORAGE_KEY = 'pneumovision_local_dev_session_v1'

function initials(name) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join('')
    .toUpperCase()
}

export function useSessionAuth() {
  const [user, setUser] = useState(null)

  useEffect(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || 'null')
      if (saved?.email && saved?.name) setUser(saved)
    } catch {
      sessionStorage.removeItem(STORAGE_KEY)
    }
  }, [])

  const signIn = ({ email }) => {
    const name = 'Md. Mahfujur Rahman'
    const nextUser = { email, name, initials: initials(name) }
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(nextUser))
    setUser(nextUser)
  }

  const signOut = () => {
    sessionStorage.removeItem(STORAGE_KEY)
    setUser(null)
  }

  return { user, signIn, signOut }
}
