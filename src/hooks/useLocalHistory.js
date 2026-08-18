import { useEffect, useState } from 'react'

const STORAGE_KEY = 'pneumovision_history_v1'
const MAX_HISTORY_ITEMS = 20

export function useLocalHistory() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
      if (Array.isArray(parsed)) {
        setHistory(parsed)
      }
    } catch {
      setHistory([])
    }
  }, [])

  const persist = (items) => {
    // A history thumbnail must never make a successful prediction crash the UI.
    // If browser storage is full, retain the newest entries that fit.
    for (let count = items.length; count > 0; count -= 1) {
      const candidate = items.slice(0, count)
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(candidate))
        return candidate
      } catch (error) {
        if (error?.name !== 'QuotaExceededError') {
          console.warn('Prediction history could not be saved.', error)
          return candidate
        }
      }
    }

    console.warn('Prediction history storage is full; the latest analysis was not saved.')
    return []
  }

  const addEntry = (entry) => {
    setHistory((current) => {
      const updated = [entry, ...current].slice(0, MAX_HISTORY_ITEMS)
      return persist(updated)
    })
  }

  const clearHistory = () => {
    setHistory(persist([]))
  }

  return {
    history,
    addEntry,
    clearHistory,
  }
}
