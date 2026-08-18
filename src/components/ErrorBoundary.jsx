import { Component } from 'react'

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // Keep unexpected render failures visible during development without
    // allowing them to replace the entire dashboard with a white screen.
    console.error('PneumoVision UI render error:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <section className="panel error-fallback" role="alert">
          <h2>{this.props.title || 'This section is unavailable'}</h2>
          <p>Please try the analysis again. The rest of the application remains available.</p>
        </section>
      )
    }

    return this.props.children
  }
}
