import {
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
  ScanLine,
  Thermometer,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function toPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

export function ResultCard({ result, loading }) {
  const isPneumonia = result?.prediction === 'Pneumonia Detected'

  const chartData = result
    ? [
        {
          name: 'Normal',
          probability: Number((result.normal_probability * 100).toFixed(2)),
        },
        {
          name: 'Pneumonia',
          probability: Number((result.pneumonia_probability * 100).toFixed(2)),
        },
      ]
    : []

  return (
    <section className="panel result-panel" aria-labelledby="result-heading">
      <div className="panel-head">
        <h2 id="result-heading">Prediction Result</h2>
        <p>DenseNet121 binary classification output with confidence scores.</p>
      </div>

      {loading ? (
        <div className="loading-state" role="status" aria-live="polite">
          <LoaderCircle size={24} className="spin" />
          <p>Analyzing X-ray...</p>
        </div>
      ) : !result ? (
        <div className="empty-state">
          <ScanLine size={24} aria-hidden="true" />
          <p>Run an analysis to view prediction and class probabilities.</p>
        </div>
      ) : (
        <>
          <div className={`prediction-banner ${isPneumonia ? 'risk' : 'safe'}`}>
            {isPneumonia ? (
              <AlertTriangle size={18} aria-hidden="true" />
            ) : (
              <CheckCircle2 size={18} aria-hidden="true" />
            )}
            <div>
              <p className="label">Prediction</p>
              <strong>{isPneumonia ? 'PNEUMONIA DETECTED' : 'NORMAL'}</strong>
            </div>
          </div>

          <dl className="score-grid">
            <div>
              <dt>Confidence</dt>
              <dd>{toPercent(result.confidence)}</dd>
            </div>
            <div>
              <dt>Normal Probability</dt>
              <dd>{toPercent(result.normal_probability)}</dd>
            </div>
            <div>
              <dt>Pneumonia Probability</dt>
              <dd>{toPercent(result.pneumonia_probability)}</dd>
            </div>
            <div>
              <dt>Model Class</dt>
              <dd>{result.predicted_class}</dd>
            </div>
          </dl>

          <div className="chart-wrap" aria-label="Probability distribution chart">
            <div className="chart-title">
              <Thermometer size={15} aria-hidden="true" /> Probability Distribution
            </div>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(9, 32, 55, 0.16)" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                <Tooltip formatter={(value) => `${value}%`} />
                <Bar dataKey="probability" fill="#0f766e" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </section>
  )
}
