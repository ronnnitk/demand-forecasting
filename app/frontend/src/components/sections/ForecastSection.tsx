import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import Card from '../ui/Card';
import { SalesRecord } from '../../types';

interface ForecastSectionProps {
  data: SalesRecord[];
  loading: boolean;
  selectedStore: number;
  selectedFamily: string;
  forecastDays: number;
  metrics: { mae: number; accuracy: number };
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;

  const isFuture = payload.some(
    (p: any) => p.name === 'Actual Sales' && (p.value === null || p.value === undefined || p.value === 0)
  );

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-xl p-4 min-w-[200px]">
      <p className="text-xs font-medium text-gray-400 mb-2">
        {label} {isFuture && <span className="text-emerald-500 font-semibold">(Predicted)</span>}
      </p>
      {payload.map((entry: any, i: number) => {
        if (entry.value === null || entry.value === undefined) return null;
        return (
          <div key={i} className="flex items-center gap-2 mb-1">
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
            <span className="text-sm text-gray-600">{entry.name}:</span>
            <span className="text-sm font-semibold text-gray-900">
              {Number(entry.value).toFixed(1)} units
            </span>
          </div>
        );
      })}
    </div>
  );
};

const ForecastSection: React.FC<ForecastSectionProps> = ({ 
  data, 
  loading,
  selectedStore,
  selectedFamily,
  forecastDays,
  metrics
}) => {
  // Use last 90 days of historical data + all forecast data for the chart
  const chartData = useMemo(() => {
    const historical = data.filter((d) => d.sales !== null);
    const forecastOnly = data.filter((d) => d.sales === null && d.forecast !== null);

    // Take last 90 days of historical
    const recentHistorical = historical.slice(-90);

    // Combine for the chart
    const combined = [
      ...recentHistorical.map((d) => ({
        date: d.date,
        actual: d.sales,
        forecast: d.forecast,
      })),
      ...forecastOnly.map((d) => ({
        date: d.date,
        actual: null as number | null,
        forecast: d.forecast,
      })),
    ];

    return combined;
  }, [data]);

  // Find the boundary date (last actual data point)
  const boundaryDate = useMemo(() => {
    const historical = data.filter((d) => d.sales !== null);
    return historical.length > 0 ? historical[historical.length - 1].date : '';
  }, [data]);

  if (loading && !data.length) {
    return (
      <section id="forecast" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="bg-white rounded-xl p-6 h-[500px] skeleton animate-pulse" />
      </section>
    );
  }

  return (
    <section id="forecast" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <div className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold text-gray-900">Dynamic AI Forecast</h2>
            <p className="text-gray-500 mt-1">
              Active prediction showing Store {selectedStore} • {selectedFamily}
            </p>
          </div>
          <span className="text-sm font-semibold bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-lg border border-emerald-100 self-start md:self-auto">
            📈 Horizon: {forecastDays} Days Predictive Trend
          </span>
        </div>

        {/* Metrics row */}
        <div className="grid gap-4 md:grid-cols-3 mb-8">
          <Card
            title="Model Accuracy"
            value={`${metrics.accuracy}%`}
            subtitle="validation score (R-squared/fit)"
            accent="accent"
          />
          <Card
            title="Mean Absolute Error"
            value={metrics.mae}
            subtitle="avg units variance"
            accent="primary"
          />
          <Card
            title="AI Model Engine"
            value="Random Forest"
            subtitle="Scikit-Learn Regression Ensemble"
            accent="secondary"
          />
        </div>

        {/* Chart */}
        <div className="bg-white rounded-xl shadow-md border border-gray-100 p-4 sm:p-6">
          <div className="flex flex-wrap items-center gap-4 mb-6">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-blue-600"></span>
              <span className="text-sm font-medium text-gray-600">Actual Sales</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
              <span className="text-sm font-medium text-gray-600">ML Model Fit / Forecast</span>
            </div>
            {boundaryDate && (
              <div className="flex items-center gap-2 ml-auto">
                <span className="text-xs font-semibold text-gray-400 bg-gray-100 px-2.5 py-1 rounded-md border border-gray-200">
                  Last Actual Date: {boundaryDate}
                </span>
              </div>
            )}
          </div>

          <ResponsiveContainer width="100%" height={420}>
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: '#94a3b8' }}
                tickLine={false}
                axisLine={{ stroke: '#e2e8f0' }}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#94a3b8' }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                verticalAlign="top"
                height={36}
                iconType="circle"
                formatter={(value: string) => (
                  <span className="text-sm font-medium text-gray-600">{value}</span>
                )}
              />
              {boundaryDate && (
                <ReferenceLine
                  x={boundaryDate}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  label={{
                    value: '🔮 Forecast Period',
                    position: 'insideTopRight',
                    fill: '#64748b',
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                />
              )}
              {/* Actual sales as bars */}
              <Bar
                dataKey="actual"
                fill="#2563eb"
                fillOpacity={0.6}
                radius={[2, 2, 0, 0]}
                name="Actual Sales"
              />
              {/* Forecast/Historical model fit as a line */}
              <Line
                type="monotone"
                dataKey="forecast"
                stroke="#10b981"
                strokeWidth={2.5}
                dot={false}
                name="Model Prediction"
                strokeDasharray="5 3"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Methodology info */}
        <div className="mt-6 bg-blue-50 rounded-xl p-5 border border-blue-100">
          <h3 className="text-sm font-bold text-blue-800 mb-2">
            🧠 ForeSight IQ Machine Learning Pipeline
          </h3>
          <p className="text-sm text-blue-700 leading-relaxed">
            The system dynamically trains a <strong>Random Forest Regressor</strong> model on the FastAPI server 
            for the selected combination of <strong>Store {selectedStore}</strong> and product family <strong>{selectedFamily}</strong>. 
            The machine learning model processes key time-series features extracted from the historical dataset:
          </p>
          <ul className="list-disc list-inside text-sm text-blue-700 mt-2 space-y-1 pl-4">
            <li><strong>Lag Features:</strong> Past sales patterns from 1 day (`lag_1`), 7 days (`lag_7`), and 14 days (`lag_14`) ago.</li>
            <li><strong>Rolling Mean:</strong> A 7-day moving average (`rolling_mean_7`) to capture mid-term trends.</li>
            <li><strong>Time Features:</strong> Temporal attributes including day of the week, month, and day of the month.</li>
          </ul>
          <p className="text-sm text-blue-700 leading-relaxed mt-2">
            A multi-step recursive rolling forecast is generated to predict sales trends for exactly <strong>{forecastDays} days</strong> into the future, enforcing no negative sales.
          </p>
        </div>
      </motion.div>
    </section>
  );
};

export default ForecastSection;