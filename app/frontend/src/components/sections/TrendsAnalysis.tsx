import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import TimeRangeSelector from '../ui/TimeRangeSelector';
import { SalesRecord, TimeRange } from '../../types';

interface TrendsAnalysisProps {
  data: SalesRecord[];
  timeRange: TimeRange;
  onTimeRangeChange: (range: TimeRange) => void;
  loading: boolean;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-xl p-4 min-w-[180px]">
      <p className="text-xs font-medium text-gray-400 mb-2">{label}</p>
      {payload.map((entry: any, i: number) => (
        <div key={i} className="flex items-center gap-2 mb-1">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
          <span className="text-sm text-gray-600 capitalize">{entry.name}:</span>
          <span className="text-sm font-semibold text-gray-900">
            {entry.value !== null && entry.value !== undefined
              ? Number(entry.value).toFixed(1)
              : '—'}
          </span>
        </div>
      ))}
    </div>
  );
};

const TrendsAnalysis: React.FC<TrendsAnalysisProps> = ({
  data,
  timeRange,
  onTimeRangeChange,
  loading,
}) => {
  // Downsample data for chart performance (max ~300 points)
  const chartData = useMemo(() => {
    if (data.length <= 300) return data;
    const step = Math.ceil(data.length / 300);
    return data.filter((_, i) => i % step === 0 || i === data.length - 1);
  }, [data]);

  if (loading) {
    return (
      <section id="trends" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="bg-white rounded-xl p-6 h-[500px] skeleton" />
      </section>
    );
  }

  return (
    <section id="trends" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
          <div>
            <h2 className="text-3xl font-bold text-gray-900">Sales Trends</h2>
            <p className="text-gray-500 mt-1">
              Historical sales with Random Forest model fit overlay
            </p>
          </div>
          <TimeRangeSelector value={timeRange} onChange={onTimeRangeChange} />
        </div>

        <div className="bg-white rounded-xl shadow-md border border-gray-100 p-4 sm:p-6">
          <ResponsiveContainer width="100%" height={420}>
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="salesGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2563eb" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
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
                  <span className="text-sm text-gray-600 capitalize">{value}</span>
                )}
              />
              <Area
                type="monotone"
                dataKey="sales"
                stroke="#2563eb"
                strokeWidth={2}
                fill="url(#salesGradient)"
                dot={false}
                name="Actual Sales"
                connectNulls={false}
              />
              <Area
                type="monotone"
                dataKey="forecast"
                stroke="#10b981"
                strokeWidth={2}
                fill="url(#forecastGradient)"
                dot={false}
                strokeDasharray="6 3"
                name="Model Prediction"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </section>
  );
};

export default TrendsAnalysis;