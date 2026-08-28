import React from 'react';
import { useRecommendations } from '../../hooks/useRecommendations';
import { Recommendation, RecommendationAction } from '../../types';

const ACTION_STYLE: Record<RecommendationAction, { badge: string; label: string; icon: string }> = {
  DISCOUNT: { badge: 'bg-indigo-50 text-indigo-700 border-indigo-200', label: 'Discount', icon: '🏷️' },
  STOCK_UP: { badge: 'bg-emerald-50 text-emerald-700 border-emerald-200', label: 'Stock Up', icon: '📦' },
  HOLD: { badge: 'bg-gray-100 text-gray-600 border-gray-200', label: 'Hold', icon: '⏸️' },
  REVIEW: { badge: 'bg-amber-50 text-amber-700 border-amber-200', label: 'Review', icon: '🔍' },
};

const trendBadge = (pct: number) => {
  const up = pct >= 0;
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-semibold ${up ? 'text-emerald-600' : 'text-rose-600'}`}>
      {up ? '▲' : '▼'} {(pct * 100).toFixed(0)}%
    </span>
  );
};

interface Props {
  selectedStore: number;
  isBackendConnected: boolean;
}

const Recommendations: React.FC<Props> = ({ selectedStore, isBackendConnected }) => {
  const { recommendations, actionCounts, loading, error } = useRecommendations(selectedStore, isBackendConnected);

  const order: RecommendationAction[] = ['DISCOUNT', 'STOCK_UP', 'HOLD', 'REVIEW'];

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 my-12">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <span>🎯 Prescriptive Actions</span>
          <span className="text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-100 px-2.5 py-0.5 rounded-full">
            Store {selectedStore}
          </span>
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          What to do about the forecast — margin-aware discounts, festival stock-ups, and items to hold or review.
        </p>
      </div>

      {/* Action summary chips */}
      <div className="flex flex-wrap gap-3 mb-6">
        {order.map((a) => (
          <div key={a} className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${ACTION_STYLE[a].badge}`}>
            <span>{ACTION_STYLE[a].icon}</span>
            <span className="text-sm font-bold">{actionCounts[a] || 0}</span>
            <span className="text-xs font-medium">{ACTION_STYLE[a].label}</span>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-gray-400">
            <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            Loading recommendations…
          </div>
        ) : error ? (
          <div className="p-8 text-center text-gray-500">
            <p className="font-semibold text-gray-700 mb-1">No recommendations available</p>
            <p className="text-sm">
              Generate them with <code className="bg-gray-100 px-1 rounded">python -m src.recommend</code>
            </p>
          </div>
        ) : recommendations.length === 0 ? (
          <div className="p-8 text-center text-gray-400">No recommendations for this store.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 text-left text-xs font-bold text-gray-500 uppercase tracking-wider">
                  <th className="px-4 py-3">Product Family</th>
                  <th className="px-4 py-3">Action</th>
                  <th className="px-4 py-3 text-right">Trailing → Forecast</th>
                  <th className="px-4 py-3 text-right">Trend</th>
                  <th className="px-4 py-3 text-right">Suggested Offer</th>
                  <th className="px-4 py-3 text-right">Δ Profit</th>
                  <th className="px-4 py-3">Rationale</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {recommendations.map((r: Recommendation, i) => (
                  <tr key={`${r.store_nbr}-${r.family}-${i}`} className="hover:bg-gray-50/60">
                    <td className="px-4 py-3 font-semibold text-gray-800">{r.family}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-xs font-bold ${ACTION_STYLE[r.action].badge}`}>
                        {ACTION_STYLE[r.action].icon} {ACTION_STYLE[r.action].label}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right text-gray-600 tabular-nums">
                      {r.trailing_daily.toLocaleString()} → <strong className="text-gray-900">{r.forecast_daily.toLocaleString()}</strong>
                    </td>
                    <td className="px-4 py-3 text-right">{trendBadge(r.trend_pct)}</td>
                    <td className="px-4 py-3 text-right tabular-nums">
                      {r.suggested_discount_pct > 0 ? (
                        <span className="font-bold text-indigo-600">{r.suggested_discount_pct}% off</span>
                      ) : (
                        <span className="text-gray-300">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">
                      {r.expected_profit_change_pct > 0 ? (
                        <span className="font-semibold text-emerald-600">+{r.expected_profit_change_pct}%</span>
                      ) : (
                        <span className="text-gray-300">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-gray-500 max-w-md">{r.rationale}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
};

export default Recommendations;
