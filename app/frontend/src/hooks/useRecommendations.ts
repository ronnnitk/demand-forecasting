import { useState, useEffect } from 'react';
import { Recommendation, RecommendationResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000/api';
const STATIC_FALLBACK_URL = '/data/recommendations.json';

/**
 * Fetches prescriptive actions (offers / discounts / stock-ups) for a store.
 * Mirrors useSalesData: try the FastAPI backend, fall back to a static JSON
 * snapshot so the dashboard still demonstrates the feature offline.
 */
export function useRecommendations(storeNbr: number, isBackendConnected: boolean) {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [actionCounts, setActionCounts] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRecommendations = async () => {
      setLoading(true);
      setError(null);
      try {
        let data: RecommendationResponse;
        if (isBackendConnected) {
          const res = await fetch(`${API_BASE_URL}/recommendations?store_nbr=${storeNbr}&limit=200`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          data = await res.json();
        } else {
          const res = await fetch(STATIC_FALLBACK_URL);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const all: RecommendationResponse = await res.json();
          const filtered = all.recommendations.filter((r) => r.store_nbr === storeNbr);
          data = {
            count: filtered.length,
            action_counts: filtered.reduce<Record<string, number>>((acc, r) => {
              acc[r.action] = (acc[r.action] || 0) + 1;
              return acc;
            }, {}),
            recommendations: filtered,
          };
        }
        setRecommendations(data.recommendations);
        setActionCounts(data.action_counts);
      } catch (err: any) {
        setError(err.message || 'Failed to load recommendations');
        setRecommendations([]);
        setActionCounts({});
      } finally {
        setLoading(false);
      }
    };
    fetchRecommendations();
  }, [storeNbr, isBackendConnected]);

  return { recommendations, actionCounts, loading, error };
}
