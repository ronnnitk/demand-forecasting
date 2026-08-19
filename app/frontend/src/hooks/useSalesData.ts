import { useState, useEffect, useMemo } from 'react';
import { SalesRecord, TimeRange } from '../types';

const API_BASE_URL = 'http://localhost:8000/api';
const STATIC_FALLBACK_URL = '/data/sales_with_forecast.json';

export function useSalesData() {
  const [allData, setAllData] = useState<SalesRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Lists fetched from backend
  const [stores, setStores] = useState<number[]>([]);
  const [families, setFamilies] = useState<string[]>([]);
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  // Selected parameters
  const [selectedStore, setSelectedStore] = useState<number>(2);
  const [selectedFamily, setSelectedFamily] = useState<string>('GROCERY II');
  const [forecastDays, setForecastDays] = useState<number>(15);
  const [timeRange, setTimeRange] = useState<TimeRange>('6m');

  // Metrics returned from model
  const [metrics, setMetrics] = useState({ mae: 3.9, accuracy: 93.5 });

  // 1. Fetch stores and families on mount
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const [storesRes, familiesRes] = await Promise.all([
          fetch(`${API_BASE_URL}/stores`),
          fetch(`${API_BASE_URL}/families`)
        ]);

        if (storesRes.ok && familiesRes.ok) {
          const storesData = await storesRes.json();
          const familiesData = await familiesRes.json();
          setStores(storesData);
          setFamilies(familiesData);
          setIsBackendConnected(true);
          console.log("✅ ForeSight IQ Backend Connected!");
        }
      } catch (err) {
        console.warn("⚠️ FastAPI backend is offline. Falling back to static mode.");
        setIsBackendConnected(false);
        // Fallback metadata defaults
        setStores([2]);
        setFamilies(['GROCERY II']);
      }
    };
    fetchMetadata();
  }, []);

  // 2. Fetch forecast data when parameters change
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      if (isBackendConnected) {
        try {
          const url = `${API_BASE_URL}/forecast?store_nbr=${selectedStore}&family=${encodeURIComponent(selectedFamily)}&days=${forecastDays}&history_limit=${timeRange}`;
          const res = await fetch(url);
          if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
          
          const result = await res.json();
          setAllData(result.data);
          setMetrics(result.metrics);
        } catch (err: any) {
          console.error("Error fetching forecast:", err);
          setError(err.message || 'Failed to fetch dynamic forecast');
        } finally {
          setLoading(false);
        }
      } else {
        // Static mode fallback
        try {
          const res = await fetch(STATIC_FALLBACK_URL);
          if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
          const staticData: SalesRecord[] = await res.json();
          
          // Apply history limit locally to keep chart clean
          const limitDays = timeRange === '6m' ? 180 : 365;
          const historical = staticData.filter(d => d.sales !== null);
          const forecastOnly = staticData.filter(d => d.sales === null && d.forecast !== null);
          
          // Limit historical length and crop future prediction to forecastDays
          const croppedHistorical = historical.slice(-limitDays);
          const croppedForecast = forecastOnly.slice(0, forecastDays);
          
          setAllData([...croppedHistorical, ...croppedForecast]);
          setMetrics({ mae: 3.9, accuracy: 93.5 });
        } catch (err: any) {
          setError(err.message || 'Failed to load static fallback data');
        } finally {
          setLoading(false);
        }
      }
    };

    fetchData();
  }, [isBackendConnected, selectedStore, selectedFamily, forecastDays, timeRange]);

  // Split historical vs forecast-only records
  const historicalData = useMemo(
    () => allData.filter((d) => d.sales !== null),
    [allData]
  );

  const forecastOnlyData = useMemo(
    () => allData.filter((d) => d.sales === null && d.forecast !== null),
    [allData]
  );

  // Summary stats (computed from historical data only)
  const stats = useMemo(() => {
    const hist = historicalData;
    if (hist.length === 0) return { total: 0, avg: 0, max: 0, min: 0, count: 0 };

    const sales = hist.map((d) => d.sales!);
    const total = sales.reduce((a, b) => a + b, 0);
    return {
      total: Math.round(total),
      avg: +(total / sales.length).toFixed(1),
      max: Math.max(...sales),
      min: Math.min(...sales),
      count: hist.length,
    };
  }, [historicalData]);

  return {
    allData,
    historicalData,
    forecastOnlyData,
    loading,
    error,
    stores,
    families,
    selectedStore,
    setSelectedStore,
    selectedFamily,
    setSelectedFamily,
    forecastDays,
    setForecastDays,
    timeRange,
    setTimeRange,
    metrics,
    stats,
    isBackendConnected
  };
}
