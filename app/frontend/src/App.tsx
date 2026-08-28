import React from 'react';
import Navbar from './components/layout/Navbar';
import HeroSection from './components/sections/Introduction';
import DataOverview from './components/sections/DataOverview';
import TrendsAnalysis from './components/sections/TrendsAnalysis';
import ForecastSection from './components/sections/ForecastSection';
import Recommendations from './components/sections/Recommendations';
import ExploreData from './components/sections/ExploreData';
import Insights from './components/sections/Insights';
import { useSalesData } from './hooks/useSalesData';
import './styles/globals.css';

function App() {
  const {
    allData,
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
  } = useSalesData();

  if (error && !allData.length) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-500" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">Failed to Load Data</h2>
          <p className="text-gray-500 mb-4">{error}</p>
          <p className="text-sm text-gray-400">
            Make sure the FastAPI backend is running on <code className="bg-gray-100 px-1 rounded">localhost:8000</code> or that static files are available.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="relative">
        
        {/* Dynamic loading overlay during API calls */}
        {loading && allData.length > 0 && (
          <div className="fixed inset-0 bg-white/40 backdrop-blur-[2px] z-50 flex items-center justify-center pointer-events-none">
            <div className="bg-white/80 backdrop-blur-md rounded-2xl shadow-xl border border-gray-100 p-6 flex flex-col items-center gap-4">
              <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-sm font-semibold text-gray-700">Updating Forecast Model...</p>
            </div>
          </div>
        )}

        <HeroSection />

        {/* --- PREMIUM CONTROL PANEL --- */}
        <section className="relative z-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-10 mb-8">
          <div className="bg-white/80 backdrop-blur-xl border border-gray-100 rounded-2xl shadow-xl p-6 sm:p-8">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 mb-6">
              <div>
                <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  <span>🎛️ AI Forecast Control Center</span>
                  <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold
                    ${isBackendConnected 
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                      : 'bg-amber-50 text-amber-700 border border-amber-200'
                    }`}
                  >
                    <span className={`w-2 h-2 rounded-full ${isBackendConnected ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
                    {isBackendConnected ? 'ML Backend Connected' : 'Static Demo Mode'}
                  </span>
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  Dynamically control and train machine learning models for any store and product combination.
                </p>
              </div>

              {/* Status Indicator */}
              <div className="flex items-center gap-2 self-start md:self-auto">
                <span className="text-xs text-gray-400 bg-gray-100 px-3 py-1.5 rounded-lg border border-gray-200">
                  Model: <strong>Random Forest Regressor</strong>
                </span>
              </div>
            </div>

            <div className="grid gap-6 md:grid-cols-4">
              
              {/* Store Selection */}
              <div className="flex flex-col">
                <label className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                  🏪 Select Store Location
                </label>
                <select
                  value={selectedStore}
                  onChange={(e) => setSelectedStore(Number(e.target.value))}
                  className="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm font-semibold text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all cursor-pointer"
                >
                  {stores.map((s) => (
                    <option key={s} value={s}>
                      Store Number {s}
                    </option>
                  ))}
                </select>
              </div>

              {/* Product Family Selection */}
              <div className="flex flex-col">
                <label className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                  🏷️ Product Family Category
                </label>
                <select
                  value={selectedFamily}
                  onChange={(e) => setSelectedFamily(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm font-semibold text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all cursor-pointer"
                >
                  {families.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </div>

              {/* Prediction Horizon Slider */}
              <div className="flex flex-col md:col-span-2">
                <div className="flex justify-between items-center mb-2">
                  <label className="text-xs font-bold text-gray-500 uppercase tracking-wider">
                    ⏱️ Prediction Horizon (Dynamic)
                  </label>
                  <span className="text-xs font-bold bg-blue-50 text-blue-700 px-2.5 py-1 rounded-md border border-blue-100">
                    <strong>{forecastDays} Days</strong> forecast
                  </span>
                </div>
                <div className="flex items-center gap-4 bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
                  <span className="text-xs font-bold text-gray-400">1d</span>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    value={forecastDays}
                    onChange={(e) => setForecastDays(Number(e.target.value))}
                    className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 focus:outline-none"
                  />
                  <span className="text-xs font-bold text-gray-400">15d</span>
                </div>
              </div>

            </div>
          </div>
        </section>

        {/* Dashboard Sections */}
        <DataOverview data={allData} stats={stats} loading={loading} />
        
        <TrendsAnalysis
          data={allData}
          timeRange={timeRange}
          onTimeRangeChange={setTimeRange}
          loading={loading}
        />
        
        <ForecastSection
          data={allData}
          loading={loading}
          selectedStore={selectedStore}
          selectedFamily={selectedFamily}
          forecastDays={forecastDays}
          metrics={metrics}
        />

        <Recommendations
          selectedStore={selectedStore}
          isBackendConnected={isBackendConnected}
        />

        <ExploreData data={allData} loading={loading} />
        <Insights />
      </main>

      {/* Footer */}
      <footer className="py-8 px-4 text-center border-t border-gray-200 bg-white">
        <p className="text-sm text-gray-400">
          ForeSight IQ Demand Forecasting Dashboard • Built with React + FastAPI + Scikit-Learn (Random Forest)
        </p>
      </footer>
    </div>
  );
}

export default App;