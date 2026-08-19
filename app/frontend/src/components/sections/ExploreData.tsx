import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { SalesRecord } from '../../types';

interface ExploreDataProps {
  data: SalesRecord[];
  loading: boolean;
}

const ROWS_PER_PAGE = 20;

const ExploreData: React.FC<ExploreDataProps> = ({ data, loading }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [salesFilter, setSalesFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Filter logic
  const filteredData = useMemo(() => {
    let result = [...data];

    // Text search (date)
    if (searchTerm.trim()) {
      result = result.filter((d) =>
        d.date.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Sales level filter
    if (salesFilter !== 'all') {
      result = result.filter((d) => {
        if (d.sales === null) return salesFilter === 'high'; // forecasts
        if (salesFilter === 'high') return d.sales >= 40;
        if (salesFilter === 'medium') return d.sales >= 15 && d.sales < 40;
        if (salesFilter === 'low') return d.sales < 15;
        return true;
      });
    }

    // Sort by date
    result.sort((a, b) => {
      const dateA = new Date(a.date).getTime();
      const dateB = new Date(b.date).getTime();
      return sortOrder === 'desc' ? dateB - dateA : dateA - dateB;
    });

    return result;
  }, [data, searchTerm, salesFilter, sortOrder]);

  // Pagination
  const totalPages = Math.ceil(filteredData.length / ROWS_PER_PAGE);
  const paginatedData = filteredData.slice(
    (currentPage - 1) * ROWS_PER_PAGE,
    currentPage * ROWS_PER_PAGE
  );

  // Reset page when filters change
  React.useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, salesFilter, sortOrder]);

  if (loading) {
    return (
      <section id="explore" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="bg-white rounded-xl p-6 h-[500px] skeleton" />
      </section>
    );
  }

  return (
    <section id="explore" className="py-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900">Explore Data</h2>
          <p className="text-gray-500 mt-1">
            Search, filter, and browse the complete dataset
          </p>
        </div>

        {/* Filters bar */}
        <div className="bg-white rounded-xl shadow-md border border-gray-100 p-4 mb-6">
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Search */}
            <div className="relative flex-1">
              <svg
                className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
                  clipRule="evenodd"
                />
              </svg>
              <input
                id="search-data"
                type="text"
                placeholder="Search by date (e.g. 2017-08)..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-gray-200 text-sm
                  focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                  placeholder:text-gray-400 transition-all"
              />
            </div>

            {/* Sales level dropdown */}
            <select
              id="filter-sales-level"
              value={salesFilter}
              onChange={(e) => setSalesFilter(e.target.value as any)}
              className="px-4 py-2.5 rounded-lg border border-gray-200 text-sm text-gray-700
                focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                bg-white cursor-pointer"
            >
              <option value="all">All Levels</option>
              <option value="high">High Sales (≥ 40)</option>
              <option value="medium">Medium (15–39)</option>
              <option value="low">Low Sales (&lt; 15)</option>
            </select>

            {/* Sort toggle */}
            <button
              id="btn-sort-order"
              onClick={() => setSortOrder((prev) => (prev === 'desc' ? 'asc' : 'desc'))}
              className="px-4 py-2.5 rounded-lg border border-gray-200 text-sm text-gray-700
                hover:bg-gray-50 transition-all flex items-center gap-2"
            >
              <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                <path d="M3 3a1 1 0 000 2h11a1 1 0 100-2H3zM3 7a1 1 0 000 2h7a1 1 0 100-2H3zM3 11a1 1 0 100 2h4a1 1 0 100-2H3zM15 8a1 1 0 10-2 0v5.586l-1.293-1.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L15 13.586V8z" />
              </svg>
              {sortOrder === 'desc' ? 'Newest First' : 'Oldest First'}
            </button>
          </div>

          <div className="mt-3 text-xs text-gray-400">
            Showing {paginatedData.length} of {filteredData.length} records
            {searchTerm && ` matching "${searchTerm}"`}
          </div>
        </div>

        {/* Table */}
        <div className="bg-white rounded-xl shadow-md border border-gray-100 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full" id="data-table">
              <thead>
                <tr className="border-b border-gray-100">
                  <th className="text-left text-xs font-semibold text-gray-400 uppercase tracking-wider px-6 py-4">
                    Date
                  </th>
                  <th className="text-right text-xs font-semibold text-gray-400 uppercase tracking-wider px-6 py-4">
                    Actual Sales
                  </th>
                  <th className="text-right text-xs font-semibold text-gray-400 uppercase tracking-wider px-6 py-4">
                    Forecast
                  </th>
                  <th className="text-right text-xs font-semibold text-gray-400 uppercase tracking-wider px-6 py-4">
                    Deviation
                  </th>
                  <th className="text-center text-xs font-semibold text-gray-400 uppercase tracking-wider px-6 py-4">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((row, i) => {
                  const deviation =
                    row.sales !== null && row.forecast !== null
                      ? row.sales - row.forecast
                      : null;
                  const isFuture = row.sales === null;

                  return (
                    <tr
                      key={row.date}
                      className={`border-b border-gray-50 transition-colors hover:bg-gray-50/80
                        ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/30'}`}
                    >
                      <td className="px-6 py-3.5 text-sm font-medium text-gray-900">
                        {row.date}
                      </td>
                      <td className="px-6 py-3.5 text-sm text-right font-mono">
                        {row.sales !== null ? (
                          <span className="text-blue-600 font-semibold">
                            {row.sales.toFixed(1)}
                          </span>
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="px-6 py-3.5 text-sm text-right font-mono">
                        {row.forecast !== null ? (
                          <span className="text-emerald-600 font-semibold">
                            {row.forecast.toFixed(1)}
                          </span>
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="px-6 py-3.5 text-sm text-right font-mono">
                        {deviation !== null ? (
                          <span
                            className={`font-semibold ${
                              deviation >= 0 ? 'text-emerald-600' : 'text-red-500'
                            }`}
                          >
                            {deviation >= 0 ? '+' : ''}
                            {deviation.toFixed(1)}
                          </span>
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="px-6 py-3.5 text-center">
                        {isFuture ? (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                            Predicted
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            Actual
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-6 py-4 border-t border-gray-100">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-all
                  disabled:opacity-40 disabled:cursor-not-allowed
                  text-gray-600 hover:bg-gray-100"
              >
                ← Previous
              </button>

              <div className="flex items-center gap-1">
                {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                  let page: number;
                  if (totalPages <= 5) {
                    page = i + 1;
                  } else if (currentPage <= 3) {
                    page = i + 1;
                  } else if (currentPage >= totalPages - 2) {
                    page = totalPages - 4 + i;
                  } else {
                    page = currentPage - 2 + i;
                  }
                  return (
                    <button
                      key={page}
                      onClick={() => setCurrentPage(page)}
                      className={`w-9 h-9 rounded-lg text-sm font-medium transition-all
                        ${
                          page === currentPage
                            ? 'bg-blue-600 text-white shadow-md'
                            : 'text-gray-600 hover:bg-gray-100'
                        }`}
                    >
                      {page}
                    </button>
                  );
                })}
              </div>

              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-all
                  disabled:opacity-40 disabled:cursor-not-allowed
                  text-gray-600 hover:bg-gray-100"
              >
                Next →
              </button>
            </div>
          )}
        </div>
      </motion.div>
    </section>
  );
};

export default ExploreData;
