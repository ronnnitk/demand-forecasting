import React from 'react';
import { TimeRange, TIME_RANGE_OPTIONS } from '../../types';

interface TimeRangeSelectorProps {
  value: TimeRange;
  onChange: (range: TimeRange) => void;
}

const TimeRangeSelector: React.FC<TimeRangeSelectorProps> = ({ value, onChange }) => {
  return (
    <div className="inline-flex items-center bg-gray-100 rounded-lg p-1 gap-1" id="time-range-selector">
      {TIME_RANGE_OPTIONS.map((opt) => (
        <button
          key={opt.key}
          id={`btn-range-${opt.key}`}
          onClick={() => onChange(opt.key)}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
            ${
              value === opt.key
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'
            }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
};

export default TimeRangeSelector;
