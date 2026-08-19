import React from 'react';

interface CardProps {
  title?: string;
  value?: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  accent?: 'primary' | 'accent' | 'danger' | 'secondary';
  children?: React.ReactNode;
  className?: string;
}

const accentColors: Record<string, string> = {
  primary: '#2563eb',
  accent: '#10b981',
  danger: '#ef4444',
  secondary: '#64748b',
};

const Card: React.FC<CardProps> = ({
  title,
  value,
  subtitle,
  icon,
  accent = 'primary',
  children,
  className = '',
}) => {
  return (
    <div
      className={`bg-white rounded-xl shadow-md border border-gray-100 p-6
        transition-all duration-300 hover:shadow-xl hover:-translate-y-1 ${className}`}
      style={{ borderTop: `3px solid ${accentColors[accent]}` }}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          {title && (
            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">
              {title}
            </h3>
          )}
          {value !== undefined && (
            <div className="text-3xl font-bold text-gray-900 mt-1">
              {typeof value === 'number' ? value.toLocaleString() : value}
            </div>
          )}
          {subtitle && (
            <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
        {icon && (
          <div
            className="flex items-center justify-center w-10 h-10 rounded-lg"
            style={{ backgroundColor: `${accentColors[accent]}15` }}
          >
            <span style={{ color: accentColors[accent] }}>{icon}</span>
          </div>
        )}
      </div>
      {children && <div className="mt-4 text-sm text-gray-600">{children}</div>}
    </div>
  );
};

export default Card;