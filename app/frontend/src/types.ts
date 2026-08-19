export interface SalesRecord {
  date: string;
  sales: number | null;
  forecast: number | null;
}

export type TimeRange = '6m' | '1y';

export interface TimeRangeOption {
  key: TimeRange;
  label: string;
  days: number;
}

export const TIME_RANGE_OPTIONS: TimeRangeOption[] = [
  { key: '6m', label: '6 Months', days: 180 },
  { key: '1y', label: '1 Year', days: 365 },
];
