export interface SalesRecord {
  date: string;
  sales: number | null;
  forecast: number | null;
  // Present when the backend serves prediction intervals (quantile models).
  forecast_lower?: number | null;
  forecast_upper?: number | null;
}

export interface InventoryItem {
  store_nbr: number;
  family: string;
  lead_time_days: number;
  mean_lead_demand: number;
  sigma_lead: number;
  safety_stock: number;
  reorder_point: number;
  service_level: number;
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

export type RecommendationAction = 'DISCOUNT' | 'STOCK_UP' | 'HOLD' | 'REVIEW';

export interface Recommendation {
  store_nbr: number;
  family: string;
  trailing_daily: number;
  forecast_daily: number;
  trend_pct: number;
  suggested_discount_pct: number;
  expected_demand_uplift_pct: number;
  expected_profit_change_pct: number;
  action: RecommendationAction;
  rationale: string;
}

export interface RecommendationResponse {
  count: number;
  action_counts: Record<string, number>;
  recommendations: Recommendation[];
}
