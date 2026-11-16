/** Shapes of the Flask API payloads. Only fields the UI reads are listed; the backend may send more. */

export type Json = string | number | boolean | null | Json[] | { [k: string]: Json };

export interface PolicyInfo {
  policy?: string;
  agent?: string;
  artifact_sha256?: string;
  fallback?: { requested: string; used: string; reason?: string } | null;
}

export interface RlPolicyMeta {
  available: boolean;
  agent?: string;
  trained_on?: string;
  reason?: string;
}

export interface Meta {
  budget: number;
  tau: number;
  policies_available?: string[];
  rl_policy?: RlPolicyMeta;
}

export interface AnswerOption {
  value: 0 | 1;
  letter: string;
  label: string;
}

export interface Question {
  index: number;
  code: string;
  text: string;
  options?: AnswerOption[];
}

export interface SelectionDiag {
  applicable?: boolean;
  criterion_vacuous?: boolean;
  support_size?: number;
  n_pos?: number;
  support_is_pure?: boolean;
  ig_spread?: number;
  tie_at_max?: number;
  top_items?: { code: string; ig: number }[];
}

export interface RlDecision {
  narrative?: string[];
  action_values?: { action: string; q: number }[];
  value_units?: string;
  faithful_to_agent_internals?: boolean;
}

export interface TraceStep {
  step: number | string;
  item: string;
  value?: number | null;
  skipped?: boolean;
  belief_before: number;
  belief_after: number;
}

export interface Contribution {
  item: string;
  rank?: number;
  phi: number;
  direction?: string;
}

export interface Counterfactual {
  item?: string;
  feature?: string;
  original_value: number | string;
  flipped_value: number | string;
  new_p: number;
}

export interface SupportRecommendation {
  title?: string;
  description?: string;
  why_selected?: string;
  triggering_question_ids?: string[];
  triggering_responses?: number[];
  source_attribution?: string;
  review_status?: string;
  limitations?: string[];
}

export interface SupportReport {
  recommendations?: SupportRecommendation[];
  limitations?: string[];
  disclaimer?: string;
  unassessed_areas?: string[];
}

export interface Explanation {
  contributions?: Contribution[];
  outcome?: { tau?: number };
  attribution?: { baseline_p?: number; full_p?: number; additivity_error?: number; method?: string };
  uncertainty?: ({ available?: boolean; reason?: string } & Record<string, Json>) | null;
  limitations?: string[];
  disclaimer?: string;
}

export interface ScreeningResult {
  decision?: string;
  p_hat?: number;
  asked?: number;
  items_asked?: string[];
  stop_reason?: string;
  stop_explanation?: { narrative?: string[]; limitations?: string[] };
  policy_info?: PolicyInfo;
  trace?: TraceStep[];
  explanation?: Explanation;
  counterfactual?: Counterfactual | null;
  support_report?: SupportReport;
}

export interface SessionSummary {
  id: string;
  policy: string;
  mode: string;
  status: 'active' | 'completed' | 'expired' | string;
  asked_count?: number;
  created_at?: string;
  updated_at?: string;
}

export interface SessionDetail extends SessionSummary {
  budget?: number;
  tau?: number;
  pending_question?: Question | null;
  belief?: number;
  trace?: TraceStep[];
  decision?: RlDecision | null;
  policy_info?: PolicyInfo | null;
  result?: ScreeningResult | null;
}

/** Response of POST /api/screening/start. `result` is present for auto mode. */
export interface StartResponse {
  session_id: string;
  mode: string;
  policy: string;
  budget: number;
  tau: number;
  policy_info?: PolicyInfo;
  question?: Question | null;
  belief?: number;
  asked?: number;
  selection?: SelectionDiag | null;
  decision?: RlDecision | null;
  result?: ScreeningResult;
}

export interface AnswerResponse {
  finished?: boolean;
  result?: ScreeningResult;
  next?: Question | null;
  belief_after?: number;
  asked?: number;
  selection?: SelectionDiag | null;
  decision?: RlDecision | null;
  trace_step: TraceStep;
}

export interface Observation {
  id: string;
  context: string;
  occurred_at?: string | null;
  created_at?: string;
  what_happened: string;
  frequency?: string | null;
  difficulty?: string | null;
  what_helped?: string | null;
  consistency?: string | null;
  insufficient_info?: boolean;
  notes?: string | null;
}

export type ObservationInput = Omit<Observation, 'id' | 'created_at'>;

export interface Strategy {
  id: string;
  title: string;
  domain?: string | null;
  triggered_observation_id?: string | null;
  why_useful?: string | null;
  how_to_try?: string | null;
  how_to_record?: string | null;
  status?: string;
  outcome?: string | null;
  notes?: string | null;
  source?: string;
}

export interface Suggestion {
  title: string;
  domain?: string;
  triggered_observation_id?: string;
  matched_terms?: { term: string; field: string }[];
  why_useful?: string;
  how_to_try?: string;
  how_to_record?: string;
}

export interface TimelineEvent {
  id?: string;
  kind: string;
  title: string;
  context?: string | null;
  summary?: string | null;
  notes?: string | null;
  event_time?: string;
  created_at?: string;
}

export interface ConsentRecord {
  scope: string;
  granted: boolean;
  details?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface ResearchDashboard {
  artifacts: Record<string, Record<string, unknown> | null>;
  tables?: Record<string, Record<string, unknown>[] | null>;
  limitations?: string[];
}

export interface RlScenario {
  id: string;
  name: string;
}

export interface RlPolicyStatus {
  name: string;
  kind: string;
  status: string;
  reason?: string;
}

export interface RlScenarioMeta {
  scenarios: RlScenario[];
  policies: RlPolicyStatus[];
  budgets: number[];
}

export interface RlReplaySummary {
  questions_asked?: number;
  final_p_hat?: number;
  decision?: string;
  stop_reason?: string;
}

export interface RlReplay {
  ok?: boolean;
  error?: string;
  scenario?: string;
  policy?: string;
  budget?: number;
  synthetic?: boolean;
  steps?: Record<string, unknown>[];
  summary?: RlReplaySummary;
  trace?: { item: string }[];
}

export interface RlCompare {
  scenario: string;
  budget: number;
  results: (RlReplay & { policy?: string })[];
}
