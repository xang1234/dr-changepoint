// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

#[cfg(feature = "rayon")]
use cpd_core::ReproMode;
use cpd_core::{
    BudgetStatus, CpdError, Diagnostics, ExecutionContext, OfflineChangePointResult,
    OfflineDetector, Penalty, Stopping, TimeSeriesView, ValidatedConstraints,
    check_missing_compatibility, penalty_value, validate_constraints, validate_stopping,
};
use cpd_costs::CostModel;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::borrow::Cow;
use std::time::Instant;

const DEFAULT_CANCEL_CHECK_EVERY: usize = 1000;
const DEFAULT_SEED: u64 = 0;

/// Interval strategy for [`Wbs`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WbsIntervalStrategy {
    /// Draw random intervals from the full series using a deterministic seed.
    #[default]
    Random,
    /// Deterministic dyadic grid over multiple scales.
    DeterministicGrid,
    /// Random intervals stratified across dyadic scales.
    Stratified,
}

/// Configuration for [`Wbs`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct WbsConfig {
    pub stopping: Stopping,
    pub params_per_segment: usize,
    pub num_intervals: Option<usize>,
    pub interval_strategy: WbsIntervalStrategy,
    pub seed: u64,
    pub cancel_check_every: usize,
}

impl Default for WbsConfig {
    fn default() -> Self {
        Self {
            stopping: Stopping::Penalized(Penalty::BIC),
            params_per_segment: 2,
            num_intervals: None,
            interval_strategy: WbsIntervalStrategy::Random,
            seed: DEFAULT_SEED,
            cancel_check_every: DEFAULT_CANCEL_CHECK_EVERY,
        }
    }
}

impl WbsConfig {
    fn validate(&self) -> Result<(), CpdError> {
        validate_stopping(&self.stopping)?;

        if self.params_per_segment == 0 {
            return Err(CpdError::invalid_input(
                "WbsConfig.params_per_segment must be >= 1; got 0",
            ));
        }

        if matches!(self.num_intervals, Some(0)) {
            return Err(CpdError::invalid_input(
                "WbsConfig.num_intervals must be >= 1 when provided; got 0",
            ));
        }

        Ok(())
    }

    fn normalized_cancel_check_every(&self) -> usize {
        self.cancel_check_every.max(1)
    }
}

/// Wild Binary Segmentation offline detector.
#[derive(Debug)]
pub struct Wbs<C: CostModel> {
    cost_model: C,
    config: WbsConfig,
}

impl<C: CostModel> Wbs<C> {
    pub fn new(cost_model: C, config: WbsConfig) -> Result<Self, CpdError> {
        config.validate()?;
        Ok(Self { cost_model, config })
    }

    pub fn cost_model(&self) -> &C {
        &self.cost_model
    }

    pub fn config(&self) -> &WbsConfig {
        &self.config
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct RuntimeStats {
    cost_evals: usize,
    candidates_considered: usize,
    intervals_considered: usize,
    soft_budget_exceeded: bool,
    used_parallel: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Interval {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    start: usize,
    end: usize,
    depth: usize,
}

#[derive(Clone, Copy, Debug)]
struct SegmentCandidate {
    segment: Segment,
    split: usize,
    gain: f64,
}

#[derive(Clone, Copy, Debug)]
struct IntervalScore {
    split: usize,
    gain: f64,
    interval_start: usize,
    #[cfg(feature = "rayon")]
    cost_evals: usize,
    #[cfg(feature = "rayon")]
    candidates_considered: usize,
}

#[derive(Clone, Copy, Debug)]
struct SplitWindow {
    start_idx: usize,
    end_idx: usize,
}

#[derive(Clone, Copy, Debug)]
struct StableRng {
    state: u64,
}

impl StableRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e3779b97f4a7c15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn gen_range(&mut self, upper_exclusive: usize) -> Result<usize, CpdError> {
        if upper_exclusive == 0 {
            return Err(CpdError::invalid_input(
                "StableRng.gen_range requires upper_exclusive >= 1; got 0",
            ));
        }

        let value = self.next_u64();
        let modulus = u64::try_from(upper_exclusive)
            .map_err(|_| CpdError::resource_limit("rng upper_exclusive conversion overflow"))?;
        let sampled = value % modulus;
        usize::try_from(sampled)
            .map_err(|_| CpdError::resource_limit("rng sampled index conversion overflow"))
    }
}

fn checked_counter_increment(counter: &mut usize, name: &str) -> Result<(), CpdError> {
    *counter = counter
        .checked_add(1)
        .ok_or_else(|| CpdError::resource_limit(format!("{name} counter overflow")))?;
    Ok(())
}

#[cfg(feature = "rayon")]
fn checked_counter_add(counter: &mut usize, delta: usize, name: &str) -> Result<(), CpdError> {
    *counter = counter
        .checked_add(delta)
        .ok_or_else(|| CpdError::resource_limit(format!("{name} counter overflow")))?;
    Ok(())
}

fn resolve_penalty_beta(
    penalty: &Penalty,
    n: usize,
    d: usize,
    params_per_segment: usize,
) -> Result<f64, CpdError> {
    let beta = penalty_value(penalty, n, d, params_per_segment)?;
    if !beta.is_finite() || beta <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "resolved penalty must be finite and > 0.0; got beta={beta}"
        )));
    }
    Ok(beta)
}

fn evaluate_segment_cost<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    start: usize,
    end: usize,
    ctx: &ExecutionContext<'_>,
    runtime: &mut RuntimeStats,
) -> Result<f64, CpdError> {
    checked_counter_increment(&mut runtime.cost_evals, "cost_evals")?;

    match ctx.check_cost_eval_budget(runtime.cost_evals)? {
        BudgetStatus::WithinBudget => {}
        BudgetStatus::ExceededSoftDegrade => {
            runtime.soft_budget_exceeded = true;
        }
    }

    let segment_cost = model.segment_cost(cache, start, end);
    if !segment_cost.is_finite() {
        return Err(CpdError::numerical_issue(format!(
            "non-finite segment cost at [{start}, {end}): {segment_cost}"
        )));
    }
    Ok(segment_cost)
}

fn check_runtime_controls(
    iteration: usize,
    cancel_check_every: usize,
    ctx: &ExecutionContext<'_>,
    started_at: Instant,
    runtime: &mut RuntimeStats,
) -> Result<(), CpdError> {
    if iteration.is_multiple_of(cancel_check_every) {
        ctx.check_cancelled_every(iteration, 1)?;
        match ctx.check_time_budget(started_at)? {
            BudgetStatus::WithinBudget => {}
            BudgetStatus::ExceededSoftDegrade => {
                runtime.soft_budget_exceeded = true;
            }
        }
    }

    Ok(())
}

fn candidate_window(candidates: &[usize], lower: usize, upper: usize) -> Option<(usize, usize)> {
    if lower > upper {
        return None;
    }

    let start_idx = candidates.partition_point(|&split| split < lower);
    let end_idx = candidates.partition_point(|&split| split <= upper);
    if start_idx >= end_idx {
        return None;
    }

    Some((start_idx, end_idx))
}

fn segment_can_split(segment: Segment, validated: &ValidatedConstraints) -> bool {
    if segment.end <= segment.start {
        return false;
    }

    if let Some(max_depth) = validated.max_depth
        && segment.depth >= max_depth
    {
        return false;
    }

    segment.end - segment.start >= validated.min_segment_len.saturating_mul(2)
}

fn split_window_for_interval(
    interval: Interval,
    candidates: &[usize],
    min_segment_len: usize,
) -> Result<Option<SplitWindow>, CpdError> {
    if interval.end <= interval.start {
        return Ok(None);
    }

    let lower = interval
        .start
        .checked_add(min_segment_len)
        .ok_or_else(|| CpdError::resource_limit("interval lower bound overflow"))?;
    let upper = interval.end.saturating_sub(min_segment_len);

    let Some((start_idx, end_idx)) = candidate_window(candidates, lower, upper) else {
        return Ok(None);
    };

    Ok(Some(SplitWindow { start_idx, end_idx }))
}

fn interval_feasible(
    interval: Interval,
    candidates: &[usize],
    min_segment_len: usize,
) -> Result<bool, CpdError> {
    if interval.end <= interval.start {
        return Ok(false);
    }

    let len = interval.end - interval.start;
    if len < min_segment_len.saturating_mul(2) {
        return Ok(false);
    }

    Ok(split_window_for_interval(interval, candidates, min_segment_len)?.is_some())
}

fn segment_contains_interval(segment: Segment, interval: Interval) -> bool {
    interval.start >= segment.start && interval.end <= segment.end
}

fn interval_score_better(candidate: IntervalScore, current: IntervalScore) -> bool {
    let better_gain = candidate.gain > current.gain;
    let tie_gain = candidate.gain == current.gain;
    let better_split = candidate.split < current.split;
    let tie_split = candidate.split == current.split;
    let better_start = candidate.interval_start < current.interval_start;

    better_gain || (tie_gain && (better_split || (tie_split && better_start)))
}

fn segment_candidate_better(candidate: SegmentCandidate, current: SegmentCandidate) -> bool {
    let better_gain = candidate.gain > current.gain;
    let tie_gain = candidate.gain == current.gain;
    let better_split = candidate.split < current.split;
    let tie_split = candidate.split == current.split;
    let better_start = candidate.segment.start < current.segment.start;

    better_gain || (tie_gain && (better_split || (tie_split && better_start)))
}

fn resolve_num_intervals(num_intervals: Option<usize>, n: usize) -> Result<usize, CpdError> {
    if let Some(explicit) = num_intervals {
        return Ok(explicit);
    }

    let adaptive = (5.0 * (n as f64).sqrt()).ceil();
    if !adaptive.is_finite() || adaptive <= 0.0 {
        return Err(CpdError::invalid_input(format!(
            "adaptive interval count must be finite and > 0.0; got {adaptive}"
        )));
    }

    let adaptive_usize = usize::try_from(adaptive as u128)
        .map_err(|_| CpdError::resource_limit("adaptive interval count overflow"))?;
    Ok(adaptive_usize.max(100))
}

fn dyadic_lengths(n: usize, min_segment_len: usize) -> Vec<usize> {
    let min_len = min_segment_len.saturating_mul(2);
    if n < min_len {
        return vec![];
    }

    let mut lengths = vec![];
    let mut current = n;
    loop {
        if current < min_len {
            break;
        }
        lengths.push(current);
        if current == min_len {
            break;
        }
        let next = (current / 2).max(min_len);
        if next == current {
            break;
        }
        current = next;
    }

    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

fn dedup_sorted_intervals(mut intervals: Vec<Interval>) -> Vec<Interval> {
    intervals.sort_unstable();
    intervals.dedup();
    intervals
}

fn downsample_evenly(intervals: &[Interval], target: usize) -> Vec<Interval> {
    if target == 0 || intervals.is_empty() {
        return vec![];
    }

    if intervals.len() <= target {
        return intervals.to_vec();
    }

    if target == 1 {
        return vec![intervals[intervals.len() / 2]];
    }

    let len = intervals.len();
    let mut out = Vec::with_capacity(target);
    let mut used = vec![false; len];

    for i in 0..target {
        let mut idx = i.saturating_mul(len.saturating_sub(1)) / target.saturating_sub(1);
        while idx < len && used[idx] {
            idx += 1;
        }
        if idx >= len {
            idx = i.saturating_mul(len.saturating_sub(1)) / target.saturating_sub(1);
            while used[idx] {
                idx = idx.saturating_sub(1);
            }
        }

        used[idx] = true;
        out.push(intervals[idx]);
    }

    out.sort_unstable();
    out
}

fn generate_random_intervals(
    n: usize,
    min_segment_len: usize,
    target: usize,
    seed: u64,
    candidates: &[usize],
) -> Result<Vec<Interval>, CpdError> {
    if n < min_segment_len.saturating_mul(2) || target == 0 {
        return Ok(vec![]);
    }

    let mut rng = StableRng::new(seed);
    let mut intervals = Vec::with_capacity(target);
    let max_attempts = target.saturating_mul(64).max(4096);

    for _ in 0..max_attempts {
        if intervals.len() >= target {
            break;
        }

        let start = rng.gen_range(n.saturating_sub(1))?;
        let min_len = min_segment_len.saturating_mul(2);
        let max_len = n.saturating_sub(start);
        if max_len < min_len {
            continue;
        }

        let len_choices = max_len
            .checked_sub(min_len)
            .and_then(|delta| delta.checked_add(1))
            .ok_or_else(|| CpdError::resource_limit("random interval length range overflow"))?;
        let len = min_len
            .checked_add(rng.gen_range(len_choices)?)
            .ok_or_else(|| CpdError::resource_limit("random interval length overflow"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| CpdError::resource_limit("random interval end overflow"))?;
        let interval = Interval { start, end };

        if interval_feasible(interval, candidates, min_segment_len)? {
            intervals.push(interval);
        }
    }

    let deduped = dedup_sorted_intervals(intervals);
    Ok(downsample_evenly(&deduped, target))
}

fn generate_deterministic_grid_intervals(
    n: usize,
    min_segment_len: usize,
    target: usize,
    candidates: &[usize],
) -> Result<Vec<Interval>, CpdError> {
    if n < min_segment_len.saturating_mul(2) || target == 0 {
        return Ok(vec![]);
    }

    let mut intervals = vec![];
    let lengths = dyadic_lengths(n, min_segment_len);

    for len in lengths {
        let mut step = len / 2;
        if step == 0 {
            step = 1;
        }

        for start in (0..=n - len).step_by(step) {
            let interval = Interval {
                start,
                end: start + len,
            };
            if interval_feasible(interval, candidates, min_segment_len)? {
                intervals.push(interval);
            }
        }

        if n > len {
            let edge_start = n - len;
            let edge_interval = Interval {
                start: edge_start,
                end: n,
            };
            if interval_feasible(edge_interval, candidates, min_segment_len)? {
                intervals.push(edge_interval);
            }
        }
    }

    let deduped = dedup_sorted_intervals(intervals);
    Ok(downsample_evenly(&deduped, target))
}

fn generate_stratified_intervals(
    n: usize,
    min_segment_len: usize,
    target: usize,
    seed: u64,
    candidates: &[usize],
) -> Result<Vec<Interval>, CpdError> {
    if n < min_segment_len.saturating_mul(2) || target == 0 {
        return Ok(vec![]);
    }

    let lengths = dyadic_lengths(n, min_segment_len);
    if lengths.is_empty() {
        return Ok(vec![]);
    }

    let mut rng = StableRng::new(seed ^ 0xa0761d6478bd642f);
    let bucket_count = lengths.len();
    let base = target / bucket_count;
    let remainder = target % bucket_count;

    let mut intervals = Vec::with_capacity(target);

    for (idx, &len) in lengths.iter().enumerate() {
        let bucket_target = base + usize::from(idx < remainder);
        if bucket_target == 0 {
            continue;
        }

        let max_start = n - len;
        let max_attempts = bucket_target.saturating_mul(64).max(256);
        let mut accepted = 0usize;

        for _ in 0..max_attempts {
            if accepted >= bucket_target {
                break;
            }
            let start = rng.gen_range(max_start + 1)?;
            let interval = Interval {
                start,
                end: start + len,
            };
            if interval_feasible(interval, candidates, min_segment_len)? {
                intervals.push(interval);
                accepted += 1;
            }
        }
    }

    if intervals.len() < target {
        let needed = target - intervals.len();
        let mut extra = generate_random_intervals(
            n,
            min_segment_len,
            needed,
            seed ^ 0xe7037ed1a0b428db,
            candidates,
        )?;
        intervals.append(&mut extra);
    }

    let deduped = dedup_sorted_intervals(intervals);
    Ok(downsample_evenly(&deduped, target))
}

fn generate_intervals(
    strategy: WbsIntervalStrategy,
    target: usize,
    seed: u64,
    n: usize,
    validated: &ValidatedConstraints,
) -> Result<Vec<Interval>, CpdError> {
    match strategy {
        WbsIntervalStrategy::Random => generate_random_intervals(
            n,
            validated.min_segment_len,
            target,
            seed,
            &validated.effective_candidates,
        ),
        WbsIntervalStrategy::DeterministicGrid => generate_deterministic_grid_intervals(
            n,
            validated.min_segment_len,
            target,
            &validated.effective_candidates,
        ),
        WbsIntervalStrategy::Stratified => generate_stratified_intervals(
            n,
            validated.min_segment_len,
            target,
            seed,
            &validated.effective_candidates,
        ),
    }
}

fn best_split_for_interval_sequential<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    candidates: &[usize],
    interval: Interval,
    validated: &ValidatedConstraints,
    ctx: &ExecutionContext<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<Option<IntervalScore>, CpdError> {
    let Some(window) = split_window_for_interval(interval, candidates, validated.min_segment_len)?
    else {
        return Ok(None);
    };

    let full_cost =
        evaluate_segment_cost(model, cache, interval.start, interval.end, ctx, runtime)?;

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_split = usize::MAX;

    for &split in &candidates[window.start_idx..window.end_idx] {
        checked_counter_increment(iteration, "iteration")?;
        check_runtime_controls(*iteration, cancel_check_every, ctx, started_at, runtime)?;

        checked_counter_increment(&mut runtime.candidates_considered, "candidates_considered")?;
        checked_counter_increment(&mut runtime.intervals_considered, "intervals_considered")?;

        let left_cost = evaluate_segment_cost(model, cache, interval.start, split, ctx, runtime)?;
        let right_cost = evaluate_segment_cost(model, cache, split, interval.end, ctx, runtime)?;

        let gain = full_cost - left_cost - right_cost;
        if !gain.is_finite() {
            return Err(CpdError::numerical_issue(format!(
                "non-finite gain at interval=[{}, {}), split={split}: full_cost={full_cost}, left_cost={left_cost}, right_cost={right_cost}, gain={gain}",
                interval.start, interval.end
            )));
        }

        if gain > best_gain || (gain == best_gain && split < best_split) {
            best_gain = gain;
            best_split = split;
        }
    }

    if best_split == usize::MAX {
        return Ok(None);
    }

    Ok(Some(IntervalScore {
        split: best_split,
        gain: best_gain,
        interval_start: interval.start,
        #[cfg(feature = "rayon")]
        cost_evals: 0,
        #[cfg(feature = "rayon")]
        candidates_considered: 0,
    }))
}

#[cfg(feature = "rayon")]
fn best_split_for_interval_parallel<C: CostModel>(
    model: &C,
    cache: &C::Cache,
    candidates: &[usize],
    interval: Interval,
    validated: &ValidatedConstraints,
) -> Result<Option<IntervalScore>, CpdError> {
    let Some(window) = split_window_for_interval(interval, candidates, validated.min_segment_len)?
    else {
        return Ok(None);
    };

    let full_cost = model.segment_cost(cache, interval.start, interval.end);
    if !full_cost.is_finite() {
        return Err(CpdError::numerical_issue(format!(
            "non-finite segment cost at [{}, {}): {full_cost}",
            interval.start, interval.end
        )));
    }

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_split = usize::MAX;
    let mut considered = 0usize;

    for &split in &candidates[window.start_idx..window.end_idx] {
        considered = considered
            .checked_add(1)
            .ok_or_else(|| CpdError::resource_limit("parallel considered counter overflow"))?;

        let left_cost = model.segment_cost(cache, interval.start, split);
        let right_cost = model.segment_cost(cache, split, interval.end);
        if !left_cost.is_finite() || !right_cost.is_finite() {
            return Err(CpdError::numerical_issue(format!(
                "non-finite child cost at interval=[{}, {}), split={split}: left_cost={left_cost}, right_cost={right_cost}",
                interval.start, interval.end
            )));
        }

        let gain = full_cost - left_cost - right_cost;
        if !gain.is_finite() {
            return Err(CpdError::numerical_issue(format!(
                "non-finite gain at interval=[{}, {}), split={split}: full_cost={full_cost}, left_cost={left_cost}, right_cost={right_cost}, gain={gain}",
                interval.start, interval.end
            )));
        }

        if gain > best_gain || (gain == best_gain && split < best_split) {
            best_gain = gain;
            best_split = split;
        }
    }

    if best_split == usize::MAX {
        return Ok(None);
    }

    let cost_evals = considered
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| CpdError::resource_limit("parallel cost_evals overflow"))?;

    Ok(Some(IntervalScore {
        split: best_split,
        gain: best_gain,
        interval_start: interval.start,
        cost_evals,
        candidates_considered: considered,
    }))
}

#[cfg(feature = "rayon")]
fn can_use_parallel(ctx: &ExecutionContext<'_>) -> bool {
    ctx.repro_mode != ReproMode::Strict
        && ctx.cancel.is_none()
        && ctx.constraints.max_cost_evals.is_none()
        && ctx.constraints.time_budget_ms.is_none()
}

#[cfg(not(feature = "rayon"))]
#[allow(dead_code)]
fn can_use_parallel(_ctx: &ExecutionContext<'_>) -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
fn best_split_for_segment<C: CostModel + Sync>(
    model: &C,
    cache: &C::Cache,
    intervals: &[Interval],
    candidates: &[usize],
    segment: Segment,
    validated: &ValidatedConstraints,
    ctx: &ExecutionContext<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<Option<SegmentCandidate>, CpdError> {
    if !segment_can_split(segment, validated) {
        return Ok(None);
    }

    let mut relevant: Vec<Interval> = intervals
        .iter()
        .copied()
        .filter(|&interval| segment_contains_interval(segment, interval))
        .collect();

    if relevant.is_empty() {
        return Ok(None);
    }

    relevant.sort_unstable();

    #[cfg(feature = "rayon")]
    if can_use_parallel(ctx) {
        let interval_scores: Vec<Option<IntervalScore>> = relevant
            .par_iter()
            .map(|&interval| {
                best_split_for_interval_parallel(model, cache, candidates, interval, validated)
            })
            .collect::<Result<Vec<_>, CpdError>>()?;

        let mut best: Option<IntervalScore> = None;
        for score_opt in interval_scores {
            let Some(score) = score_opt else {
                continue;
            };

            checked_counter_add(&mut runtime.cost_evals, score.cost_evals, "cost_evals")?;
            checked_counter_add(
                &mut runtime.candidates_considered,
                score.candidates_considered,
                "candidates_considered",
            )?;
            checked_counter_add(
                &mut runtime.intervals_considered,
                score.candidates_considered,
                "intervals_considered",
            )?;

            best = match best {
                Some(current) => {
                    if interval_score_better(score, current) {
                        Some(score)
                    } else {
                        Some(current)
                    }
                }
                None => Some(score),
            };
        }

        if let Some(best_score) = best {
            runtime.used_parallel = true;
            return Ok(Some(SegmentCandidate {
                segment,
                split: best_score.split,
                gain: best_score.gain,
            }));
        }

        return Ok(None);
    }

    let mut best: Option<IntervalScore> = None;

    for interval in relevant {
        let Some(score) = best_split_for_interval_sequential(
            model,
            cache,
            candidates,
            interval,
            validated,
            ctx,
            cancel_check_every,
            started_at,
            runtime,
            iteration,
        )?
        else {
            continue;
        };

        best = match best {
            Some(current) => {
                if interval_score_better(score, current) {
                    Some(score)
                } else {
                    Some(current)
                }
            }
            None => Some(score),
        };
    }

    if let Some(best_score) = best {
        return Ok(Some(SegmentCandidate {
            segment,
            split: best_score.split,
            gain: best_score.gain,
        }));
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn add_segment_to_frontier<C: CostModel + Sync>(
    frontier: &mut Vec<SegmentCandidate>,
    model: &C,
    cache: &C::Cache,
    intervals: &[Interval],
    candidates: &[usize],
    segment: Segment,
    validated: &ValidatedConstraints,
    ctx: &ExecutionContext<'_>,
    cancel_check_every: usize,
    started_at: Instant,
    runtime: &mut RuntimeStats,
    iteration: &mut usize,
) -> Result<(), CpdError> {
    if let Some(candidate) = best_split_for_segment(
        model,
        cache,
        intervals,
        candidates,
        segment,
        validated,
        ctx,
        cancel_check_every,
        started_at,
        runtime,
        iteration,
    )? {
        frontier.push(candidate);
    }

    Ok(())
}

fn pick_best_frontier_index(frontier: &[SegmentCandidate]) -> Option<usize> {
    let mut best_idx: Option<usize> = None;

    for (idx, &candidate) in frontier.iter().enumerate() {
        if let Some(current_best_idx) = best_idx {
            let current_best = frontier[current_best_idx];
            if segment_candidate_better(candidate, current_best) {
                best_idx = Some(idx);
            }
        } else {
            best_idx = Some(idx);
        }
    }

    best_idx
}

fn insert_sorted_unique(values: &mut Vec<usize>, value: usize) -> Result<(), CpdError> {
    match values.binary_search(&value) {
        Ok(_) => Err(CpdError::invalid_input(format!(
            "duplicate split selected at {value}; internal WBS state is inconsistent"
        ))),
        Err(idx) => {
            values.insert(idx, value);
            Ok(())
        }
    }
}

fn build_result_breakpoints(n: usize, change_points: Vec<usize>) -> Vec<usize> {
    let mut breakpoints = change_points;
    breakpoints.push(n);
    breakpoints
}

impl<C: CostModel + Sync> OfflineDetector for Wbs<C> {
    fn detect(
        &self,
        x: &TimeSeriesView<'_>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OfflineChangePointResult, CpdError> {
        self.config.validate()?;

        let validated = validate_constraints(ctx.constraints, x.n)?;

        self.cost_model.validate(x)?;
        check_missing_compatibility(x.missing, self.cost_model.missing_support())?;
        let cache = self.cost_model.precompute(x, &validated.cache_policy)?;

        let started_at = Instant::now();
        let cancel_check_every = self.config.normalized_cancel_check_every();
        let mut runtime = RuntimeStats::default();
        let mut iteration = 0usize;

        let requested_intervals = resolve_num_intervals(self.config.num_intervals, x.n)?;
        let intervals = generate_intervals(
            self.config.interval_strategy,
            requested_intervals,
            self.config.seed,
            x.n,
            &validated,
        )?;

        let mut notes = vec![format!(
            "interval_strategy={:?}, requested_intervals={}, effective_intervals={}",
            self.config.interval_strategy,
            requested_intervals,
            intervals.len()
        )];
        let mut warnings = vec![];

        if intervals.is_empty() {
            notes.push("no feasible WBS intervals under current constraints".to_string());
        }

        let candidates = &validated.effective_candidates;

        let root = Segment {
            start: 0,
            end: x.n,
            depth: 0,
        };

        let mut frontier = Vec::new();
        add_segment_to_frontier(
            &mut frontier,
            &self.cost_model,
            &cache,
            &intervals,
            candidates,
            root,
            &validated,
            ctx,
            cancel_check_every,
            started_at,
            &mut runtime,
            &mut iteration,
        )?;

        let mut accepted_splits = vec![];

        match &self.config.stopping {
            Stopping::KnownK(k) => {
                if let Some(max_change_points) = validated.max_change_points
                    && max_change_points < *k
                {
                    return Err(CpdError::invalid_input(format!(
                        "KnownK={k} exceeds constraints.max_change_points={max_change_points}"
                    )));
                }

                for _round in 0..*k {
                    checked_counter_increment(&mut iteration, "iteration")?;
                    check_runtime_controls(
                        iteration,
                        cancel_check_every,
                        ctx,
                        started_at,
                        &mut runtime,
                    )?;

                    let Some(best_idx) = pick_best_frontier_index(&frontier) else {
                        return Err(CpdError::invalid_input(format!(
                            "KnownK exact solution unreachable: requested k={k}, accepted={} before frontier exhaustion",
                            accepted_splits.len()
                        )));
                    };

                    let best = frontier.swap_remove(best_idx);
                    insert_sorted_unique(&mut accepted_splits, best.split)?;

                    let child_depth = best
                        .segment
                        .depth
                        .checked_add(1)
                        .ok_or_else(|| CpdError::resource_limit("segment depth overflow"))?;
                    let left = Segment {
                        start: best.segment.start,
                        end: best.split,
                        depth: child_depth,
                    };
                    let right = Segment {
                        start: best.split,
                        end: best.segment.end,
                        depth: child_depth,
                    };

                    add_segment_to_frontier(
                        &mut frontier,
                        &self.cost_model,
                        &cache,
                        &intervals,
                        candidates,
                        left,
                        &validated,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;
                    add_segment_to_frontier(
                        &mut frontier,
                        &self.cost_model,
                        &cache,
                        &intervals,
                        candidates,
                        right,
                        &validated,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;

                    ctx.report_progress(accepted_splits.len() as f32 / *k as f32);
                }

                notes.push(format!(
                    "stopping=KnownK({k}), accepted_splits={}",
                    accepted_splits.len()
                ));
            }
            Stopping::Penalized(penalty) => {
                let beta = resolve_penalty_beta(penalty, x.n, x.d, self.config.params_per_segment)?;
                notes.push(format!("stopping=Penalized({penalty:?}), beta={beta}"));

                let mut processed_frontier_items = 0usize;
                loop {
                    checked_counter_increment(&mut iteration, "iteration")?;
                    check_runtime_controls(
                        iteration,
                        cancel_check_every,
                        ctx,
                        started_at,
                        &mut runtime,
                    )?;

                    if let Some(max_change_points) = validated.max_change_points
                        && accepted_splits.len() >= max_change_points
                    {
                        break;
                    }

                    let Some(best_idx) = pick_best_frontier_index(&frontier) else {
                        break;
                    };

                    let best = frontier[best_idx];
                    if best.gain <= beta {
                        break;
                    }

                    let best = frontier.swap_remove(best_idx);
                    insert_sorted_unique(&mut accepted_splits, best.split)?;
                    checked_counter_increment(
                        &mut processed_frontier_items,
                        "processed_frontier_items",
                    )?;

                    let child_depth = best
                        .segment
                        .depth
                        .checked_add(1)
                        .ok_or_else(|| CpdError::resource_limit("segment depth overflow"))?;
                    let left = Segment {
                        start: best.segment.start,
                        end: best.split,
                        depth: child_depth,
                    };
                    let right = Segment {
                        start: best.split,
                        end: best.segment.end,
                        depth: child_depth,
                    };

                    add_segment_to_frontier(
                        &mut frontier,
                        &self.cost_model,
                        &cache,
                        &intervals,
                        candidates,
                        left,
                        &validated,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;
                    add_segment_to_frontier(
                        &mut frontier,
                        &self.cost_model,
                        &cache,
                        &intervals,
                        candidates,
                        right,
                        &validated,
                        ctx,
                        cancel_check_every,
                        started_at,
                        &mut runtime,
                        &mut iteration,
                    )?;

                    let progress_denom = (processed_frontier_items + frontier.len() + 1) as f32;
                    if progress_denom.is_finite() && progress_denom > 0.0 {
                        ctx.report_progress(processed_frontier_items as f32 / progress_denom);
                    }
                }
            }
            Stopping::PenaltyPath(path) => {
                return Err(CpdError::not_supported(format!(
                    "WBS penalty sweep is deferred for this issue; got PenaltyPath of length {}",
                    path.len()
                )));
            }
        }

        if runtime.soft_budget_exceeded {
            warnings.push(
                "budget exceeded under SoftDegrade mode; run continued without algorithm fallback"
                    .to_string(),
            );
        }

        let runtime_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);

        ctx.record_scalar("offline.wbs.cost_evals", runtime.cost_evals as f64);
        ctx.record_scalar(
            "offline.wbs.candidates_considered",
            runtime.candidates_considered as f64,
        );
        ctx.record_scalar(
            "offline.wbs.intervals_considered",
            runtime.intervals_considered as f64,
        );
        ctx.record_scalar("offline.wbs.runtime_ms", runtime_ms as f64);
        ctx.report_progress(1.0);

        notes.push(format!(
            "final_change_count={}, cost_evals={}, candidates_considered={}, intervals_considered={}, used_parallel={}",
            accepted_splits.len(),
            runtime.cost_evals,
            runtime.candidates_considered,
            runtime.intervals_considered,
            runtime.used_parallel
        ));

        #[cfg(feature = "rayon")]
        let thread_count = if runtime.used_parallel {
            Some(rayon::current_num_threads())
        } else {
            None
        };

        #[cfg(not(feature = "rayon"))]
        let thread_count = None;

        let diagnostics = Diagnostics {
            n: x.n,
            d: x.d,
            runtime_ms: Some(runtime_ms),
            notes,
            warnings,
            algorithm: Cow::Borrowed("wbs"),
            cost_model: Cow::Borrowed(self.cost_model.name()),
            seed: Some(self.config.seed),
            repro_mode: ctx.repro_mode,
            thread_count,
            ..Diagnostics::default()
        };

        let breakpoints = build_result_breakpoints(x.n, accepted_splits);
        OfflineChangePointResult::new(x.n, breakpoints, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::{Wbs, WbsConfig, WbsIntervalStrategy};
    use crate::{BinSeg, BinSegConfig};
    use cpd_core::{
        BudgetMode, Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy,
        OfflineDetector, Penalty, ProgressSink, Stopping, TimeIndex, TimeSeriesView,
    };
    use cpd_costs::{CostL2Mean, CostNormalMeanVar};
    use std::thread;
    use std::time::Duration;

    fn make_f64_view<'a>(
        values: &'a [f64],
        n: usize,
        d: usize,
        layout: MemoryLayout,
        missing: MissingPolicy,
    ) -> TimeSeriesView<'a> {
        TimeSeriesView::new(
            DTypeView::F64(values),
            n,
            d,
            layout,
            None,
            TimeIndex::None,
            missing,
        )
        .expect("test view should be valid")
    }

    fn constraints_with_min_segment_len(min_segment_len: usize) -> Constraints {
        Constraints {
            min_segment_len,
            ..Constraints::default()
        }
    }

    fn assert_strictly_increasing(values: &[usize]) {
        for window in values.windows(2) {
            assert!(window[0] < window[1], "not strictly increasing: {values:?}");
        }
    }

    #[test]
    fn config_defaults_and_validation() {
        let default_cfg = WbsConfig::default();
        assert_eq!(default_cfg.stopping, Stopping::Penalized(Penalty::BIC));
        assert_eq!(default_cfg.params_per_segment, 2);
        assert_eq!(default_cfg.num_intervals, None);
        assert_eq!(default_cfg.interval_strategy, WbsIntervalStrategy::Random);
        assert_eq!(default_cfg.seed, 0);
        assert_eq!(default_cfg.cancel_check_every, 1000);

        let ok = Wbs::new(CostL2Mean::default(), default_cfg.clone())
            .expect("default config should be valid");
        assert_eq!(ok.config(), &default_cfg);

        let err = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                params_per_segment: 0,
                ..default_cfg.clone()
            },
        )
        .expect_err("params_per_segment=0 must fail");
        assert!(err.to_string().contains("params_per_segment"));

        let err_num = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                num_intervals: Some(0),
                ..default_cfg
            },
        )
        .expect_err("num_intervals=0 must fail");
        assert!(err_num.to_string().contains("num_intervals"));
    }

    #[test]
    fn cancel_check_every_zero_is_normalized() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 2,
                num_intervals: Some(128),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 7,
                cancel_check_every: 0,
            },
        )
        .expect("config with zero cadence should normalize");

        let values = vec![0.0, 0.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(1);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
    }

    #[test]
    fn known_small_example_one_change_l2() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 2,
                num_intervals: Some(256),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 11,
                cancel_check_every: 8,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![4, 8]);
    }

    #[test]
    fn known_small_example_two_changes_l2() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                num_intervals: Some(512),
                interval_strategy: WbsIntervalStrategy::Stratified,
                seed: 13,
                cancel_check_every: 4,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, -4.0, -4.0, -4.0, -4.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![4, 8, 12]);
    }

    #[test]
    fn normal_cost_detects_variance_change() {
        let detector = Wbs::new(
            CostNormalMeanVar::default(),
            WbsConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 3,
                num_intervals: Some(256),
                interval_strategy: WbsIntervalStrategy::DeterministicGrid,
                seed: 17,
                cancel_check_every: 2,
            },
        )
        .expect("config should be valid");

        let mut values = Vec::with_capacity(20);
        for _ in 0..5 {
            values.push(-1.0);
            values.push(1.0);
        }
        for _ in 0..5 {
            values.push(-5.0);
            values.push(5.0);
        }

        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 2,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");
        assert_eq!(result.breakpoints, vec![10, 20]);
    }

    #[test]
    fn known_k_exact_unreachable_is_clear_error() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                num_intervals: Some(128),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 23,
                cancel_check_every: 2,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 2,
            candidate_splits: Some(vec![4]),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("known-k should be unreachable");
        assert!(
            err.to_string()
                .contains("KnownK exact solution unreachable")
        );
    }

    #[test]
    fn known_k_fails_when_max_change_points_below_k() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::KnownK(2),
                params_per_segment: 2,
                num_intervals: Some(128),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 29,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_change_points: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("KnownK must reject when max_change_points < k");
        assert!(err.to_string().contains("max_change_points"));
    }

    #[test]
    fn penalty_path_returns_not_supported() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::PenaltyPath(vec![Penalty::Manual(1.0)]),
                params_per_segment: 2,
                num_intervals: Some(64),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 31,
                cancel_check_every: 16,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 2.0, 3.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(1);
        let ctx = ExecutionContext::new(&constraints);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("PenaltyPath should be deferred");
        assert!(matches!(err, cpd_core::CpdError::NotSupported(_)));
    }

    #[test]
    fn repeated_runs_with_same_seed_are_deterministic() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(2.0)),
                params_per_segment: 2,
                num_intervals: Some(256),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 37,
                cancel_check_every: 2,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            jump: 1,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let first = detector
            .detect(&view, &ctx)
            .expect("first detect should pass");
        let second = detector
            .detect(&view, &ctx)
            .expect("second detect should pass");
        assert_eq!(first.breakpoints, second.breakpoints);
        assert_eq!(first.diagnostics.seed, Some(37));
    }

    #[test]
    fn different_seeds_change_sampling_but_still_find_clean_split() {
        let values = vec![0.0; 100]
            .into_iter()
            .enumerate()
            .map(|(idx, value)| if idx < 50 { value } else { value + 8.0 })
            .collect::<Vec<_>>();
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 2,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let det_a = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 2,
                num_intervals: Some(128),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 41,
                cancel_check_every: 8,
            },
        )
        .expect("detector A should be valid");
        let det_b = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                seed: 43,
                ..det_a.config().clone()
            },
        )
        .expect("detector B should be valid");

        let result_a = det_a.detect(&view, &ctx).expect("A should succeed");
        let result_b = det_b.detect(&view, &ctx).expect("B should succeed");

        assert_eq!(result_a.breakpoints, vec![50, 100]);
        assert_eq!(result_b.breakpoints, vec![50, 100]);
        assert_ne!(result_a.diagnostics.seed, result_b.diagnostics.seed);
    }

    #[test]
    fn all_interval_strategies_produce_valid_results() {
        let values = vec![
            0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0, -3.0, -3.0, -3.0, -3.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 2,
            jump: 1,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        for strategy in [
            WbsIntervalStrategy::Random,
            WbsIntervalStrategy::DeterministicGrid,
            WbsIntervalStrategy::Stratified,
        ] {
            let detector = Wbs::new(
                CostL2Mean::default(),
                WbsConfig {
                    stopping: Stopping::KnownK(2),
                    params_per_segment: 2,
                    num_intervals: Some(256),
                    interval_strategy: strategy,
                    seed: 47,
                    cancel_check_every: 4,
                },
            )
            .expect("strategy config should be valid");

            let result = detector
                .detect(&view, &ctx)
                .expect("strategy should detect");
            assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
            assert_strictly_increasing(&result.breakpoints);
            for &cp in &result.change_points {
                assert_eq!(cp % 1, 0);
            }
        }
    }

    #[test]
    fn constraints_min_segment_len_and_jump_are_enforced() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.5)),
                params_per_segment: 2,
                num_intervals: Some(256),
                interval_strategy: WbsIntervalStrategy::Stratified,
                seed: 53,
                cancel_check_every: 3,
            },
        )
        .expect("config should be valid");

        let n = 24;
        let values: Vec<f64> = (0..n)
            .map(|idx| {
                if idx < 8 {
                    0.0
                } else if idx < 16 {
                    5.0
                } else {
                    10.0
                }
            })
            .collect();
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = Constraints {
            min_segment_len: 4,
            jump: 4,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");

        for &cp in &result.change_points {
            assert_eq!(cp % 4, 0, "change point must respect jump=4");
        }
        assert_eq!(result.breakpoints.last().copied(), Some(n));
        let mut start = 0usize;
        for &end in &result.breakpoints {
            assert!(
                end - start >= 4,
                "segment [{start}, {end}) violates min_segment_len=4"
            );
            start = end;
        }
    }

    #[test]
    fn explicit_candidate_splits_and_max_change_points_are_enforced() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                num_intervals: Some(128),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 59,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_change_points: Some(1),
            candidate_splits: Some(vec![4, 8]),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");

        assert!(result.change_points.len() <= 1);
        for cp in result.change_points {
            assert!(cp == 4 || cp == 8);
        }
    }

    #[test]
    fn max_depth_reached_stops_cleanly() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                num_intervals: Some(128),
                interval_strategy: WbsIntervalStrategy::Stratified,
                seed: 61,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -2.0, -2.0, -2.0, 4.0, 4.0, 4.0,
        ];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_depth: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector.detect(&view, &ctx).expect("detect should succeed");

        assert!(result.change_points.len() <= 1);
        assert_eq!(result.breakpoints.last().copied(), Some(values.len()));
    }

    #[test]
    fn cancellation_mid_run_returns_cancelled() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                num_intervals: Some(1024),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 67,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let n = 5_000;
        let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );

        let constraints = constraints_with_min_segment_len(1);
        let cancel = cpd_core::CancelToken::new();
        cancel.cancel();
        let ctx = ExecutionContext::new(&constraints).with_cancel(&cancel);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("cancelled token must stop detect");
        assert_eq!(err.to_string(), "cancelled");
    }

    #[test]
    fn cost_eval_budget_exceeded_hard_fail() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                num_intervals: Some(64),
                interval_strategy: WbsIntervalStrategy::DeterministicGrid,
                seed: 71,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_cost_evals: Some(1),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("hard budget should fail");
        assert!(err.to_string().contains("max_cost_evals"));
    }

    struct SlowProgressSink;

    impl ProgressSink for SlowProgressSink {
        fn on_progress(&self, _fraction: f32) {
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn time_budget_exceeded_hard_fail() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                num_intervals: Some(128),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 73,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            time_budget_ms: Some(1),
            ..Constraints::default()
        };
        let slow_progress = SlowProgressSink;
        let ctx = ExecutionContext::new(&constraints)
            .with_budget_mode(BudgetMode::HardFail)
            .with_progress_sink(&slow_progress);
        let err = detector
            .detect(&view, &ctx)
            .expect_err("hard time budget should fail");
        assert!(err.to_string().contains("time_budget_ms"));
    }

    #[test]
    fn diagnostics_include_soft_budget_and_seed() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(0.1)),
                params_per_segment: 2,
                num_intervals: Some(64),
                interval_strategy: WbsIntervalStrategy::Random,
                seed: 79,
                cancel_check_every: 1,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 1.0, 0.0, 1.0, 5.0, 6.0, 5.0, 6.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 1,
            max_cost_evals: Some(2),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::SoftDegrade);
        let result = detector
            .detect(&view, &ctx)
            .expect("soft degrade should continue");
        let diagnostics = result.diagnostics;
        assert_eq!(diagnostics.algorithm, "wbs");
        assert_eq!(diagnostics.cost_model, "l2_mean");
        assert_eq!(diagnostics.seed, Some(79));
        assert!(
            diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("SoftDegrade"))
        );
    }

    #[test]
    fn masking_demo_wbs_recovers_more_changes_than_binseg() {
        let mut values = vec![0.0; 40];
        values.extend(std::iter::repeat_n(6.0, 6));
        values.extend(std::iter::repeat_n(0.0, 40));
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 2,
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);

        let binseg = BinSeg::new(
            CostL2Mean::default(),
            BinSegConfig {
                stopping: Stopping::Penalized(Penalty::Manual(40.0)),
                params_per_segment: 2,
                cancel_check_every: 1,
            },
        )
        .expect("binseg config should be valid");
        let wbs = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::Manual(40.0)),
                params_per_segment: 2,
                num_intervals: Some(1024),
                interval_strategy: WbsIntervalStrategy::Stratified,
                seed: 83,
                cancel_check_every: 1,
            },
        )
        .expect("wbs config should be valid");

        let binseg_result = binseg.detect(&view, &ctx).expect("binseg should pass");
        let wbs_result = wbs.detect(&view, &ctx).expect("wbs should pass");

        assert!(
            wbs_result.change_points.len() >= binseg_result.change_points.len(),
            "wbs should recover at least as many changes as binseg on masking demo; wbs={:?}, binseg={:?}",
            wbs_result.change_points,
            binseg_result.change_points
        );
    }

    #[test]
    fn large_n_regression_smoke() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::Penalized(Penalty::BIC),
                params_per_segment: 2,
                num_intervals: Some(100),
                interval_strategy: WbsIntervalStrategy::Stratified,
                seed: 89,
                cancel_check_every: 1024,
            },
        )
        .expect("config should be valid");

        let n = 100_000;
        let mut values = vec![0.0; n];
        for v in values.iter_mut().skip(n / 2) {
            *v = 2.0;
        }
        let view = make_f64_view(
            &values,
            n,
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = Constraints {
            min_segment_len: 50,
            jump: 50,
            max_change_points: Some(8),
            ..Constraints::default()
        };
        let ctx = ExecutionContext::new(&constraints);
        let result = detector
            .detect(&view, &ctx)
            .expect("large-n smoke should pass");

        assert_eq!(result.breakpoints.last().copied(), Some(n));
        assert_strictly_increasing(&result.breakpoints);
        assert!(result.change_points.len() <= 8);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn strict_repro_mode_disables_parallel_thread_count() {
        let detector = Wbs::new(
            CostL2Mean::default(),
            WbsConfig {
                stopping: Stopping::KnownK(1),
                params_per_segment: 2,
                num_intervals: Some(128),
                interval_strategy: WbsIntervalStrategy::Stratified,
                seed: 97,
                cancel_check_every: 32,
            },
        )
        .expect("config should be valid");

        let values = vec![0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0];
        let view = make_f64_view(
            &values,
            values.len(),
            1,
            MemoryLayout::CContiguous,
            MissingPolicy::Error,
        );
        let constraints = constraints_with_min_segment_len(2);
        let ctx = ExecutionContext::new(&constraints).with_repro_mode(cpd_core::ReproMode::Strict);
        let result = detector
            .detect(&view, &ctx)
            .expect("strict mode should detect");
        assert_eq!(result.diagnostics.thread_count, None);
    }
}
