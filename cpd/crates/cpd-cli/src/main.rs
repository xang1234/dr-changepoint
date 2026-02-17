// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_cli::run_pipeline;
use cpd_core::{
    Constraints, CpdError, MemoryLayout, MissingPolicy, OfflineChangePointResult, OnlineStepResult,
    Penalty, Stopping, TimeIndex, TimeSeriesView,
};
use cpd_doctor::{
    CostConfig, DetectorConfig, Objective, OfflineDetectorConfig, PipelineSpec, Recommendation,
    recommend,
};
use cpd_eval::{offline_metrics, online_metrics};
use cpd_offline::{BinSegConfig, FpopConfig, PeltConfig, WbsConfig, WbsIntervalStrategy};
use serde::Serialize;
use serde_json::{Map, Value};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

struct Cli {
    command: Command,
}

enum Command {
    Detect(DetectArgs),
    Run(RunArgs),
    Doctor(DoctorArgs),
    Eval(EvalArgs),
}

#[derive(Debug)]
struct DetectArgs {
    algorithm: AlgorithmArg,
    cost: CostArg,
    penalty: PenaltyArg,
    penalty_explicit: bool,
    penalty_value: Option<f64>,
    k: Option<usize>,
    seed: Option<u64>,
    min_segment_len: Option<usize>,
    max_change_points: Option<usize>,
    max_depth: Option<usize>,
    jump: Option<usize>,
    input: PathBuf,
    output: Option<PathBuf>,
}

impl Default for DetectArgs {
    fn default() -> Self {
        Self {
            algorithm: AlgorithmArg::Pelt,
            cost: CostArg::L2,
            penalty: PenaltyArg::Bic,
            penalty_explicit: false,
            penalty_value: None,
            k: None,
            seed: None,
            min_segment_len: None,
            max_change_points: None,
            max_depth: None,
            jump: None,
            input: PathBuf::new(),
            output: None,
        }
    }
}

#[derive(Debug)]
struct RunArgs {
    pipeline: PathBuf,
    input: PathBuf,
    output: Option<PathBuf>,
}

#[derive(Debug)]
struct DoctorArgs {
    objective: ObjectiveArg,
    min_confidence: f64,
    allow_abstain: bool,
    online: bool,
    constraints: Option<PathBuf>,
    input: PathBuf,
    output: Option<PathBuf>,
}

impl Default for DoctorArgs {
    fn default() -> Self {
        Self {
            objective: ObjectiveArg::Balanced,
            min_confidence: 0.2,
            allow_abstain: false,
            online: false,
            constraints: None,
            input: PathBuf::new(),
            output: None,
        }
    }
}

#[derive(Debug)]
struct EvalArgs {
    predictions: PathBuf,
    ground_truth: PathBuf,
    tolerance: usize,
    output: Option<PathBuf>,
}

impl Default for EvalArgs {
    fn default() -> Self {
        Self {
            predictions: PathBuf::new(),
            ground_truth: PathBuf::new(),
            tolerance: 1,
            output: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AlgorithmArg {
    Pelt,
    Binseg,
    Fpop,
    Wbs,
}

#[derive(Clone, Copy, Debug)]
enum CostArg {
    Ar,
    Cosine,
    L1Median,
    L2,
    Normal,
    NormalFullCov,
    Nig,
    Rank,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PenaltyArg {
    Bic,
    Aic,
    Manual,
}

#[derive(Clone, Copy, Debug)]
enum ObjectiveArg {
    Balanced,
    Speed,
    Accuracy,
    Robustness,
}

impl AlgorithmArg {
    fn parse(raw: &str) -> Result<Self, CliError> {
        match raw.to_ascii_lowercase().as_str() {
            "pelt" => Ok(Self::Pelt),
            "binseg" => Ok(Self::Binseg),
            "fpop" => Ok(Self::Fpop),
            "wbs" => Ok(Self::Wbs),
            _ => Err(CliError::invalid_input(format!(
                "invalid --algorithm '{raw}'; expected one of: pelt, binseg, fpop, wbs"
            ))),
        }
    }
}

impl CostArg {
    fn parse(raw: &str) -> Result<Self, CliError> {
        match raw.to_ascii_lowercase().as_str() {
            "ar" => Ok(Self::Ar),
            "cosine" => Ok(Self::Cosine),
            "l1" | "l1_median" => Ok(Self::L1Median),
            "l2" => Ok(Self::L2),
            "normal" => Ok(Self::Normal),
            "normal_full_cov" | "normal_fullcov" | "normalfullcov" => Ok(Self::NormalFullCov),
            "nig" => Ok(Self::Nig),
            "rank" => Ok(Self::Rank),
            _ => Err(CliError::invalid_input(format!(
                "invalid --cost '{raw}'; expected one of: ar, cosine, l1_median, l2, normal, normal_full_cov, nig, rank"
            ))),
        }
    }
}

impl PenaltyArg {
    fn parse(raw: &str) -> Result<Self, CliError> {
        match raw.to_ascii_lowercase().as_str() {
            "bic" => Ok(Self::Bic),
            "aic" => Ok(Self::Aic),
            "manual" => Ok(Self::Manual),
            _ => Err(CliError::invalid_input(format!(
                "invalid --penalty '{raw}'; expected one of: bic, aic, manual"
            ))),
        }
    }
}

impl ObjectiveArg {
    fn as_objective(self) -> Objective {
        match self {
            Self::Balanced => Objective::Balanced,
            Self::Speed => Objective::Speed,
            Self::Accuracy => Objective::Accuracy,
            Self::Robustness => Objective::Robustness,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Speed => "speed",
            Self::Accuracy => "accuracy",
            Self::Robustness => "robustness",
        }
    }

    fn parse(raw: &str) -> Result<Self, CliError> {
        match raw.to_ascii_lowercase().as_str() {
            "balanced" => Ok(Self::Balanced),
            "speed" => Ok(Self::Speed),
            "accuracy" => Ok(Self::Accuracy),
            "robustness" => Ok(Self::Robustness),
            _ => Err(CliError::invalid_input(format!(
                "invalid --objective '{raw}'; expected one of: balanced, speed, accuracy, robustness"
            ))),
        }
    }
}

#[derive(Debug)]
enum CliError {
    Cpd(CpdError),
    Io {
        context: String,
        source: std::io::Error,
    },
    Json {
        context: String,
        source: serde_json::Error,
    },
    InvalidInput(String),
    NotSupported(String),
}

impl CliError {
    fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    fn not_supported(msg: impl Into<String>) -> Self {
        Self::NotSupported(msg.into())
    }

    fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Cpd(CpdError::InvalidInput(_)) | Self::InvalidInput(_) => "invalid_input",
            Self::Cpd(CpdError::NumericalIssue(_)) => "numerical_issue",
            Self::Cpd(CpdError::NotSupported(_)) | Self::NotSupported(_) => "not_supported",
            Self::Cpd(CpdError::ResourceLimit(_)) => "resource_limit",
            Self::Cpd(CpdError::Cancelled) => "cancelled",
            Self::Io { .. } => "io_error",
            Self::Json { .. } => "json_error",
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpd(err) => write!(f, "{err}"),
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::Json { context, source } => write!(f, "{context}: {source}"),
            Self::InvalidInput(msg) => write!(f, "{msg}"),
            Self::NotSupported(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cpd(err) => Some(err),
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::InvalidInput(_) | Self::NotSupported(_) => None,
        }
    }
}

impl From<CpdError> for CliError {
    fn from(value: CpdError) -> Self {
        Self::Cpd(value)
    }
}

#[derive(Clone, Debug)]
struct LoadedSeries {
    path: PathBuf,
    format: &'static str,
    values: Vec<f64>,
    n: usize,
    d: usize,
}

impl LoadedSeries {
    fn as_view(&self) -> Result<TimeSeriesView<'_>, CliError> {
        TimeSeriesView::from_f64(
            self.values.as_slice(),
            self.n,
            self.d,
            MemoryLayout::CContiguous,
            None,
            TimeIndex::None,
            MissingPolicy::Error,
        )
        .map_err(CliError::from)
    }

    fn summary(&self) -> InputSummary {
        InputSummary {
            path: self.path.display().to_string(),
            format: self.format.to_string(),
            n: self.n,
            d: self.d,
        }
    }
}

#[derive(Serialize)]
struct InputSummary {
    path: String,
    format: String,
    n: usize,
    d: usize,
}

#[derive(Serialize)]
struct DetectOutput {
    command: &'static str,
    input: InputSummary,
    pipeline: PipelineSpec,
    result: OfflineChangePointResult,
}

#[derive(Serialize)]
struct RunOutput {
    command: &'static str,
    input: InputSummary,
    pipeline: PipelineSpec,
    result: OfflineChangePointResult,
}

#[derive(Serialize)]
struct DoctorOutput {
    command: &'static str,
    input: InputSummary,
    objective: String,
    online: bool,
    min_confidence: f64,
    allow_abstain: bool,
    recommendations: Vec<DoctorRecommendationOutput>,
}

#[derive(Serialize)]
struct DoctorRecommendationOutput {
    rank: usize,
    confidence: f64,
    confidence_interval: [f64; 2],
    abstain_reason: Option<String>,
    warnings: Vec<String>,
    pipeline: PipelineSpec,
    explanation: ExplanationOutput,
    resource_estimate: ResourceEstimateOutput,
    validation: Option<ValidationOutput>,
    objective_fit: Vec<ObjectiveFitOutput>,
}

#[derive(Serialize)]
struct ExplanationOutput {
    summary: String,
    drivers: Vec<ObjectiveFitOutput>,
    tradeoffs: Vec<String>,
}

#[derive(Serialize)]
struct ResourceEstimateOutput {
    time_complexity: String,
    memory_complexity: String,
    relative_time_score: f64,
    relative_memory_score: f64,
}

#[derive(Serialize)]
struct ValidationOutput {
    method: String,
    notes: Vec<String>,
}

#[derive(Serialize)]
struct ObjectiveFitOutput {
    key: String,
    value: f64,
}

#[derive(Serialize)]
struct OfflineEvalOutput {
    command: &'static str,
    mode: &'static str,
    tolerance: usize,
    metrics: OfflineMetricsOutput,
}

#[derive(Serialize)]
struct OfflineMetricsOutput {
    f1: F1MetricsOutput,
    hausdorff_distance: f64,
    rand_index: f64,
    annotation_error: f64,
}

#[derive(Serialize)]
struct F1MetricsOutput {
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Serialize)]
struct OnlineEvalOutput {
    command: &'static str,
    mode: &'static str,
    metrics: OnlineMetricsOutput,
}

#[derive(Serialize)]
struct OnlineMetricsOutput {
    mean_detection_delay: Option<f64>,
    false_alarm_rate: f64,
    arl0: f64,
    arl1: Option<f64>,
    detected_changes: usize,
    missed_changes: usize,
    false_alerts: usize,
    total_alerts: usize,
    roc_curve: Vec<RocPointOutput>,
}

#[derive(Serialize)]
struct RocPointOutput {
    threshold: f64,
    true_positive_rate: f64,
    false_positive_rate: f64,
    detected_changes: usize,
    false_alerts: usize,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorPayload,
}

#[derive(Serialize)]
struct ErrorPayload {
    code: String,
    message: String,
}

fn main() {
    if let Err(err) = run() {
        emit_structured_error(&err);
        process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    let Some(cli) = parse_cli_from_env()? else {
        return Ok(());
    };

    match cli.command {
        Command::Detect(args) => handle_detect(args),
        Command::Run(args) => handle_run(args),
        Command::Doctor(args) => handle_doctor(args),
        Command::Eval(args) => handle_eval(args),
    }
}

fn parse_cli_from_env() -> Result<Option<Cli>, CliError> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        print_root_help();
        return Ok(None);
    }

    if matches!(args[0].as_str(), "-h" | "--help") {
        print_root_help();
        return Ok(None);
    }
    if matches!(args[0].as_str(), "-V" | "--version") {
        print_version();
        return Ok(None);
    }

    let command_name = args[0].clone();
    let rest = &args[1..];

    if rest
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        print_command_help(command_name.as_str())?;
        return Ok(None);
    }
    if rest
        .iter()
        .any(|arg| matches!(arg.as_str(), "-V" | "--version"))
    {
        print_version();
        return Ok(None);
    }

    let command = match command_name.as_str() {
        "detect" => Command::Detect(parse_detect_args(rest)?),
        "run" => Command::Run(parse_run_args(rest)?),
        "doctor" => Command::Doctor(parse_doctor_args(rest)?),
        "eval" => Command::Eval(parse_eval_args(rest)?),
        _ => {
            return Err(CliError::invalid_input(format!(
                "unknown command '{}'; expected one of: detect, run, doctor, eval",
                command_name
            )));
        }
    };

    Ok(Some(Cli { command }))
}

fn parse_detect_args(tokens: &[String]) -> Result<DetectArgs, CliError> {
    let mut args = DetectArgs::default();
    let mut idx = 0usize;
    while idx < tokens.len() {
        let (flag, inline_value) = split_flag(tokens[idx].as_str())?;
        match flag {
            "--algorithm" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.algorithm = AlgorithmArg::parse(raw.as_str())?;
            }
            "--cost" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.cost = CostArg::parse(raw.as_str())?;
            }
            "--penalty" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.penalty = PenaltyArg::parse(raw.as_str())?;
                args.penalty_explicit = true;
            }
            "--penalty-value" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.penalty_value = Some(parse_f64_arg(raw.as_str(), flag)?);
            }
            "--k" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.k = Some(parse_usize_arg(raw.as_str(), flag)?);
            }
            "--seed" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.seed = Some(parse_u64_arg(raw.as_str(), flag)?);
            }
            "--min-segment-len" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.min_segment_len = Some(parse_usize_arg(raw.as_str(), flag)?);
            }
            "--max-change-points" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.max_change_points = Some(parse_usize_arg(raw.as_str(), flag)?);
            }
            "--max-depth" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.max_depth = Some(parse_usize_arg(raw.as_str(), flag)?);
            }
            "--jump" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.jump = Some(parse_usize_arg(raw.as_str(), flag)?);
            }
            "--input" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.input = PathBuf::from(raw);
            }
            "--output" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.output = Some(PathBuf::from(raw));
            }
            other => {
                return Err(CliError::invalid_input(format!(
                    "unknown detect option '{other}'"
                )));
            }
        }
        idx += 1;
    }

    if args.input.as_os_str().is_empty() {
        return Err(CliError::invalid_input("detect requires --input <path>"));
    }

    Ok(args)
}

fn parse_run_args(tokens: &[String]) -> Result<RunArgs, CliError> {
    let mut pipeline = PathBuf::new();
    let mut input = PathBuf::new();
    let mut output: Option<PathBuf> = None;

    let mut idx = 0usize;
    while idx < tokens.len() {
        let (flag, inline_value) = split_flag(tokens[idx].as_str())?;
        match flag {
            "--pipeline" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                pipeline = PathBuf::from(raw);
            }
            "--input" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                input = PathBuf::from(raw);
            }
            "--output" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                output = Some(PathBuf::from(raw));
            }
            other => {
                return Err(CliError::invalid_input(format!(
                    "unknown run option '{other}'"
                )));
            }
        }
        idx += 1;
    }

    if pipeline.as_os_str().is_empty() {
        return Err(CliError::invalid_input("run requires --pipeline <path>"));
    }
    if input.as_os_str().is_empty() {
        return Err(CliError::invalid_input("run requires --input <path>"));
    }

    Ok(RunArgs {
        pipeline,
        input,
        output,
    })
}

fn parse_doctor_args(tokens: &[String]) -> Result<DoctorArgs, CliError> {
    let mut args = DoctorArgs::default();
    let mut idx = 0usize;
    while idx < tokens.len() {
        let (flag, inline_value) = split_flag(tokens[idx].as_str())?;
        match flag {
            "--objective" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.objective = ObjectiveArg::parse(raw.as_str())?;
            }
            "--min-confidence" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.min_confidence = parse_f64_arg(raw.as_str(), flag)?;
            }
            "--allow-abstain" => {
                ensure_no_inline_value(flag, inline_value)?;
                args.allow_abstain = true;
            }
            "--online" => {
                ensure_no_inline_value(flag, inline_value)?;
                args.online = true;
            }
            "--constraints" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.constraints = Some(PathBuf::from(raw));
            }
            "--input" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.input = PathBuf::from(raw);
            }
            "--output" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.output = Some(PathBuf::from(raw));
            }
            other => {
                return Err(CliError::invalid_input(format!(
                    "unknown doctor option '{other}'"
                )));
            }
        }
        idx += 1;
    }

    if args.input.as_os_str().is_empty() {
        return Err(CliError::invalid_input("doctor requires --input <path>"));
    }

    Ok(args)
}

fn parse_eval_args(tokens: &[String]) -> Result<EvalArgs, CliError> {
    let mut args = EvalArgs::default();
    let mut idx = 0usize;
    while idx < tokens.len() {
        let (flag, inline_value) = split_flag(tokens[idx].as_str())?;
        match flag {
            "--predictions" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.predictions = PathBuf::from(raw);
            }
            "--ground-truth" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.ground_truth = PathBuf::from(raw);
            }
            "--tolerance" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.tolerance = parse_usize_arg(raw.as_str(), flag)?;
            }
            "--output" => {
                let raw = take_flag_value(flag, inline_value, tokens, &mut idx)?;
                args.output = Some(PathBuf::from(raw));
            }
            other => {
                return Err(CliError::invalid_input(format!(
                    "unknown eval option '{other}'"
                )));
            }
        }
        idx += 1;
    }

    if args.predictions.as_os_str().is_empty() {
        return Err(CliError::invalid_input(
            "eval requires --predictions <path>",
        ));
    }
    if args.ground_truth.as_os_str().is_empty() {
        return Err(CliError::invalid_input(
            "eval requires --ground-truth <path>",
        ));
    }

    Ok(args)
}

fn split_flag(token: &str) -> Result<(&str, Option<String>), CliError> {
    if !token.starts_with("--") {
        return Err(CliError::invalid_input(format!(
            "unexpected positional argument '{token}'; expected --flag value"
        )));
    }
    if let Some((flag, value)) = token.split_once('=') {
        return Ok((flag, Some(value.to_string())));
    }
    Ok((token, None))
}

fn take_flag_value(
    flag: &str,
    inline_value: Option<String>,
    tokens: &[String],
    idx: &mut usize,
) -> Result<String, CliError> {
    if let Some(value) = inline_value {
        return Ok(value);
    }

    *idx += 1;
    let value = tokens
        .get(*idx)
        .ok_or_else(|| CliError::invalid_input(format!("{flag} requires a value")))?;
    if value.starts_with("--") {
        return Err(CliError::invalid_input(format!(
            "{flag} requires a value, but got option '{value}'"
        )));
    }
    Ok(value.clone())
}

fn ensure_no_inline_value(flag: &str, inline_value: Option<String>) -> Result<(), CliError> {
    if inline_value.is_some() {
        return Err(CliError::invalid_input(format!(
            "{flag} does not accept a value"
        )));
    }
    Ok(())
}

fn parse_usize_arg(raw: &str, flag: &str) -> Result<usize, CliError> {
    raw.parse::<usize>().map_err(|_| {
        CliError::invalid_input(format!(
            "{flag} expects a non-negative integer, got '{raw}'"
        ))
    })
}

fn parse_u64_arg(raw: &str, flag: &str) -> Result<u64, CliError> {
    raw.parse::<u64>().map_err(|_| {
        CliError::invalid_input(format!(
            "{flag} expects a non-negative integer, got '{raw}'"
        ))
    })
}

fn parse_f64_arg(raw: &str, flag: &str) -> Result<f64, CliError> {
    raw.parse::<f64>()
        .map_err(|_| CliError::invalid_input(format!("{flag} expects a number, got '{raw}'")))
}

fn print_version() {
    println!("cpd {}", env!("CARGO_PKG_VERSION"));
}

fn print_root_help() {
    println!(
        "cpd {}\n\nUSAGE:\n  cpd <COMMAND> [OPTIONS]\n\nCOMMANDS:\n  detect   Run offline detection from CLI flags\n  run      Execute a pipeline spec JSON against input data\n  doctor   Generate doctor recommendations for input data\n  eval     Evaluate predictions against ground truth\n\nGLOBAL OPTIONS:\n  -h, --help      Show help\n  -V, --version   Show version\n\nRun 'cpd <COMMAND> --help' for subcommand options.",
        env!("CARGO_PKG_VERSION")
    );
}

fn print_command_help(command: &str) -> Result<(), CliError> {
    match command {
        "detect" => {
            println!(
                "USAGE:\n  cpd detect --input <path> [OPTIONS]\n\nOPTIONS:\n  --algorithm <pelt|binseg|fpop|wbs>                                   Default: pelt\n  --cost <ar|cosine|l1_median|l2|normal|normal_full_cov|nig|rank>      Default: l2\n  --penalty <bic|aic|manual>                                            Default: bic\n  --penalty-value <float>                                               Required when --penalty=manual\n  --k <usize>                                                           Use KnownK stopping\n  --seed <u64>                                                          WBS seed only\n  --min-segment-len <usize>\n  --max-change-points <usize>\n  --max-depth <usize>\n  --jump <usize>\n  --input <path>                                                        Required (.csv or .npy)\n  --output <path>                                                       Write JSON output to file"
            );
            Ok(())
        }
        "run" => {
            println!(
                "USAGE:\n  cpd run --pipeline <spec.json> --input <path> [OPTIONS]\n\nOPTIONS:\n  --pipeline <path>                  Required pipeline JSON\n  --input <path>                     Required input (.csv or .npy)\n  --output <path>                    Write JSON output to file"
            );
            Ok(())
        }
        "doctor" => {
            println!(
                "USAGE:\n  cpd doctor --input <path> [OPTIONS]\n\nOPTIONS:\n  --objective <balanced|speed|accuracy|robustness>   Default: balanced\n  --min-confidence <float>                             Default: 0.2\n  --allow-abstain\n  --online\n  --constraints <path>                                 Optional constraints JSON patch\n  --input <path>                                       Required (.csv or .npy)\n  --output <path>                                      Write JSON output to file"
            );
            Ok(())
        }
        "eval" => {
            println!(
                "USAGE:\n  cpd eval --predictions <path> --ground-truth <path> [OPTIONS]\n\nOPTIONS:\n  --predictions <path>               Required predictions JSON\n  --ground-truth <path>              Required ground-truth JSON\n  --tolerance <usize>                Default: 1 (offline mode)\n  --output <path>                    Write JSON output to file"
            );
            Ok(())
        }
        _ => Err(CliError::invalid_input(format!(
            "unknown command '{command}'; expected one of: detect, run, doctor, eval"
        ))),
    }
}

fn handle_detect(args: DetectArgs) -> Result<(), CliError> {
    let input = load_series(args.input.as_path())?;
    let view = input.as_view()?;
    let pipeline = build_detect_pipeline(&args)?;
    let result = run_pipeline(&view, &pipeline)?;

    write_json_output(
        &DetectOutput {
            command: "detect",
            input: input.summary(),
            pipeline,
            result,
        },
        args.output.as_deref(),
    )
}

fn handle_run(args: RunArgs) -> Result<(), CliError> {
    let input = load_series(args.input.as_path())?;
    let view = input.as_view()?;
    let pipeline = load_pipeline_spec(args.pipeline.as_path())?;
    let result = run_pipeline(&view, &pipeline)?;

    write_json_output(
        &RunOutput {
            command: "run",
            input: input.summary(),
            pipeline,
            result,
        },
        args.output.as_deref(),
    )
}

fn handle_doctor(args: DoctorArgs) -> Result<(), CliError> {
    let input = load_series(args.input.as_path())?;
    let view = input.as_view()?;
    let constraints = args
        .constraints
        .as_deref()
        .map(load_constraints)
        .transpose()?;

    let recommendations = recommend(
        &view,
        args.objective.as_objective(),
        args.online,
        constraints,
        args.min_confidence,
        args.allow_abstain,
    )?;

    write_json_output(
        &DoctorOutput {
            command: "doctor",
            input: input.summary(),
            objective: args.objective.as_str().to_string(),
            online: args.online,
            min_confidence: args.min_confidence,
            allow_abstain: args.allow_abstain,
            recommendations: recommendations_to_output(recommendations),
        },
        args.output.as_deref(),
    )
}

fn handle_eval(args: EvalArgs) -> Result<(), CliError> {
    let predictions = read_json_value(args.predictions.as_path())?;
    let truth = read_json_value(args.ground_truth.as_path())?;

    if has_steps_input(&predictions) {
        let steps = extract_online_steps(&predictions)?;
        let true_change_points = extract_true_change_points(&truth)?;
        let metrics = online_metrics(steps.as_slice(), true_change_points.as_slice())?;
        return write_json_output(
            &OnlineEvalOutput {
                command: "eval",
                mode: "online",
                metrics: OnlineMetricsOutput {
                    mean_detection_delay: metrics.mean_detection_delay,
                    false_alarm_rate: metrics.false_alarm_rate,
                    arl0: metrics.arl0,
                    arl1: metrics.arl1,
                    detected_changes: metrics.detected_changes,
                    missed_changes: metrics.missed_changes,
                    false_alerts: metrics.false_alerts,
                    total_alerts: metrics.total_alerts,
                    roc_curve: metrics
                        .roc_curve
                        .into_iter()
                        .map(|point| RocPointOutput {
                            threshold: point.threshold,
                            true_positive_rate: point.true_positive_rate,
                            false_positive_rate: point.false_positive_rate,
                            detected_changes: point.detected_changes,
                            false_alerts: point.false_alerts,
                        })
                        .collect(),
                },
            },
            args.output.as_deref(),
        );
    }

    let detected = extract_offline_result(&predictions)?;
    let truth_result = extract_offline_result(&truth)?;
    let metrics = offline_metrics(&detected, &truth_result, args.tolerance)?;

    write_json_output(
        &OfflineEvalOutput {
            command: "eval",
            mode: "offline",
            tolerance: args.tolerance,
            metrics: OfflineMetricsOutput {
                f1: F1MetricsOutput {
                    true_positives: metrics.f1.true_positives,
                    false_positives: metrics.f1.false_positives,
                    false_negatives: metrics.f1.false_negatives,
                    precision: metrics.f1.precision,
                    recall: metrics.f1.recall,
                    f1: metrics.f1.f1,
                },
                hausdorff_distance: metrics.hausdorff_distance,
                rand_index: metrics.rand_index,
                annotation_error: metrics.annotation_error,
            },
        },
        args.output.as_deref(),
    )
}

fn build_detect_pipeline(args: &DetectArgs) -> Result<PipelineSpec, CliError> {
    let stopping = resolve_stopping(
        args.k,
        args.penalty,
        args.penalty_explicit,
        args.penalty_value,
    )?;
    let constraints = constraints_from_detect_args(args);

    let detector = match args.algorithm {
        AlgorithmArg::Pelt => {
            if args.seed.is_some() {
                return Err(CliError::invalid_input(
                    "--seed is only supported when --algorithm=wbs",
                ));
            }
            OfflineDetectorConfig::Pelt(PeltConfig {
                stopping: stopping.clone(),
                ..PeltConfig::default()
            })
        }
        AlgorithmArg::Binseg => {
            if args.seed.is_some() {
                return Err(CliError::invalid_input(
                    "--seed is only supported when --algorithm=wbs",
                ));
            }
            OfflineDetectorConfig::BinSeg(BinSegConfig {
                stopping: stopping.clone(),
                ..BinSegConfig::default()
            })
        }
        AlgorithmArg::Fpop => {
            if args.seed.is_some() {
                return Err(CliError::invalid_input(
                    "--seed is only supported when --algorithm=wbs",
                ));
            }
            if !matches!(args.cost, CostArg::L2) {
                return Err(CliError::invalid_input(
                    "--algorithm=fpop requires --cost=l2",
                ));
            }
            OfflineDetectorConfig::Fpop(FpopConfig {
                stopping: stopping.clone(),
                ..FpopConfig::default()
            })
        }
        AlgorithmArg::Wbs => {
            let mut config = WbsConfig {
                stopping: stopping.clone(),
                ..WbsConfig::default()
            };
            if let Some(seed) = args.seed {
                config.seed = seed;
            }
            OfflineDetectorConfig::Wbs(config)
        }
    };

    Ok(PipelineSpec {
        detector: DetectorConfig::Offline(detector),
        cost: match args.cost {
            CostArg::Ar => CostConfig::Ar,
            CostArg::Cosine => CostConfig::Cosine,
            CostArg::L1Median => CostConfig::L1Median,
            CostArg::L2 => CostConfig::L2,
            CostArg::Normal => CostConfig::Normal,
            CostArg::NormalFullCov => CostConfig::NormalFullCov,
            CostArg::Nig => CostConfig::Nig,
            CostArg::Rank => CostConfig::Rank,
        },
        preprocess: None,
        constraints,
        stopping: Some(stopping),
        seed: args.seed,
    })
}

fn resolve_stopping(
    k: Option<usize>,
    penalty: PenaltyArg,
    penalty_explicit: bool,
    penalty_value: Option<f64>,
) -> Result<Stopping, CliError> {
    if let Some(k_value) = k {
        if k_value == 0 {
            return Err(CliError::invalid_input("--k must be >= 1 when provided"));
        }
        if penalty_explicit {
            return Err(CliError::invalid_input(
                "--penalty cannot be combined with --k",
            ));
        }
        if penalty_value.is_some() {
            return Err(CliError::invalid_input(
                "--penalty-value cannot be combined with --k",
            ));
        }
        return Ok(Stopping::KnownK(k_value));
    }

    let parsed_penalty = match penalty {
        PenaltyArg::Bic => {
            if penalty_value.is_some() {
                return Err(CliError::invalid_input(
                    "--penalty-value requires --penalty=manual",
                ));
            }
            Penalty::BIC
        }
        PenaltyArg::Aic => {
            if penalty_value.is_some() {
                return Err(CliError::invalid_input(
                    "--penalty-value requires --penalty=manual",
                ));
            }
            Penalty::AIC
        }
        PenaltyArg::Manual => {
            let value = penalty_value.ok_or_else(|| {
                CliError::invalid_input("--penalty=manual requires --penalty-value")
            })?;
            if !value.is_finite() || value <= 0.0 {
                return Err(CliError::invalid_input(
                    "--penalty-value must be finite and > 0.0",
                ));
            }
            Penalty::Manual(value)
        }
    };

    Ok(Stopping::Penalized(parsed_penalty))
}

fn constraints_from_detect_args(args: &DetectArgs) -> Constraints {
    let mut constraints = Constraints::default();
    if let Some(min_segment_len) = args.min_segment_len {
        constraints.min_segment_len = min_segment_len;
    }
    if let Some(max_change_points) = args.max_change_points {
        constraints.max_change_points = Some(max_change_points);
    }
    if let Some(max_depth) = args.max_depth {
        constraints.max_depth = Some(max_depth);
    }
    if let Some(jump) = args.jump {
        constraints.jump = jump;
    }
    constraints
}

fn load_series(path: &Path) -> Result<LoadedSeries, CliError> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .ok_or_else(|| {
            CliError::not_supported(format!(
                "unable to infer input format for '{}'; expected .csv or .npy",
                path.display()
            ))
        })?;

    match extension.as_str() {
        "csv" => {
            let raw = fs::read_to_string(path).map_err(|source| {
                CliError::io(format!("failed to read '{}'", path.display()), source)
            })?;
            let (values, n, d) = parse_csv_data(raw.as_str())?;
            Ok(LoadedSeries {
                path: path.to_path_buf(),
                format: "csv",
                values,
                n,
                d,
            })
        }
        "npy" => {
            let bytes = fs::read(path).map_err(|source| {
                CliError::io(format!("failed to read '{}'", path.display()), source)
            })?;
            let (values, n, d) = parse_npy_bytes(bytes.as_slice())?;
            Ok(LoadedSeries {
                path: path.to_path_buf(),
                format: "npy",
                values,
                n,
                d,
            })
        }
        _ => Err(CliError::not_supported(format!(
            "unsupported input format '{}'; expected .csv or .npy",
            extension
        ))),
    }
}

fn parse_csv_data(raw: &str) -> Result<(Vec<f64>, usize, usize), CliError> {
    let rows = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();

    if rows.is_empty() {
        return Err(CliError::invalid_input("CSV input is empty"));
    }

    match parse_csv_rows(rows.as_slice()) {
        Ok(parsed) => Ok(parsed),
        Err(err) => {
            if rows.len() > 1 && first_row_looks_like_header(rows[0], rows[1]) {
                if let Ok(without_header) = parse_csv_rows(&rows[1..]) {
                    return Ok(without_header);
                }
            }
            Err(err)
        }
    }
}

fn parse_csv_rows(rows: &[&str]) -> Result<(Vec<f64>, usize, usize), CliError> {
    let mut values = Vec::<f64>::new();
    let mut expected_cols: Option<usize> = None;

    for (row_idx, row) in rows.iter().enumerate() {
        let cells = row.split(',').map(str::trim).collect::<Vec<_>>();

        if cells.is_empty() {
            return Err(CliError::invalid_input(format!(
                "CSV row {} is empty",
                row_idx + 1
            )));
        }

        if let Some(cols) = expected_cols {
            if cells.len() != cols {
                return Err(CliError::invalid_input(format!(
                    "CSV row {} has {} columns but expected {}",
                    row_idx + 1,
                    cells.len(),
                    cols
                )));
            }
        } else {
            expected_cols = Some(cells.len());
        }

        for (col_idx, cell) in cells.iter().enumerate() {
            if cell.is_empty() {
                return Err(CliError::invalid_input(format!(
                    "CSV row {} column {} is empty",
                    row_idx + 1,
                    col_idx + 1
                )));
            }

            let value = cell.parse::<f64>().map_err(|_| {
                CliError::invalid_input(format!(
                    "CSV row {} column {} is not a valid float: '{}'",
                    row_idx + 1,
                    col_idx + 1,
                    cell
                ))
            })?;
            values.push(value);
        }
    }

    let d = expected_cols.ok_or_else(|| CliError::invalid_input("CSV input is empty"))?;
    let n = rows.len();
    Ok((values, n, d))
}

fn first_row_looks_like_header(first_row: &str, second_row: &str) -> bool {
    let first_cells = first_row.split(',').map(str::trim).collect::<Vec<_>>();
    let second_cells = second_row.split(',').map(str::trim).collect::<Vec<_>>();

    if first_cells.is_empty()
        || first_cells.len() != second_cells.len()
        || first_cells.iter().any(|cell| cell.is_empty())
        || second_cells.iter().any(|cell| cell.is_empty())
    {
        return false;
    }

    let first_all_non_numeric = first_cells.iter().all(|cell| cell.parse::<f64>().is_err());
    let second_all_numeric = second_cells.iter().all(|cell| cell.parse::<f64>().is_ok());

    first_all_non_numeric && second_all_numeric
}

struct ParsedNpyHeader {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
enum ByteOrder {
    Little,
    Big,
}

fn parse_npy_bytes(bytes: &[u8]) -> Result<(Vec<f64>, usize, usize), CliError> {
    const MAGIC: &[u8; 6] = b"\x93NUMPY";

    if bytes.len() < 10 {
        return Err(CliError::invalid_input(
            "NPY input is too short to contain a valid header",
        ));
    }
    if &bytes[..6] != MAGIC {
        return Err(CliError::invalid_input(
            "invalid NPY magic; expected '\\x93NUMPY'",
        ));
    }

    let major = bytes[6];
    let (header_offset, header_len) = match major {
        1 => {
            let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (10usize, header_len)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(CliError::invalid_input(
                    "NPY header is truncated for version >= 2",
                ));
            }
            let header_len =
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (12usize, header_len)
        }
        other => {
            return Err(CliError::not_supported(format!(
                "unsupported NPY version {other}; expected major version 1, 2, or 3"
            )));
        }
    };

    let header_end = header_offset
        .checked_add(header_len)
        .ok_or_else(|| CliError::invalid_input("NPY header length overflow"))?;
    if header_end > bytes.len() {
        return Err(CliError::invalid_input(
            "NPY header exceeds file length; file is truncated",
        ));
    }

    let header_text = std::str::from_utf8(&bytes[header_offset..header_end])
        .map_err(|_| CliError::invalid_input("NPY header is not valid UTF-8"))?;
    let header = parse_npy_header_text(header_text)?;

    let (n, d) = match header.shape.as_slice() {
        [n] => (*n, 1usize),
        [n, d] => (*n, *d),
        _ => {
            return Err(CliError::not_supported(format!(
                "NPY shape {:?} is unsupported; expected 1D or 2D array",
                header.shape
            )));
        }
    };

    if n == 0 || d == 0 {
        return Err(CliError::invalid_input(format!(
            "NPY shape {:?} has zero-sized dimension; expected n>=1 and d>=1",
            header.shape
        )));
    }

    let element_count = n
        .checked_mul(d)
        .ok_or_else(|| CliError::invalid_input("NPY shape overflow (n*d exceeds usize)"))?;
    let (byte_order, element_width) = parse_npy_descr(header.descr.as_str())?;

    let payload = &bytes[header_end..];
    let expected_payload_len = element_count
        .checked_mul(element_width)
        .ok_or_else(|| CliError::invalid_input("NPY payload length overflow"))?;
    if payload.len() != expected_payload_len {
        return Err(CliError::invalid_input(format!(
            "NPY payload length mismatch: got {}, expected {} for shape {:?} and descr '{}'",
            payload.len(),
            expected_payload_len,
            header.shape,
            header.descr
        )));
    }

    let mut values = Vec::<f64>::with_capacity(element_count);
    match element_width {
        4 => {
            for chunk in payload.chunks_exact(4) {
                let mut raw = [0u8; 4];
                raw.copy_from_slice(chunk);
                let value = match byte_order {
                    ByteOrder::Little => f32::from_le_bytes(raw),
                    ByteOrder::Big => f32::from_be_bytes(raw),
                };
                values.push(f64::from(value));
            }
        }
        8 => {
            for chunk in payload.chunks_exact(8) {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(chunk);
                let value = match byte_order {
                    ByteOrder::Little => f64::from_le_bytes(raw),
                    ByteOrder::Big => f64::from_be_bytes(raw),
                };
                values.push(value);
            }
        }
        _ => {
            return Err(CliError::not_supported(format!(
                "unsupported NPY element width {element_width}; expected 4 or 8 bytes"
            )));
        }
    }

    if header.fortran_order && d > 1 {
        let mut c_values = vec![0.0f64; element_count];
        for row in 0..n {
            for col in 0..d {
                c_values[row * d + col] = values[col * n + row];
            }
        }
        return Ok((c_values, n, d));
    }

    Ok((values, n, d))
}

fn parse_npy_header_text(header: &str) -> Result<ParsedNpyHeader, CliError> {
    let descr = extract_header_string(header, "descr")?;
    let fortran_order = extract_header_bool(header, "fortran_order")?;
    let shape = extract_header_shape(header, "shape")?;

    Ok(ParsedNpyHeader {
        descr,
        fortran_order,
        shape,
    })
}

fn extract_header_field<'a>(header: &'a str, key: &str) -> Result<&'a str, CliError> {
    let marker = format!("'{key}':");
    let start = header.find(marker.as_str()).ok_or_else(|| {
        CliError::invalid_input(format!("NPY header missing required key '{key}'"))
    })?;
    Ok(header[start + marker.len()..].trim_start())
}

fn extract_header_string(header: &str, key: &str) -> Result<String, CliError> {
    let rest = extract_header_field(header, key)?;
    let Some(after_quote) = rest.strip_prefix('\'') else {
        return Err(CliError::invalid_input(format!(
            "NPY header field '{key}' must be a quoted string"
        )));
    };
    let end = after_quote.find('\'').ok_or_else(|| {
        CliError::invalid_input(format!("NPY header field '{key}' has unterminated string"))
    })?;
    Ok(after_quote[..end].to_string())
}

fn extract_header_bool(header: &str, key: &str) -> Result<bool, CliError> {
    let rest = extract_header_field(header, key)?;
    if rest.starts_with("True") {
        Ok(true)
    } else if rest.starts_with("False") {
        Ok(false)
    } else {
        Err(CliError::invalid_input(format!(
            "NPY header field '{key}' must be True or False"
        )))
    }
}

fn extract_header_shape(header: &str, key: &str) -> Result<Vec<usize>, CliError> {
    let rest = extract_header_field(header, key)?;
    let Some(after_paren) = rest.strip_prefix('(') else {
        return Err(CliError::invalid_input(format!(
            "NPY header field '{key}' must start with '('"
        )));
    };
    let end = after_paren.find(')').ok_or_else(|| {
        CliError::invalid_input(format!("NPY header field '{key}' has unterminated tuple"))
    })?;
    let shape_body = &after_paren[..end];
    let dims = shape_body
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>().map_err(|_| {
                CliError::invalid_input(format!(
                    "NPY shape entry '{part}' is not a valid non-negative integer"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if dims.is_empty() {
        return Err(CliError::not_supported(
            "NPY scalar arrays are unsupported; expected shape with 1 or 2 dimensions",
        ));
    }

    Ok(dims)
}

fn parse_npy_descr(descr: &str) -> Result<(ByteOrder, usize), CliError> {
    let trimmed = descr.trim();
    if trimmed.is_empty() {
        return Err(CliError::invalid_input("NPY descr is empty"));
    }

    let (byte_order, dtype) = match trimmed.chars().next().unwrap_or('<') {
        '<' => (ByteOrder::Little, &trimmed[1..]),
        '>' => (ByteOrder::Big, &trimmed[1..]),
        '|' => (ByteOrder::Little, &trimmed[1..]),
        '=' => {
            let order = if cfg!(target_endian = "little") {
                ByteOrder::Little
            } else {
                ByteOrder::Big
            };
            (order, &trimmed[1..])
        }
        'f' => {
            let order = if cfg!(target_endian = "little") {
                ByteOrder::Little
            } else {
                ByteOrder::Big
            };
            (order, trimmed)
        }
        _ => {
            return Err(CliError::not_supported(format!(
                "unsupported NPY descr '{trimmed}'; expected floating-point dtype f4 or f8"
            )));
        }
    };

    let width = match dtype {
        "f4" => 4,
        "f8" => 8,
        _ => {
            return Err(CliError::not_supported(format!(
                "unsupported NPY dtype '{dtype}'; expected f4 or f8"
            )));
        }
    };

    Ok((byte_order, width))
}

fn load_constraints(path: &Path) -> Result<Constraints, CliError> {
    let value = read_json_value(path)?;
    parse_constraints_patch(Some(&value))
}

fn load_pipeline_spec(path: &Path) -> Result<PipelineSpec, CliError> {
    let raw = fs::read_to_string(path)
        .map_err(|source| CliError::io(format!("failed to read '{}'", path.display()), source))?;
    parse_pipeline_spec_document(raw.as_str())
}

fn parse_pipeline_spec_document(raw: &str) -> Result<PipelineSpec, CliError> {
    let direct_parse_error = match serde_json::from_str::<PipelineSpec>(raw) {
        Ok(spec) => {
            validate_pipeline_spec(&spec)?;
            return Ok(spec);
        }
        Err(err) => err,
    };

    let value: Value = serde_json::from_str(raw)
        .map_err(|source| CliError::json("invalid pipeline JSON", source))?;
    let fallback_target = match value.as_object() {
        Some(obj) if obj.get("kind") == Some(&Value::String("pipeline_spec".to_string())) => {
            obj.get("payload").unwrap_or(&value)
        }
        _ => &value,
    };

    let parsed = parse_simple_pipeline_spec(fallback_target).map_err(|fallback_error| {
        CliError::invalid_input(format!(
            "failed to parse pipeline spec: serde parser error: {direct_parse_error}; fallback parser error: {fallback_error}"
        ))
    })?;
    validate_pipeline_spec(&parsed)?;
    Ok(parsed)
}

fn parse_simple_pipeline_spec(value: &Value) -> Result<PipelineSpec, CliError> {
    let object = value.as_object().ok_or_else(|| {
        CliError::invalid_input("pipeline spec must be a JSON object or pipeline_spec envelope")
    })?;

    let detector_value = object
        .get("detector")
        .ok_or_else(|| CliError::invalid_input("pipeline.detector is required"))?;
    let mut detector = parse_simple_detector(detector_value)?;

    let cost = parse_cost_value(object.get("cost"), "pipeline.cost")?;
    let constraints = parse_constraints_patch(object.get("constraints"))?;
    let stopping = match object.get("stopping") {
        Some(stopping) if !stopping.is_null() => parse_pipeline_stopping_compat(stopping)?,
        _ => detector_stopping(&detector),
    };
    let seed = match object.get("seed") {
        Some(seed) if !seed.is_null() => Some(parse_u64(seed, "pipeline.seed")?),
        _ => detector_seed(&detector),
    };

    if let Some(preprocess) = object.get("preprocess")
        && !preprocess.is_null()
    {
        return Err(CliError::not_supported(
            "pipeline.preprocess is not supported by this cpd-cli build",
        ));
    }

    if matches!(&detector, OfflineDetectorConfig::Fpop(_)) && !matches!(cost, CostConfig::L2) {
        return Err(CliError::invalid_input(
            "pipeline.detector='fpop' requires pipeline.cost='l2'",
        ));
    }

    apply_pipeline_controls(&mut detector, &stopping, seed)?;

    Ok(PipelineSpec {
        detector: DetectorConfig::Offline(detector),
        cost,
        preprocess: None,
        constraints,
        stopping: Some(stopping),
        seed,
    })
}

fn validate_pipeline_spec(pipeline: &PipelineSpec) -> Result<(), CliError> {
    if matches!(
        &pipeline.detector,
        DetectorConfig::Offline(OfflineDetectorConfig::Fpop(_))
    ) && !matches!(pipeline.cost, CostConfig::L2)
    {
        return Err(CliError::invalid_input(
            "pipeline.detector='fpop' requires pipeline.cost='l2'",
        ));
    }
    Ok(())
}

fn parse_simple_detector(value: &Value) -> Result<OfflineDetectorConfig, CliError> {
    if let Some(kind) = value.as_str() {
        return parse_detector_kind(kind, None, "pipeline.detector");
    }

    let dict = value
        .as_object()
        .ok_or_else(|| CliError::invalid_input("pipeline.detector must be a string or object"))?;

    if let Some(kind_value) = dict.get("kind") {
        let kind = parse_string(kind_value, "pipeline.detector.kind")?;
        return parse_detector_kind(kind.as_str(), Some(dict), "pipeline.detector");
    }

    let Some(offline_value) = dict.get("Offline") else {
        return Err(CliError::invalid_input(
            "pipeline.detector object must include either 'kind' or serde key 'Offline'",
        ));
    };

    let offline_dict = offline_value
        .as_object()
        .ok_or_else(|| CliError::invalid_input("pipeline.detector['Offline'] must be an object"))?;

    if offline_dict.len() != 1 {
        return Err(CliError::invalid_input(
            "pipeline.detector['Offline'] must contain exactly one variant key",
        ));
    }

    let (variant, config_value) = offline_dict
        .iter()
        .next()
        .expect("offline_dict length checked");
    let config_dict = config_value.as_object().ok_or_else(|| {
        CliError::invalid_input(format!(
            "pipeline.detector['Offline']['{variant}'] must be an object"
        ))
    })?;

    parse_detector_kind(
        variant.as_str(),
        Some(config_dict),
        "pipeline.detector['Offline']",
    )
}

fn parse_detector_kind(
    kind: &str,
    config_dict: Option<&Map<String, Value>>,
    context: &str,
) -> Result<OfflineDetectorConfig, CliError> {
    match kind.to_ascii_lowercase().as_str() {
        "pelt" => {
            let config = match config_dict {
                Some(dict) => parse_pelt_config(dict, context)?,
                None => PeltConfig::default(),
            };
            Ok(OfflineDetectorConfig::Pelt(config))
        }
        "binseg" => {
            let config = match config_dict {
                Some(dict) => parse_binseg_config(dict, context)?,
                None => BinSegConfig::default(),
            };
            Ok(OfflineDetectorConfig::BinSeg(config))
        }
        "fpop" => {
            let config = match config_dict {
                Some(dict) => parse_fpop_config(dict, context)?,
                None => FpopConfig::default(),
            };
            Ok(OfflineDetectorConfig::Fpop(config))
        }
        "wbs" => {
            let config = match config_dict {
                Some(dict) => parse_wbs_config(dict, context)?,
                None => WbsConfig::default(),
            };
            Ok(OfflineDetectorConfig::Wbs(config))
        }
        _ => Err(CliError::invalid_input(format!(
            "unsupported {context} kind '{kind}'; expected one of: 'pelt', 'binseg', 'fpop', 'wbs'"
        ))),
    }
}

fn parse_pelt_config(dict: &Map<String, Value>, context: &str) -> Result<PeltConfig, CliError> {
    let mut config = PeltConfig::default();
    for (key, value) in dict {
        match key.as_str() {
            "kind" => {}
            "stopping" => config.stopping = parse_pipeline_stopping_compat(value)?,
            "params_per_segment" => {
                config.params_per_segment =
                    parse_usize(value, "pipeline.detector.params_per_segment")?
            }
            "cancel_check_every" => {
                config.cancel_check_every =
                    parse_usize(value, "pipeline.detector.cancel_check_every")?
            }
            _ => {
                return Err(CliError::invalid_input(format!(
                    "unsupported key '{key}' in {context} for detector kind='pelt'"
                )));
            }
        }
    }
    Ok(config)
}

fn parse_binseg_config(dict: &Map<String, Value>, context: &str) -> Result<BinSegConfig, CliError> {
    let mut config = BinSegConfig::default();
    for (key, value) in dict {
        match key.as_str() {
            "kind" => {}
            "stopping" => config.stopping = parse_pipeline_stopping_compat(value)?,
            "params_per_segment" => {
                config.params_per_segment =
                    parse_usize(value, "pipeline.detector.params_per_segment")?
            }
            "cancel_check_every" => {
                config.cancel_check_every =
                    parse_usize(value, "pipeline.detector.cancel_check_every")?
            }
            _ => {
                return Err(CliError::invalid_input(format!(
                    "unsupported key '{key}' in {context} for detector kind='binseg'"
                )));
            }
        }
    }
    Ok(config)
}

fn parse_fpop_config(dict: &Map<String, Value>, context: &str) -> Result<FpopConfig, CliError> {
    let mut config = FpopConfig::default();
    for (key, value) in dict {
        match key.as_str() {
            "kind" => {}
            "stopping" => config.stopping = parse_pipeline_stopping_compat(value)?,
            "params_per_segment" => {
                config.params_per_segment =
                    parse_usize(value, "pipeline.detector.params_per_segment")?
            }
            "cancel_check_every" => {
                config.cancel_check_every =
                    parse_usize(value, "pipeline.detector.cancel_check_every")?
            }
            _ => {
                return Err(CliError::invalid_input(format!(
                    "unsupported key '{key}' in {context} for detector kind='fpop'"
                )));
            }
        }
    }
    Ok(config)
}

fn parse_wbs_config(dict: &Map<String, Value>, context: &str) -> Result<WbsConfig, CliError> {
    let mut config = WbsConfig::default();
    for (key, value) in dict {
        match key.as_str() {
            "kind" => {}
            "stopping" => config.stopping = parse_pipeline_stopping_compat(value)?,
            "params_per_segment" => {
                config.params_per_segment =
                    parse_usize(value, "pipeline.detector.params_per_segment")?
            }
            "cancel_check_every" => {
                config.cancel_check_every =
                    parse_usize(value, "pipeline.detector.cancel_check_every")?
            }
            "num_intervals" => {
                config.num_intervals =
                    parse_optional_usize(value, "pipeline.detector.num_intervals")?
            }
            "interval_strategy" => {
                let raw = parse_string(value, "pipeline.detector.interval_strategy")?;
                config.interval_strategy = parse_interval_strategy(raw.as_str())?;
            }
            "seed" => config.seed = parse_u64(value, "pipeline.detector.seed")?,
            _ => {
                return Err(CliError::invalid_input(format!(
                    "unsupported key '{key}' in {context} for detector kind='wbs'"
                )));
            }
        }
    }
    Ok(config)
}

fn parse_interval_strategy(raw: &str) -> Result<WbsIntervalStrategy, CliError> {
    match raw.to_ascii_lowercase().as_str() {
        "random" => Ok(WbsIntervalStrategy::Random),
        "deterministic_grid" | "deterministicgrid" => Ok(WbsIntervalStrategy::DeterministicGrid),
        "stratified" => Ok(WbsIntervalStrategy::Stratified),
        _ => Err(CliError::invalid_input(format!(
            "unsupported pipeline.detector.interval_strategy '{raw}'; expected one of: 'random', 'deterministic_grid', 'stratified'"
        ))),
    }
}

fn parse_pipeline_stopping_compat(value: &Value) -> Result<Stopping, CliError> {
    let dict = value
        .as_object()
        .ok_or_else(|| CliError::invalid_input("pipeline.stopping must be an object"))?;

    let has_legacy_shape =
        dict.contains_key("n_bkps") || dict.contains_key("pen") || dict.contains_key("penalty");
    if has_legacy_shape {
        return parse_legacy_stopping(dict);
    }

    if dict.len() != 1 {
        return Err(CliError::invalid_input(
            "pipeline.stopping serde form must contain exactly one key: 'KnownK', 'Penalized', or 'PenaltyPath'",
        ));
    }

    let (key, stopping_value) = dict.iter().next().expect("dict length checked");
    match key.as_str() {
        "KnownK" => {
            let k = parse_usize(stopping_value, "pipeline.stopping.KnownK")?;
            if k == 0 {
                return Err(CliError::invalid_input(
                    "pipeline.stopping['KnownK'] must be >= 1",
                ));
            }
            Ok(Stopping::KnownK(k))
        }
        "Penalized" => Ok(Stopping::Penalized(parse_penalty_compat(
            stopping_value,
            "pipeline.stopping.Penalized",
        )?)),
        "PenaltyPath" => {
            let penalties = stopping_value.as_array().ok_or_else(|| {
                CliError::invalid_input("pipeline.stopping['PenaltyPath'] must be an array")
            })?;
            if penalties.is_empty() {
                return Err(CliError::invalid_input(
                    "pipeline.stopping['PenaltyPath'] must be non-empty",
                ));
            }
            let mut path = Vec::with_capacity(penalties.len());
            for penalty_value in penalties {
                path.push(parse_penalty_compat(
                    penalty_value,
                    "pipeline.stopping.PenaltyPath entry",
                )?);
            }
            Ok(Stopping::PenaltyPath(path))
        }
        _ => Err(CliError::invalid_input(format!(
            "unsupported pipeline.stopping key '{key}'; expected one of: 'KnownK', 'Penalized', 'PenaltyPath'"
        ))),
    }
}

fn parse_legacy_stopping(dict: &Map<String, Value>) -> Result<Stopping, CliError> {
    let n_bkps = dict
        .get("n_bkps")
        .map(|value| parse_optional_usize(value, "pipeline.stopping.n_bkps"))
        .transpose()?
        .flatten();

    let pen = dict
        .get("pen")
        .map(|value| parse_optional_f64(value, "pipeline.stopping.pen"))
        .transpose()?
        .flatten();

    let penalty = dict
        .get("penalty")
        .map(|value| parse_penalty_compat(value, "pipeline.stopping.penalty"))
        .transpose()?;

    if pen.is_some() && penalty.is_some() {
        return Err(CliError::invalid_input(
            "pipeline.stopping accepts at most one of 'pen' or 'penalty'",
        ));
    }

    let penalty = match (pen, penalty) {
        (Some(beta), None) => {
            if !beta.is_finite() || beta <= 0.0 {
                return Err(CliError::invalid_input(
                    "pipeline.stopping.pen must be finite and > 0.0",
                ));
            }
            Some(Penalty::Manual(beta))
        }
        (None, some_penalty) => some_penalty,
        (Some(_), Some(_)) => unreachable!(),
    };

    match (n_bkps, penalty) {
        (Some(_), Some(_)) => Err(CliError::invalid_input(
            "pipeline.stopping requires exactly one of n_bkps or (pen/penalty); got both",
        )),
        (None, None) => Err(CliError::invalid_input(
            "pipeline.stopping requires one of n_bkps, pen, or penalty",
        )),
        (Some(k), None) => {
            if k == 0 {
                return Err(CliError::invalid_input(
                    "pipeline.stopping.n_bkps must be >= 1",
                ));
            }
            Ok(Stopping::KnownK(k))
        }
        (None, Some(parsed_penalty)) => Ok(Stopping::Penalized(parsed_penalty)),
    }
}

fn parse_penalty_compat(value: &Value, context: &str) -> Result<Penalty, CliError> {
    if let Some(named) = value.as_str() {
        return match named.to_ascii_lowercase().as_str() {
            "bic" => Ok(Penalty::BIC),
            "aic" => Ok(Penalty::AIC),
            _ => Err(CliError::invalid_input(format!(
                "unsupported {context} '{named}'; expected 'bic', 'aic', or a positive number"
            ))),
        };
    }

    if let Some(beta) = value.as_f64() {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(CliError::invalid_input(format!(
                "{context} must be finite and > 0.0"
            )));
        }
        return Ok(Penalty::Manual(beta));
    }

    if let Some(dict) = value.as_object() {
        if dict.len() != 1 || !dict.contains_key("Manual") {
            return Err(CliError::invalid_input(format!(
                "{context} object form must be {{\"Manual\": <positive-number>}}"
            )));
        }
        let manual = dict.get("Manual").expect("contains_key checked");
        let beta = parse_f64(manual, &format!("{context}.Manual"))?;
        if !beta.is_finite() || beta <= 0.0 {
            return Err(CliError::invalid_input(format!(
                "{context}.Manual must be finite and > 0.0"
            )));
        }
        return Ok(Penalty::Manual(beta));
    }

    Err(CliError::invalid_input(format!(
        "{context} must be a string, number, or {{\"Manual\": number}}"
    )))
}

fn parse_cost_value(value: Option<&Value>, context: &str) -> Result<CostConfig, CliError> {
    let Some(value) = value else {
        return Ok(CostConfig::L2);
    };

    let raw = parse_string(value, context)?;
    let cost = match raw.to_ascii_lowercase().as_str() {
        "ar" => CostConfig::Ar,
        "cosine" => CostConfig::Cosine,
        "l1" | "l1_median" | "l1median" => CostConfig::L1Median,
        "l2" => CostConfig::L2,
        "normal" => CostConfig::Normal,
        "normal_full_cov" | "normal_fullcov" | "normalfullcov" => CostConfig::NormalFullCov,
        "nig" => CostConfig::Nig,
        "rank" => CostConfig::Rank,
        "none" => CostConfig::None,
        _ => {
            return Err(CliError::invalid_input(format!(
                "unsupported {context} '{raw}'; expected one of: 'ar', 'cosine', 'l1_median', 'l2', 'normal', 'normal_full_cov', 'nig', 'rank', 'none'"
            )));
        }
    };

    if matches!(cost, CostConfig::None) {
        return Err(CliError::invalid_input(
            "offline pipeline requires a concrete cost; 'none' is not allowed",
        ));
    }

    Ok(cost)
}

fn parse_constraints_patch(value: Option<&Value>) -> Result<Constraints, CliError> {
    let mut constraints = Constraints::default();
    let Some(value) = value else {
        return Ok(constraints);
    };
    let dict = value
        .as_object()
        .ok_or_else(|| CliError::invalid_input("constraints must be an object"))?;

    for (key, field_value) in dict {
        match key.as_str() {
            "min_segment_len" => {
                constraints.min_segment_len =
                    parse_usize(field_value, "constraints.min_segment_len")?
            }
            "jump" => constraints.jump = parse_usize(field_value, "constraints.jump")?,
            "max_change_points" => {
                constraints.max_change_points =
                    parse_optional_usize(field_value, "constraints.max_change_points")?
            }
            "max_depth" => {
                constraints.max_depth = parse_optional_usize(field_value, "constraints.max_depth")?
            }
            "candidate_splits" => {
                constraints.candidate_splits =
                    parse_optional_usize_vec(field_value, "constraints.candidate_splits")?
            }
            "time_budget_ms" => {
                constraints.time_budget_ms =
                    parse_optional_u64(field_value, "constraints.time_budget_ms")?
            }
            "max_cost_evals" => {
                constraints.max_cost_evals =
                    parse_optional_usize(field_value, "constraints.max_cost_evals")?
            }
            "memory_budget_bytes" => {
                constraints.memory_budget_bytes =
                    parse_optional_usize(field_value, "constraints.memory_budget_bytes")?
            }
            "max_cache_bytes" => {
                constraints.max_cache_bytes =
                    parse_optional_usize(field_value, "constraints.max_cache_bytes")?
            }
            "cache_policy" => {
                constraints.cache_policy = serde_json::from_value(field_value.clone())
                    .map_err(|source| CliError::json("invalid constraints.cache_policy", source))?;
            }
            "degradation_plan" => {
                constraints.degradation_plan = serde_json::from_value(field_value.clone())
                    .map_err(|source| {
                        CliError::json("invalid constraints.degradation_plan", source)
                    })?;
            }
            "allow_algorithm_fallback" => {
                constraints.allow_algorithm_fallback =
                    parse_bool(field_value, "constraints.allow_algorithm_fallback")?
            }
            _ => {
                return Err(CliError::invalid_input(format!(
                    "unsupported constraints key '{key}'"
                )));
            }
        }
    }

    Ok(constraints)
}

fn detector_stopping(detector: &OfflineDetectorConfig) -> Stopping {
    match detector {
        OfflineDetectorConfig::Pelt(config) => config.stopping.clone(),
        OfflineDetectorConfig::BinSeg(config) => config.stopping.clone(),
        OfflineDetectorConfig::Fpop(config) => config.stopping.clone(),
        OfflineDetectorConfig::Wbs(config) => config.stopping.clone(),
    }
}

fn detector_seed(detector: &OfflineDetectorConfig) -> Option<u64> {
    match detector {
        OfflineDetectorConfig::Wbs(config) => Some(config.seed),
        OfflineDetectorConfig::Pelt(_)
        | OfflineDetectorConfig::BinSeg(_)
        | OfflineDetectorConfig::Fpop(_) => None,
    }
}

fn apply_pipeline_controls(
    detector: &mut OfflineDetectorConfig,
    stopping: &Stopping,
    seed: Option<u64>,
) -> Result<(), CliError> {
    match detector {
        OfflineDetectorConfig::Pelt(config) => {
            config.stopping = stopping.clone();
            if seed.is_some() {
                return Err(CliError::invalid_input(
                    "pipeline.seed is only supported for detector='wbs'",
                ));
            }
        }
        OfflineDetectorConfig::BinSeg(config) => {
            config.stopping = stopping.clone();
            if seed.is_some() {
                return Err(CliError::invalid_input(
                    "pipeline.seed is only supported for detector='wbs'",
                ));
            }
        }
        OfflineDetectorConfig::Fpop(config) => {
            config.stopping = stopping.clone();
            if seed.is_some() {
                return Err(CliError::invalid_input(
                    "pipeline.seed is only supported for detector='wbs'",
                ));
            }
        }
        OfflineDetectorConfig::Wbs(config) => {
            config.stopping = stopping.clone();
            if let Some(seed_value) = seed {
                config.seed = seed_value;
            }
        }
    }
    Ok(())
}

fn parse_bool(value: &Value, context: &str) -> Result<bool, CliError> {
    value
        .as_bool()
        .ok_or_else(|| CliError::invalid_input(format!("{context} must be a boolean")))
}

fn parse_string(value: &Value, context: &str) -> Result<String, CliError> {
    value
        .as_str()
        .map(ToString::to_string)
        .ok_or_else(|| CliError::invalid_input(format!("{context} must be a string")))
}

fn parse_usize(value: &Value, context: &str) -> Result<usize, CliError> {
    let as_u64 = value.as_u64().ok_or_else(|| {
        CliError::invalid_input(format!("{context} must be a non-negative integer"))
    })?;
    usize::try_from(as_u64).map_err(|_| {
        CliError::invalid_input(format!(
            "{context} value {as_u64} does not fit in usize on this platform"
        ))
    })
}

fn parse_u64(value: &Value, context: &str) -> Result<u64, CliError> {
    value
        .as_u64()
        .ok_or_else(|| CliError::invalid_input(format!("{context} must be a non-negative integer")))
}

fn parse_f64(value: &Value, context: &str) -> Result<f64, CliError> {
    value
        .as_f64()
        .ok_or_else(|| CliError::invalid_input(format!("{context} must be a number")))
}

fn parse_optional_usize(value: &Value, context: &str) -> Result<Option<usize>, CliError> {
    if value.is_null() {
        return Ok(None);
    }
    parse_usize(value, context).map(Some)
}

fn parse_optional_u64(value: &Value, context: &str) -> Result<Option<u64>, CliError> {
    if value.is_null() {
        return Ok(None);
    }
    parse_u64(value, context).map(Some)
}

fn parse_optional_f64(value: &Value, context: &str) -> Result<Option<f64>, CliError> {
    if value.is_null() {
        return Ok(None);
    }
    parse_f64(value, context).map(Some)
}

fn parse_optional_usize_vec(value: &Value, context: &str) -> Result<Option<Vec<usize>>, CliError> {
    if value.is_null() {
        return Ok(None);
    }
    let values = value
        .as_array()
        .ok_or_else(|| CliError::invalid_input(format!("{context} must be an array or null")))?
        .iter()
        .enumerate()
        .map(|(index, item)| parse_usize(item, &format!("{context}[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(values))
}

fn recommendations_to_output(
    recommendations: Vec<Recommendation>,
) -> Vec<DoctorRecommendationOutput> {
    recommendations
        .into_iter()
        .enumerate()
        .map(|(idx, recommendation)| DoctorRecommendationOutput {
            rank: idx + 1,
            confidence: recommendation.confidence,
            confidence_interval: [
                recommendation.confidence_interval.0,
                recommendation.confidence_interval.1,
            ],
            abstain_reason: recommendation.abstain_reason,
            warnings: recommendation.warnings,
            pipeline: recommendation.pipeline,
            explanation: ExplanationOutput {
                summary: recommendation.explanation.summary,
                drivers: recommendation
                    .explanation
                    .drivers
                    .into_iter()
                    .map(|(key, value)| ObjectiveFitOutput { key, value })
                    .collect(),
                tradeoffs: recommendation.explanation.tradeoffs,
            },
            resource_estimate: ResourceEstimateOutput {
                time_complexity: recommendation.resource_estimate.time_complexity,
                memory_complexity: recommendation.resource_estimate.memory_complexity,
                relative_time_score: recommendation.resource_estimate.relative_time_score,
                relative_memory_score: recommendation.resource_estimate.relative_memory_score,
            },
            validation: recommendation.validation.map(|summary| ValidationOutput {
                method: summary.method,
                notes: summary.notes,
            }),
            objective_fit: recommendation
                .objective_fit
                .into_iter()
                .map(|(key, value)| ObjectiveFitOutput { key, value })
                .collect(),
        })
        .collect()
}

fn read_json_value(path: &Path) -> Result<Value, CliError> {
    let raw = fs::read_to_string(path)
        .map_err(|source| CliError::io(format!("failed to read '{}'", path.display()), source))?;
    serde_json::from_str(raw.as_str())
        .map_err(|source| CliError::json(format!("invalid JSON in '{}'", path.display()), source))
}

fn has_steps_input(predictions: &Value) -> bool {
    predictions
        .as_object()
        .and_then(|obj| obj.get("steps"))
        .is_some()
}

fn extract_online_steps(predictions: &Value) -> Result<Vec<OnlineStepResult>, CliError> {
    let steps_value = predictions
        .as_object()
        .and_then(|obj| obj.get("steps"))
        .ok_or_else(|| CliError::invalid_input("online predictions must contain 'steps' array"))?;

    serde_json::from_value(steps_value.clone())
        .map_err(|source| CliError::json("failed to parse online predictions steps", source))
}

fn extract_true_change_points(value: &Value) -> Result<Vec<usize>, CliError> {
    if let Some(array_value) = value
        .as_object()
        .and_then(|obj| obj.get("true_change_points"))
    {
        return parse_usize_array(array_value, "ground-truth.true_change_points");
    }
    if let Some(array_value) = value.as_object().and_then(|obj| obj.get("change_points")) {
        return parse_usize_array(array_value, "ground-truth.change_points");
    }
    if let Some(array_value) = value
        .as_object()
        .and_then(|obj| obj.get("result"))
        .and_then(|result| result.as_object())
        .and_then(|obj| obj.get("change_points"))
    {
        return parse_usize_array(array_value, "ground-truth.result.change_points");
    }

    Err(CliError::invalid_input(
        "ground-truth JSON for online eval must include 'true_change_points' or 'change_points'",
    ))
}

fn parse_usize_array(value: &Value, context: &str) -> Result<Vec<usize>, CliError> {
    let array = value
        .as_array()
        .ok_or_else(|| CliError::invalid_input(format!("{context} must be an array")))?;
    array
        .iter()
        .enumerate()
        .map(|(idx, item)| parse_usize(item, &format!("{context}[{idx}]")))
        .collect()
}

fn extract_offline_result(value: &Value) -> Result<OfflineChangePointResult, CliError> {
    let payload = value
        .as_object()
        .and_then(|obj| obj.get("result"))
        .unwrap_or(value)
        .clone();

    serde_json::from_value(payload).map_err(|source| {
        CliError::json("failed to parse offline change-point result JSON", source)
    })
}

fn write_json_output<T: Serialize>(
    payload: &T,
    output_path: Option<&Path>,
) -> Result<(), CliError> {
    let encoded = serde_json::to_string_pretty(payload)
        .map_err(|source| CliError::json("failed to serialize JSON output", source))?;

    if let Some(path) = output_path {
        fs::write(path, format!("{encoded}\n"))
            .map_err(|source| CliError::io(format!("failed to write '{}'", path.display()), source))
    } else {
        println!("{encoded}");
        Ok(())
    }
}

fn emit_structured_error(err: &CliError) {
    let envelope = ErrorEnvelope {
        error: ErrorPayload {
            code: err.code().to_string(),
            message: err.to_string(),
        },
    };

    match serde_json::to_string_pretty(&envelope) {
        Ok(json) => eprintln!("{json}"),
        Err(_) => eprintln!(
            "{{\"error\":{{\"code\":\"{}\",\"message\":\"{}\"}}}}",
            err.code(),
            err
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_detect_pipeline, parse_csv_data, parse_detect_args, parse_npy_bytes,
        parse_pipeline_spec_document, resolve_stopping,
    };
    use cpd_core::{Penalty, Stopping};
    use cpd_doctor::{CostConfig, DetectorConfig, OfflineDetectorConfig};

    #[test]
    fn csv_parser_supports_rectangular_data() {
        let raw = "1.0,2.0\n3.0,4.0\n";
        let (values, n, d) = parse_csv_data(raw).expect("csv should parse");
        assert_eq!(n, 2);
        assert_eq!(d, 2);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn csv_parser_skips_single_header_row_when_present() {
        let raw = "a,b\n1.0,2.0\n3.0,4.0\n";
        let (values, n, d) = parse_csv_data(raw).expect("csv with header should parse");
        assert_eq!(n, 2);
        assert_eq!(d, 2);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn csv_parser_rejects_malformed_first_row_instead_of_dropping_it() {
        let raw = "1,2,\n3.0,4.0\n5.0,6.0\n";
        let err = parse_csv_data(raw).expect_err("malformed first data row should fail");
        assert!(
            err.to_string().contains("CSV row 1"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn npy_parser_reads_f64_2d_c_contiguous() {
        let payload = make_npy_v1("<f8", false, &[2, 2], &[1.0f64, 2.0, 3.0, 4.0]);
        let (values, n, d) = parse_npy_bytes(payload.as_slice()).expect("npy should parse");
        assert_eq!(n, 2);
        assert_eq!(d, 2);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn npy_parser_transposes_fortran_order_to_c_layout() {
        let payload = make_npy_v1("<f8", true, &[2, 2], &[1.0f64, 3.0, 2.0, 4.0]);
        let (values, n, d) = parse_npy_bytes(payload.as_slice()).expect("npy should parse");
        assert_eq!(n, 2);
        assert_eq!(d, 2);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pipeline_parser_accepts_pipeline_spec_envelope() {
        let raw = r#"
        {
          "schema_version": 0,
          "kind": "pipeline_spec",
          "payload": {
            "detector": "pelt",
            "cost": "l2",
            "constraints": {
              "min_segment_len": 2
            },
            "stopping": {
              "n_bkps": 2
            }
          }
        }
        "#;

        let pipeline = parse_pipeline_spec_document(raw).expect("pipeline should parse");
        assert!(matches!(pipeline.cost, CostConfig::L2));
        assert!(matches!(
            pipeline.detector,
            DetectorConfig::Offline(OfflineDetectorConfig::Pelt(_))
        ));
        assert!(matches!(pipeline.stopping, Some(Stopping::KnownK(2))));
    }

    #[test]
    fn pipeline_parser_accepts_pipeline_spec_envelope_with_serde_l1median_cost() {
        let raw = r#"
        {
          "schema_version": 0,
          "kind": "pipeline_spec",
          "payload": {
            "detector": "pelt",
            "cost": "L1Median",
            "constraints": {
              "min_segment_len": 2
            },
            "stopping": {
              "n_bkps": 1
            }
          }
        }
        "#;

        let pipeline = parse_pipeline_spec_document(raw).expect("pipeline should parse");
        assert!(matches!(pipeline.cost, CostConfig::L1Median));
        assert!(matches!(
            pipeline.detector,
            DetectorConfig::Offline(OfflineDetectorConfig::Pelt(_))
        ));
        assert!(matches!(pipeline.stopping, Some(Stopping::KnownK(1))));
    }

    #[test]
    fn pipeline_parser_accepts_serde_stopping_penalized() {
        let raw = r#"
        {
          "detector": "pelt",
          "cost": "l2",
          "stopping": {
            "Penalized": "bic"
          }
        }
        "#;

        let pipeline = parse_pipeline_spec_document(raw).expect("pipeline should parse");
        assert!(matches!(
            pipeline.stopping,
            Some(Stopping::Penalized(Penalty::BIC))
        ));
    }

    #[test]
    fn detect_pipeline_supports_fpop_algorithm_with_l2_cost() {
        let tokens = vec![
            "--input".to_string(),
            "/tmp/series.csv".to_string(),
            "--algorithm".to_string(),
            "fpop".to_string(),
            "--cost".to_string(),
            "l2".to_string(),
            "--k".to_string(),
            "2".to_string(),
        ];
        let args = parse_detect_args(tokens.as_slice()).expect("detect args should parse");
        let pipeline = build_detect_pipeline(&args).expect("fpop+l2 should build");
        assert!(matches!(
            pipeline.detector,
            DetectorConfig::Offline(OfflineDetectorConfig::Fpop(_))
        ));
        assert!(matches!(pipeline.cost, CostConfig::L2));
    }

    #[test]
    fn detect_pipeline_accepts_normal_full_cov_cost() {
        let tokens = vec![
            "--input".to_string(),
            "/tmp/series.csv".to_string(),
            "--algorithm".to_string(),
            "pelt".to_string(),
            "--cost".to_string(),
            "normal_full_cov".to_string(),
            "--k".to_string(),
            "2".to_string(),
        ];
        let args = parse_detect_args(tokens.as_slice()).expect("detect args should parse");
        let pipeline = build_detect_pipeline(&args).expect("pelt+normal_full_cov should build");
        assert!(matches!(pipeline.cost, CostConfig::NormalFullCov));
    }

    #[test]
    fn detect_pipeline_rejects_fpop_with_non_l2_cost() {
        let tokens = vec![
            "--input".to_string(),
            "/tmp/series.csv".to_string(),
            "--algorithm".to_string(),
            "fpop".to_string(),
            "--cost".to_string(),
            "normal".to_string(),
            "--k".to_string(),
            "2".to_string(),
        ];
        let args = parse_detect_args(tokens.as_slice()).expect("detect args should parse");
        let err = build_detect_pipeline(&args).expect_err("fpop+normal should fail");
        assert!(
            err.to_string()
                .contains("--algorithm=fpop requires --cost=l2"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn pipeline_parser_accepts_fpop_with_l2_cost() {
        let raw = r#"
        {
          "detector": {"kind": "fpop"},
          "cost": "l2",
          "stopping": {"n_bkps": 2}
        }
        "#;
        let pipeline = parse_pipeline_spec_document(raw).expect("pipeline should parse");
        assert!(matches!(
            pipeline.detector,
            DetectorConfig::Offline(OfflineDetectorConfig::Fpop(_))
        ));
    }

    #[test]
    fn pipeline_parser_accepts_normal_full_cov_cost() {
        let raw = r#"
        {
          "detector": {"kind": "pelt"},
          "cost": "normal_full_cov",
          "stopping": {"n_bkps": 2}
        }
        "#;
        let pipeline = parse_pipeline_spec_document(raw).expect("pipeline should parse");
        assert!(matches!(pipeline.cost, CostConfig::NormalFullCov));
    }

    #[test]
    fn pipeline_parser_rejects_fpop_with_non_l2_cost() {
        let raw = r#"
        {
          "detector": {"kind": "fpop"},
          "cost": "normal",
          "stopping": {"n_bkps": 2}
        }
        "#;
        let err = parse_pipeline_spec_document(raw).expect_err("fpop+normal should fail");
        assert!(
            err.to_string()
                .contains("pipeline.detector='fpop' requires pipeline.cost='l2'"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn detect_rejects_explicit_penalty_when_k_is_set() {
        let tokens = vec![
            "--input".to_string(),
            "/tmp/series.csv".to_string(),
            "--k".to_string(),
            "2".to_string(),
            "--penalty".to_string(),
            "manual".to_string(),
        ];
        let args = parse_detect_args(tokens.as_slice()).expect("detect args should parse");
        let err = resolve_stopping(
            args.k,
            args.penalty,
            args.penalty_explicit,
            args.penalty_value,
        )
        .expect_err("explicit penalty with --k should fail");
        assert!(
            err.to_string()
                .contains("--penalty cannot be combined with --k"),
            "unexpected error message: {err}"
        );
    }

    fn make_npy_v1(descr: &str, fortran_order: bool, shape: &[usize], values: &[f64]) -> Vec<u8> {
        let shape_repr = match shape {
            [n] => format!("({n},)"),
            [n, d] => format!("({n}, {d})"),
            _ => panic!("shape must be 1D or 2D"),
        };
        let header_dict = format!(
            "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
            descr,
            if fortran_order { "True" } else { "False" },
            shape_repr
        );

        let mut header = header_dict;
        let prefix_len = 10usize;
        let mut total_len = prefix_len + header.len() + 1;
        let padding = (16 - (total_len % 16)) % 16;
        header.push_str(" ".repeat(padding).as_str());
        header.push('\n');
        total_len = prefix_len + header.len();
        assert_eq!(total_len % 16, 0);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1u8);
        bytes.push(0u8);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in values {
            bytes.extend_from_slice(value.to_le_bytes().as_slice());
        }
        bytes
    }
}
