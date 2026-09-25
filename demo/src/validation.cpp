#include "aoi/aoi.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <sys/utsname.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <onnxruntime_c_api.h>
#include <opencv2/imgcodecs.hpp>

namespace aoi {
namespace {
std::string quoted(const std::string& value) {
  std::string result = "\"";
  for (unsigned char ch : value) {
    if (ch == '"') result += "\\\"";
    else if (ch == '\n') result += "\\n";
    else if (ch == '\r') result += "\\r";
    else if (ch == '\t') result += "\\t";
    else if (ch == '\\') result += "\\\\";
    else if (ch < 0x20) {
      std::ostringstream escaped;
      escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch);
      result += escaped.str();
    } else result += static_cast<char>(ch);
  }
  return result + '"';
}

std::string csv_quoted(const std::string& value) {
  std::string result = "\"";
  for (char ch : value) {
    if (ch == '"') result += '"';
    result += ch;
  }
  return result + '"';
}

LatencyStats stats(std::vector<double> values) {
  if (values.empty()) throw std::runtime_error("No latency measurements");
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double median = n % 2 ? values[n / 2]
                              : (values[n / 2 - 1] + values[n / 2]) / 2.0;
  const auto p95_index = static_cast<std::size_t>(std::ceil(0.95 * n)) - 1;
  const auto p99_index = static_cast<std::size_t>(std::ceil(0.99 * n)) - 1;
  return {sum / n, median, values[p95_index], values[p99_index],
          values.front(), values.back()};
}

void write_stats_json(std::ostream& out, const LatencyStats& value) {
  out << "{\"mean_ms\": " << value.mean_ms
      << ", \"median_ms\": " << value.median_ms
      << ", \"p95_ms\": " << value.p95_ms
      << ", \"p99_ms\": " << value.p99_ms
      << ", \"min_ms\": " << value.min_ms
      << ", \"max_ms\": " << value.max_ms << '}';
}

void write_stats_csv(std::ostream& out, const std::string& phase,
                     const LatencyStats& value) {
  out << phase << ',' << value.mean_ms << ',' << value.median_ms << ','
      << value.p95_ms << ',' << value.p99_ms << ',' << value.min_ms << ','
      << value.max_ms << '\n';
}

std::string hardware_name() {
  struct utsname info{};
  if (uname(&info) != 0) return "unavailable";
  std::string result = std::string(info.sysname) + " " + info.machine;
#ifdef __APPLE__
  char brand[256]{};
  size_t length = sizeof(brand);
  if (sysctlbyname("machdep.cpu.brand_string", brand, &length, nullptr, 0) == 0)
    result += " / " + std::string(brand);
#endif
  return result;
}
}  // namespace

BenchmarkResult run_benchmark(const std::vector<fs::path>& images,
                              ScoreProvider& provider, const Config& config,
                              const fs::path& visualization_dir) {
  if (images.empty()) throw std::runtime_error("No benchmark images");
  for (int i = 0; i < config.warmup_count; ++i) {
    const auto& path = images[static_cast<std::size_t>(i) % images.size()];
    const auto result = inspect(path, provider, config, visualization_dir);
    if (!result.score || !result.error_detail.empty())
      throw std::runtime_error("Benchmark warmup failed for " + path.string());
  }
  std::vector<double> pre, infer, total, decode, quality, post, visualization;
  const auto sample_count = images.size() *
                            static_cast<std::size_t>(config.benchmark_repetitions);
  pre.reserve(sample_count);
  infer.reserve(sample_count);
  total.reserve(sample_count);
  decode.reserve(sample_count);
  quality.reserve(sample_count);
  post.reserve(sample_count);
  visualization.reserve(sample_count);
  const auto started = std::chrono::steady_clock::now();
  for (int repetition = 0; repetition < config.benchmark_repetitions;
       ++repetition) {
    for (const auto& path : images) {
      const auto result = inspect(path, provider, config, visualization_dir);
      if (!result.score || !result.error_detail.empty())
        throw std::runtime_error("Benchmark failed for " + path.string());
      pre.push_back(result.score_preprocess_ms);
      infer.push_back(result.score_inference_ms);
      total.push_back(result.processing_ms);
      decode.push_back(result.decode_ms);
      quality.push_back(result.quality_ms);
      post.push_back(result.score_postprocess_ms);
      if (!visualization_dir.empty()) visualization.push_back(result.visualization_ms);
    }
  }
  const double wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  BenchmarkResult result;
  result.total_images = images.size();
  result.warmup_count = config.warmup_count;
  result.repetitions = config.benchmark_repetitions;
  result.measured_samples = sample_count;
  result.throughput_images_per_s = sample_count / std::max(wall_seconds, 1e-9);
  result.preprocessing = stats(pre);
  result.inference = stats(infer);
  result.decode = stats(decode);
  result.quality = stats(quality);
  result.postprocessing = stats(post);
  if (!visualization.empty()) result.visualization = stats(visualization);
  result.total = stats(total);
  result.hardware = hardware_name();
  result.runtime = "OpenCV " + cv::getVersionString() + "; ONNX Runtime " +
                   OrtGetApiBase()->GetVersionString() + "; CPU; sequential";
  result.visualization_enabled = !visualization_dir.empty();
  return result;
}

void write_benchmark(const fs::path& output_dir, const BenchmarkResult& result,
                     const std::string& provider_name) {
  fs::create_directories(output_dir);
  std::ofstream json(output_dir / "benchmark.json");
  std::ofstream csv(output_dir / "benchmark.csv");
  if (!json || !csv) throw std::runtime_error("Cannot create benchmark reports");
  json << std::setprecision(17)
       << "{\"provider_type\": " << quoted(provider_name)
       << ", \"total_images\": " << result.total_images
       << ", \"warmup_count\": " << result.warmup_count
       << ", \"repetitions\": " << result.repetitions
       << ", \"measured_samples\": " << result.measured_samples
       << ", \"batch_size\": " << result.batch_size
       << ", \"hardware\": " << quoted(result.hardware)
       << ", \"runtime\": " << quoted(result.runtime)
       << ", \"visualization_enabled\": "
       << (result.visualization_enabled ? "true" : "false")
       << ", \"throughput_images_per_s\": "
       << result.throughput_images_per_s << ", \"decode\": ";
  write_stats_json(json, result.decode);
  json << ", \"quality_metrics\": ";
  write_stats_json(json, result.quality);
  json << ", \"preprocessing\": ";
  write_stats_json(json, result.preprocessing);
  json << ", \"inference\": ";
  write_stats_json(json, result.inference);
  json << ", \"postprocessing\": ";
  write_stats_json(json, result.postprocessing);
  json << ", \"visualization\": ";
  if (result.visualization_enabled) write_stats_json(json, result.visualization);
  else json << "null";
  json << ", \"total\": ";
  write_stats_json(json, result.total);
  json << "}\n";
  csv << std::setprecision(17)
      << "phase,mean_ms,median_ms,p95_ms,p99_ms,min_ms,max_ms\n";
  write_stats_csv(csv, "decode", result.decode);
  write_stats_csv(csv, "quality_metrics", result.quality);
  write_stats_csv(csv, "score_preprocessing", result.preprocessing);
  write_stats_csv(csv, "score_inference", result.inference);
  write_stats_csv(csv, "score_postprocessing", result.postprocessing);
  if (result.visualization_enabled)
    write_stats_csv(csv, "visualization", result.visualization);
  else csv << "visualization,unavailable,unavailable,unavailable,unavailable,unavailable,unavailable\n";
  write_stats_csv(csv, "total_inspection", result.total);
  if (!json || !csv) throw std::runtime_error("Failed to write benchmark reports");
}

std::size_t write_parity(const fs::path& output_dir,
                         const std::vector<fs::path>& images,
                         ScoreProvider& provider, const Config& config,
                         const fs::path& reference_csv,
                         const fs::path& image_root) {
  if (provider.name() != "onnx" && provider.name() != "efficientad")
    throw std::runtime_error("Parity report requires ONNX or EfficientAD provider");
  if (config.model_sha256.empty())
    throw std::runtime_error("Parity requires model_sha256 in config");
  std::unordered_map<std::string, double> reference_map_means;
  {
    std::ifstream reference_input(reference_csv);
    std::string line;
    if (!std::getline(reference_input, line))
      throw std::runtime_error("Empty parity reference CSV");
    const auto header = parse_csv_row(line);
    const auto hash_column = std::find(header.begin(), header.end(), "model_sha256");
    if (hash_column == header.end())
      throw std::runtime_error("Parity reference needs model_sha256 column");
    const auto hash_index = static_cast<std::size_t>(hash_column - header.begin());
    const auto path_column = std::find(header.begin(), header.end(), "image_path");
    const auto map_column = std::find(header.begin(), header.end(), "map_mean");
    if (provider.name() == "efficientad" &&
        (path_column == header.end() || map_column == header.end()))
      throw std::runtime_error("EfficientAD parity requires image_path and map_mean");
    while (std::getline(reference_input, line)) {
      if (line.empty()) continue;
      const auto row = parse_csv_row(line);
      if (row.size() != header.size() || row[hash_index] != config.model_sha256)
        throw std::runtime_error("Parity reference model_sha256 mismatch");
      if (map_column != header.end()) {
        const auto map_text = row[static_cast<std::size_t>(map_column - header.begin())];
        const double map_mean = std::stod(map_text);
        if (!std::isfinite(map_mean))
          throw std::runtime_error("Non-finite parity map_mean");
        fs::path map_path(row[static_cast<std::size_t>(path_column - header.begin())]);
        if (map_path.is_relative()) map_path = image_root / map_path;
        reference_map_means[normalized_path(map_path)] = map_mean;
      }
    }
  }
  const auto references = read_scores(reference_csv, image_root);
  fs::create_directories(output_dir);
  std::ofstream csv(output_dir / "parity.csv");
  std::ofstream json(output_dir / "parity.json");
  if (!csv || !json) throw std::runtime_error("Cannot create parity reports");
  csv << std::setprecision(17)
      << "image_path,reference_score,onnx_score,absolute_error,relative_error,"
         "reference_decision,onnx_decision,agreement,mismatch,"
         "reference_map_mean,cpp_map_mean,map_mean_absolute_error\n";
  json << std::setprecision(17)
       << "{\"kind\": \"same_model_cross_runtime_parity\", "
          "\"provider_type\": " << quoted(provider.name())
       << ", \"total_images\": " << images.size()
       << ", \"absolute_tolerance\": " << config.parity_abs_tolerance
       << ", \"relative_tolerance\": " << config.parity_rel_tolerance
       << ", \"mismatches\": [";
  std::size_t mismatches = 0, agreements = 0, decision_mismatches = 0,
              score_mismatches = 0, map_mismatches = 0;
  double max_map_mean_absolute_error = 0.0;
  for (const auto& image_path : images) {
    const auto it = references.find(normalized_path(image_path));
    if (it == references.end())
      throw std::runtime_error("No frozen reference for " + image_path.string());
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty())
      throw std::runtime_error("Cannot decode parity image: " + image_path.string());
    const auto observed = provider.score(image_path, image);
    if (!observed.score || !observed.error.empty())
      throw std::runtime_error("Parity inference failed: " + image_path.string() +
                               " " + observed.error);
    const double expected = it->second;
    const double actual = *observed.score;
    const double absolute_error = std::abs(expected - actual);
    const double relative_error = absolute_error / std::max(std::abs(expected), 1e-12);
    const std::string expected_decision =
        expected >= config.score_threshold ? "NG" : "PASS";
    const std::string actual_decision =
        actual >= config.score_threshold ? "NG" : "PASS";
    const bool agreement = expected_decision == actual_decision;
    const bool score_mismatch =
        (absolute_error > config.parity_abs_tolerance &&
         relative_error > config.parity_rel_tolerance);
    std::optional<double> map_expected, map_actual, map_absolute_error;
    bool map_mismatch = false;
    if (provider.name() == "efficientad") {
      const auto map_it = reference_map_means.find(normalized_path(image_path));
      if (map_it == reference_map_means.end() || observed.anomaly_map.empty())
        throw std::runtime_error("Missing EfficientAD parity map for " + image_path.string());
      map_expected = map_it->second;
      map_actual = cv::mean(observed.anomaly_map)[0];
      map_absolute_error = std::abs(*map_expected - *map_actual);
      const double map_relative_error = *map_absolute_error /
          std::max(std::abs(*map_expected), 1e-12);
      map_mismatch = *map_absolute_error > config.parity_abs_tolerance &&
                     map_relative_error > config.parity_rel_tolerance;
      max_map_mean_absolute_error = std::max(max_map_mean_absolute_error,
                                             *map_absolute_error);
    }
    const bool mismatch = !agreement || score_mismatch || map_mismatch;
    decision_mismatches += !agreement;
    score_mismatches += score_mismatch;
    map_mismatches += map_mismatch;
    agreements += agreement;
    csv << csv_quoted(image_path.string()) << ',' << expected << ',' << actual
        << ',' << absolute_error << ',' << relative_error << ','
        << expected_decision << ',' << actual_decision << ','
        << (agreement ? "true" : "false") << ','
        << (mismatch ? "true" : "false") << ',';
    if (map_expected) csv << *map_expected;
    csv << ',';
    if (map_actual) csv << *map_actual;
    csv << ',';
    if (map_absolute_error) csv << *map_absolute_error;
    csv << '\n';
    if (mismatch) {
      if (mismatches) json << ',';
      json << "{\"image_path\": " << quoted(image_path.string())
           << ", \"reference_score\": " << expected
           << ", \"onnx_score\": " << actual
           << ", \"absolute_error\": " << absolute_error
           << ", \"relative_error\": " << relative_error
           << ", \"map_mean_absolute_error\": "
           << (map_absolute_error ? *map_absolute_error : 0.0)
           << ", \"reference_decision\": " << quoted(expected_decision)
           << ", \"onnx_decision\": " << quoted(actual_decision) << '}';
      ++mismatches;
    }
  }
  json << "], \"mismatch_count\": " << mismatches
       << ", \"decision_mismatch_count\": " << decision_mismatches
       << ", \"score_mismatch_count\": " << score_mismatches
       << ", \"map_mismatch_count\": " << map_mismatches
       << ", \"max_map_mean_absolute_error\": " << max_map_mean_absolute_error
       << ", \"decision_agreement_count\": " << agreements << "}\n";
  if (!csv || !json) throw std::runtime_error("Failed to write parity reports");
  return mismatches;
}
}  // namespace aoi
