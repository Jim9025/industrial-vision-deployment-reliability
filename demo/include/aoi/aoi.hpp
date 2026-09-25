#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

namespace aoi {
namespace fs = std::filesystem;

struct Config {
  std::string provider_type = "csv";
  fs::path model_path;
  std::string model_sha256;
  int input_width = 320;
  int input_height = 320;
  std::string normalization = "zero_one";
  std::string score_reduction = "mean";
  int output_index = 0;
  int warmup_count = 0;
  int benchmark_repetitions = 1;
  double parity_abs_tolerance = 1e-4;
  double parity_rel_tolerance = 1e-3;
  double score_threshold = 0.0;
  double review_margin = 0.0;
  double brightness_min = 0.0;
  double brightness_max = 1.0;
  double saturation_max = 1.0;
  double shadow_max = 1.0;
  double contrast_min = 0.0;
  double sharpness_min = 0.0;
  int max_side = 1024;
  int gaussian_kernel = 1;
  int saturation_level = 250;
  int shadow_level = 5;
  bool visualizations = false;
  bool benchmark_visualizations = false;
};

struct Metrics {
  double brightness = 0.0;
  double saturation_fraction = 0.0;
  double shadow_fraction = 0.0;
  double contrast = 0.0;
  double laplacian_sharpness = 0.0;
  double high_frequency_energy = 0.0;
};

struct Record {
  fs::path image_path;
  std::optional<double> score;
  std::optional<Metrics> metrics;
  std::string decision;
  std::string reason;
  bool review = false;
  double score_preprocess_ms = 0.0;
  double score_inference_ms = 0.0;
  double decode_ms = 0.0;
  double quality_ms = 0.0;
  double score_postprocess_ms = 0.0;
  double visualization_ms = 0.0;
  std::optional<double> max_anomaly;
  fs::path visualization_path;
  std::string inference_status = "not_run";
  std::string error_detail;
  double processing_ms = 0.0;
};

struct ScoreResult {
  std::optional<double> score;
  double preprocessing_ms = 0.0;
  double inference_ms = 0.0;
  double postprocessing_ms = 0.0;
  cv::Mat anomaly_map;
  std::string error;
};

class ScoreProvider {
 public:
  virtual ~ScoreProvider() = default;
  virtual ScoreResult score(const fs::path& image_path,
                            const cv::Mat& bgr_image) = 0;
  virtual std::string name() const = 0;
};

struct ScoreSourceOptions {
  fs::path csv_path;
  fs::path image_root;
};

struct LatencyStats {
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

struct BenchmarkResult {
  std::size_t total_images = 0;
  int warmup_count = 0;
  int repetitions = 0;
  std::size_t measured_samples = 0;
  double throughput_images_per_s = 0.0;
  LatencyStats preprocessing;
  LatencyStats inference;
  LatencyStats decode;
  LatencyStats quality;
  LatencyStats postprocessing;
  LatencyStats visualization;
  LatencyStats total;
  std::string hardware;
  std::string runtime;
  int batch_size = 1;
  bool visualization_enabled = false;
};

Config read_config(const fs::path& path);
std::vector<std::string> parse_csv_row(const std::string& line);
std::unordered_map<std::string, double> read_scores(
    const fs::path& csv, const fs::path& image_root);
std::unique_ptr<ScoreProvider> make_score_provider(
    const Config& config, const ScoreSourceOptions& options);
Metrics measure(const cv::Mat& input, const Config& config);
Record inspect(const fs::path& image_path, ScoreProvider& provider,
               const Config& config, const fs::path& visualization_dir = {});
std::vector<fs::path> list_images(const fs::path& directory);
void write_reports(const fs::path& output_dir, const std::vector<Record>& records,
                   const Config& config);
BenchmarkResult run_benchmark(const std::vector<fs::path>& images,
                              ScoreProvider& provider, const Config& config,
                              const fs::path& visualization_dir = {});
void write_benchmark(const fs::path& output_dir, const BenchmarkResult& result,
                     const std::string& provider_name);
std::size_t write_parity(const fs::path& output_dir,
                         const std::vector<fs::path>& images,
                         ScoreProvider& provider, const Config& config,
                         const fs::path& reference_csv,
                         const fs::path& image_root);
std::string normalized_path(const fs::path& path);
}  // namespace aoi
