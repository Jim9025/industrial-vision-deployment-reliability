#include "aoi/aoi.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_throws(Function action, const char* message) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error(message);
}

struct TempDirectory {
  aoi::fs::path path = aoi::fs::temp_directory_path() /
      ("aoi_tests_" + std::to_string(std::chrono::steady_clock::now()
                                          .time_since_epoch().count()));
  TempDirectory() { aoi::fs::create_directories(path); }
  ~TempDirectory() { std::error_code ignored; aoi::fs::remove_all(path, ignored); }
};

class FixedScoreProvider final : public aoi::ScoreProvider {
 public:
  explicit FixedScoreProvider(std::optional<double> score) : value_(score) {}
  aoi::ScoreResult score(const aoi::fs::path&, const cv::Mat&) override {
    aoi::ScoreResult result;
    result.score = value_;
    return result;
  }
  std::string name() const override { return "fixed_test"; }
 private:
  std::optional<double> value_;
};
}  // namespace

int main() {
  try {
    TempDirectory temp;
    const auto config_path = temp.path / "config.conf";
    {
      std::ofstream out(config_path);
      out << "provider_type=csv\ninput_width=320\ninput_height=320\n"
             "normalization=zero_one\nscore_reduction=mean\noutput_index=0\n"
             "warmup_count=1\nbenchmark_repetitions=2\n"
             "parity_abs_tolerance=0.0001\nparity_rel_tolerance=0.001\n"
             "score_threshold=0.5\nreview_margin=0.1\nbrightness_min=0.1\n"
             "brightness_max=0.9\nsaturation_max=0.2\nshadow_max=0.2\n"
             "contrast_min=0.01\nsharpness_min=0.001\nmax_side=100\n"
             "gaussian_kernel=1\nsaturation_level=250\nshadow_level=5\n";
    }
    const auto config = aoi::read_config(config_path);
    require(config.max_side == 100, "config parsing failed");
    require(config.provider_type == "csv" && config.benchmark_repetitions == 2,
            "provider config parsing failed");
    {
      std::ofstream out(temp.path / "invalid.conf");
      out << "score_threshold=nan\n";
    }
    require_throws([&] { aoi::read_config(temp.path / "invalid.conf"); },
                   "non-finite config value should fail");
    {
      std::ifstream source(config_path);
      std::ofstream out(temp.path / "invalid_provider.conf");
      std::string line;
      while (std::getline(source, line)) {
        if (line == "provider_type=csv") line = "provider_type=unknown";
        out << line << '\n';
      }
    }
    require_throws([&] { aoi::read_config(temp.path / "invalid_provider.conf"); },
                   "invalid provider config should fail");
    {
      std::ifstream source(config_path);
      std::ofstream out(temp.path / "invalid_visualization.conf");
      std::string line;
      while (std::getline(source, line)) out << line << '\n';
      out << "visualizations=2\n";
    }
    require_throws([&] { aoi::read_config(temp.path / "invalid_visualization.conf"); },
                   "invalid visualization flag should fail");
    const auto columns = aoi::parse_csv_row("\"a,b\",\"c\"\"d\",3");
    require(columns.size() == 3 && columns[0] == "a,b" && columns[1] == "c\"d",
            "quoted CSV parsing failed");

    const auto good_path = temp.path / "good.png";
    const auto dark_path = temp.path / "dark.png";
    cv::Mat textured(32, 32, CV_8UC3);
    for (int row = 0; row < textured.rows; ++row)
      for (int col = 0; col < textured.cols; ++col)
        textured.at<cv::Vec3b>(row, col) =
            cv::Vec3b((row + col) % 2 ? 200 : 40,
                      (row + col) % 2 ? 200 : 40,
                      (row + col) % 2 ? 200 : 40);
    require(cv::imwrite(good_path.string(), textured), "test image write failed");
    require(cv::imwrite(dark_path.string(), cv::Mat::zeros(32, 32, CV_8UC3)),
            "dark image write failed");
    const auto corrupt_path = temp.path / "corrupt.png";
    {
      std::ofstream corrupt(corrupt_path);
      corrupt << "not a PNG";
    }

    const auto prediction_path = temp.path / "scores.csv";
    {
      std::ofstream out(prediction_path);
      out << "image_path,image_score\n" << good_path.filename().string()
          << ",0.2\n" << dark_path.string() << ",0.8\n";
    }
    const auto scores = aoi::read_scores(prediction_path, temp.path);
    require(scores.size() == 2, "score import failed");
    require(std::abs(scores.at(aoi::normalized_path(good_path)) - 0.2) < 1e-12,
            "relative image path failed");
    const auto link_path = temp.path / "linked.png";
    aoi::fs::create_symlink(good_path, link_path);
    require(aoi::normalized_path(link_path) == aoi::normalized_path(good_path),
            "symlink should resolve to score path");
    auto csv_provider = aoi::make_score_provider(config, {prediction_path, temp.path});
    require(csv_provider->name() == "csv", "CSV provider factory failed");
    const auto first_score = csv_provider->score(good_path, textured);
    const auto second_score = csv_provider->score(good_path, textured);
    require(first_score.score && first_score.score == second_score.score &&
                *first_score.score == 0.2,
            "CSV provider must be deterministic");
    FixedScoreProvider low(0.2), high(0.8), near(0.45), boundary(0.5),
                       none(std::nullopt);
    const auto good = aoi::inspect(good_path, low, config);
    const auto dark = aoi::inspect(dark_path, high, config);
    const auto quality_only = aoi::inspect(dark_path, low, config);
    const auto near_threshold = aoi::inspect(good_path, near, config);
    const auto missing = aoi::inspect(good_path, none, config);
    const auto unreadable = aoi::inspect(temp.path / "absent.png", low, config);
    const auto corrupt = aoi::inspect(corrupt_path, low, config);
    const auto at_boundary = aoi::inspect(good_path, boundary, config);
    const auto visualized = aoi::inspect(good_path, low, config,
                                           temp.path / "visualizations");
    const auto blocked_visualization = temp.path / "blocked_visualization";
    { std::ofstream blocked(blocked_visualization); blocked << "file"; }
    const auto visualization_error = aoi::inspect(good_path, low, config,
                                                   blocked_visualization);
    const auto csv_decision = aoi::inspect(good_path, *csv_provider, config);
    require(csv_decision.decision == good.decision &&
                csv_decision.score == good.score,
            "decision should not depend on provider implementation");
    const auto benchmark = aoi::run_benchmark({good_path, dark_path}, low, config);
    require(benchmark.total_images == 2 && benchmark.warmup_count == 1 &&
                benchmark.measured_samples == 4 && benchmark.total.mean_ms > 0.0 &&
                benchmark.total.p95_ms >= benchmark.total.median_ms,
            "benchmark count or latency stats failed");
    aoi::write_benchmark(temp.path / "benchmark", benchmark, low.name());
    require(aoi::fs::file_size(temp.path / "benchmark" / "benchmark.json") > 0,
            "benchmark output missing");
    auto bad_provider_config = config;
    bad_provider_config.provider_type = "invalid";
    require_throws([&] { aoi::make_score_provider(bad_provider_config, {}); },
                   "invalid provider must fail");
    bad_provider_config.provider_type = "onnx";
    bad_provider_config.model_path = temp.path / "missing.onnx";
    require_throws([&] { aoi::make_score_provider(bad_provider_config, {}); },
                   "missing ONNX model must fail");
#ifdef AOI_TEST_MODEL_PATH
    bad_provider_config.model_path = AOI_TEST_MODEL_PATH;
    bad_provider_config.model_sha256 =
        "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8";
    auto onnx_provider = aoi::make_score_provider(bad_provider_config, {});
    require(onnx_provider->name() == "onnx", "ONNX provider factory failed");
    const auto onnx_result = onnx_provider->score(good_path, textured);
    require(onnx_result.score && std::isfinite(*onnx_result.score) &&
                onnx_result.error.empty(), "ONNX smoke inference failed");
    require_throws([&] {
      aoi::write_parity(temp.path / "wrong_model", {good_path},
                        *onnx_provider, bad_provider_config,
                        prediction_path, temp.path);
    }, "parity must reject an unlabelled model reference");
    const auto wrong_reference = temp.path / "wrong_reference.csv";
    {
      std::ofstream out(wrong_reference);
      out << "image_path,image_score,model_sha256\n" << good_path.string()
          << ",999," << bad_provider_config.model_sha256 << "\n";
    }
    require(aoi::write_parity(temp.path / "mismatch", {good_path},
                              *onnx_provider, bad_provider_config,
                              wrong_reference, temp.path) == 1,
            "parity mismatch should be reported");
    bad_provider_config.input_width = 319;
    require_throws([&] { aoi::make_score_provider(bad_provider_config, {}); },
                   "ONNX input shape mismatch must fail");
#endif
#ifdef AOI_TEST_EFFICIENTAD_PATH
    auto industrial_config = config;
    industrial_config.provider_type = "efficientad";
    industrial_config.model_path = AOI_TEST_EFFICIENTAD_PATH;
    industrial_config.input_width = 256;
    industrial_config.input_height = 256;
    industrial_config.normalization = "imagenet";
    industrial_config.score_reduction = "max";
    auto industrial_provider = aoi::make_score_provider(industrial_config, {});
    require(industrial_provider->name() == "efficientad",
            "EfficientAD provider factory failed");
    const auto industrial_score = industrial_provider->score(good_path, textured);
    require(industrial_score.score && std::isfinite(*industrial_score.score) &&
                industrial_score.anomaly_map.rows == textured.rows &&
                industrial_score.anomaly_map.cols == textured.cols,
            "EfficientAD smoke inference or anomaly map failed");
    const auto industrial_record = aoi::inspect(good_path, *industrial_provider,
                                                 industrial_config,
                                                 temp.path / "industrial_visuals");
    require(industrial_record.inference_status == "ok" &&
                industrial_record.max_anomaly &&
                aoi::fs::is_regular_file(industrial_record.visualization_path),
            "EfficientAD inspection or heatmap overlay failed");
    industrial_config.input_width = 255;
    require_throws([&] { aoi::make_score_provider(industrial_config, {}); },
                   "EfficientAD invalid input shape must fail");
    industrial_config.model_path = temp.path / "missing_efficientad.onnx";
    require_throws([&] { aoi::make_score_provider(industrial_config, {}); },
                   "EfficientAD missing model must fail");
#endif
    require(good.decision == "PASS" && good.metrics && good.processing_ms >= 0.0,
            "expected PASS with metrics");
    require(dark.decision == "NG" && dark.review &&
                dark.reason.find("score_threshold") != std::string::npos,
            "expected threshold NG in review queue");
    require(quality_only.decision == "NG" && quality_only.review &&
                quality_only.reason.find("score_threshold") == std::string::npos &&
                quality_only.reason.find("brightness") != std::string::npos,
            "quality gate should reject independently of score");
    require(near_threshold.decision == "PASS" && near_threshold.review,
            "near-threshold PASS should enter review queue");
    require(missing.decision == "NG" && missing.review &&
                missing.reason.find("missing_score") != std::string::npos,
            "missing score must fail closed");
    require(unreadable.decision == "NG" && !unreadable.metrics &&
                unreadable.reason.find("unreadable_image") != std::string::npos &&
                unreadable.error_detail.find("absent.png") != std::string::npos,
            "unreadable image must fail closed");
    require(corrupt.decision == "NG" && corrupt.inference_status == "not_run",
            "corrupt image must fail before inference");
    require(at_boundary.decision == "NG" &&
                at_boundary.reason.find("score_threshold") != std::string::npos,
            "threshold boundary must be NG");
    require(visualized.inference_status == "ok" &&
                aoi::fs::is_regular_file(visualized.visualization_path),
            "PASS visualization must be written");
    require(visualization_error.decision == "NG" &&
                visualization_error.inference_status == "ok" &&
                visualization_error.reason.find("visualization_error") != std::string::npos &&
                visualization_error.error_detail.find("visualization:") != std::string::npos,
            "visualization failure must name its stage and fail closed");
    require(dark.metrics->brightness < good.metrics->brightness &&
                dark.metrics->contrast < good.metrics->contrast,
            "quality metrics did not distinguish dark image");

    const auto output = temp.path / "out";
    aoi::write_reports(output, {good, dark, missing}, config);
    require(aoi::fs::file_size(output / "report.csv") > 0 &&
                aoi::fs::file_size(output / "report.json") > 0 &&
                aoi::fs::file_size(output / "summary.json") > 0,
            "reports missing");
    {
      std::ifstream csv(output / "report.csv");
      std::string header;
      std::getline(csv, header);
      require(header.find("inference_status") != std::string::npos &&
                  header.find("threshold") != std::string::npos &&
                  header.find("decode_ms") != std::string::npos,
              "industrial report columns missing");
    }
    {
      std::ifstream queue(output / "review_queue.csv");
      std::string line;
      int lines = 0;
      while (std::getline(queue, line)) ++lines;
      require(lines == 3, "review queue should contain header and two rows");
    }
    {
      std::ofstream out(prediction_path, std::ios::app);
      out << good_path.filename().string() << ",0.3\n";
    }
    require_throws([&] { aoi::read_scores(prediction_path, temp.path); },
                   "duplicate score path should fail");
    std::cout << "aoi_tests: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "aoi_tests: " << error.what() << '\n';
    return 1;
  }
}
