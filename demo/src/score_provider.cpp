#include "aoi/aoi.hpp"

#include <stdexcept>

namespace aoi {
std::unique_ptr<ScoreProvider> make_onnx_provider(const Config& config);

namespace {
class CsvScoreProvider final : public ScoreProvider {
 public:
  explicit CsvScoreProvider(const ScoreSourceOptions& options)
      : scores_(read_scores(options.csv_path, options.image_root)) {}

  ScoreResult score(const fs::path& image_path, const cv::Mat&) override {
    const auto it = scores_.find(normalized_path(image_path));
    if (it == scores_.end()) return {};
    ScoreResult result;
    result.score = it->second;
    return result;
  }

  std::string name() const override { return "csv"; }

 private:
  std::unordered_map<std::string, double> scores_;
};
}  // namespace

std::unique_ptr<ScoreProvider> make_score_provider(
    const Config& config, const ScoreSourceOptions& options) {
  if (config.provider_type == "csv") {
    if (options.csv_path.empty())
      throw std::runtime_error("CSV provider requires --scores");
    return std::make_unique<CsvScoreProvider>(options);
  }
  if (config.provider_type == "onnx" || config.provider_type == "efficientad")
    return make_onnx_provider(config);
  throw std::runtime_error("Unsupported provider_type: " + config.provider_type);
}
}  // namespace aoi
