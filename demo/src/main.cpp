#include "aoi/aoi.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
void usage(std::ostream& out) {
  out << "Usage: aoi_demo --config FILE --input IMAGE_DIR --output DIR "
         "[--scores CSV --image-root DIR] [--max-images N] [--benchmark] "
         "[--parity-reference CSV --parity-image-root DIR]\n";
}

std::size_t max_images(const std::string& text) {
  std::size_t used = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &used);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid --max-images value");
  }
  if (used != text.size() || value == 0)
    throw std::runtime_error("--max-images must be a positive integer");
  return static_cast<std::size_t>(value);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      usage(std::cout);
      return EXIT_SUCCESS;
    }
    std::unordered_map<std::string, std::string> args;
    bool benchmark = false;
    for (int i = 1; i < argc; ++i) {
      const std::string key(argv[i]);
      if (key == "--benchmark") {
        if (benchmark) throw std::runtime_error("Duplicate --benchmark");
        benchmark = true;
        continue;
      }
      if (key != "--config" && key != "--input" && key != "--scores" &&
          key != "--image-root" && key != "--output" && key != "--max-images" &&
          key != "--parity-reference" && key != "--parity-image-root")
        throw std::runtime_error("Unknown argument: " + key);
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
      if (!args.emplace(key, argv[++i]).second)
        throw std::runtime_error("Duplicate argument: " + key);
    }
    for (const std::string key : {"--config", "--input", "--output"})
      if (!args.count(key)) throw std::runtime_error("Missing argument: " + key);

    const auto config = aoi::read_config(args.at("--config"));
    const aoi::ScoreSourceOptions source_options{
        args.count("--scores") ? args.at("--scores") : "",
        args.count("--image-root") ? args.at("--image-root") : ""};
    auto provider = aoi::make_score_provider(config, source_options);
    auto images = aoi::list_images(args.at("--input"));
    if (args.count("--max-images")) {
      const auto limit = max_images(args.at("--max-images"));
      if (images.size() > limit) images.resize(limit);
    }
    std::vector<aoi::Record> records;
    records.reserve(images.size());
    std::size_t passed = 0, rejected = 0, queued = 0;
    for (const auto& path : images) {
      const auto record = aoi::inspect(
          path, *provider, config,
          config.visualizations ? aoi::fs::path(args.at("--output")) / "visualizations"
                                : aoi::fs::path{});
      passed += record.decision == "PASS";
      rejected += record.decision == "NG";
      queued += record.review;
      std::cout << record.decision << "  " << path.filename().string()
                << "  score=";
      if (record.score) std::cout << *record.score;
      else std::cout << "missing";
      if (record.metrics) {
        const auto& m = *record.metrics;
        std::cout << "  brightness=" << m.brightness
                  << "  saturation=" << m.saturation_fraction
                  << "  shadow=" << m.shadow_fraction
                  << "  contrast=" << m.contrast
                  << "  sharpness=" << m.laplacian_sharpness
                  << "  hf_energy=" << m.high_frequency_energy;
      }
      std::cout << "  ms=" << record.processing_ms
                << "  reason=" << record.reason << '\n';
      records.push_back(record);
    }
    aoi::write_reports(args.at("--output"), records, config);
    std::cout << "Summary: " << records.size() << " images, PASS=" << passed
              << ", NG=" << rejected << ", review=" << queued << '\n'
              << "Reports: " << args.at("--output") << '\n';
    if (benchmark) {
      const auto result = aoi::run_benchmark(
          images, *provider, config,
          config.benchmark_visualizations
              ? aoi::fs::path(args.at("--output")) / "benchmark_visualizations"
              : aoi::fs::path{});
      aoi::write_benchmark(args.at("--output"), result, provider->name());
      std::cout << "Benchmark: " << result.measured_samples << " samples, "
                << "total mean=" << result.total.mean_ms << " ms, p95="
                << result.total.p95_ms << " ms, throughput="
                << result.throughput_images_per_s << " images/s\n";
    }
    if (args.count("--parity-reference")) {
      const auto mismatches = aoi::write_parity(
          args.at("--output"), images, *provider, config,
          args.at("--parity-reference"),
          args.count("--parity-image-root")
              ? aoi::fs::path(args.at("--parity-image-root"))
              : aoi::fs::current_path());
      std::cout << "Parity: " << images.size() << " images, mismatches="
                << mismatches << '\n';
      if (mismatches) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "aoi_demo: " << error.what() << '\n';
    usage(std::cerr);
    return EXIT_FAILURE;
  }
}
