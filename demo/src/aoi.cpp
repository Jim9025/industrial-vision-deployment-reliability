#include "aoi/aoi.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace aoi {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

double finite_number(const std::string& value, const std::string& name) {
  std::size_t used = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &used);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid number for " + name + ": " + value);
  }
  if (used != value.size() || !std::isfinite(parsed))
    throw std::runtime_error("Invalid number for " + name + ": " + value);
  return parsed;
}

int integer(const std::string& value, const std::string& name) {
  const double parsed = finite_number(value, name);
  if (parsed != std::floor(parsed) || parsed < -2147483647.0 ||
      parsed > 2147483647.0)
    throw std::runtime_error("Invalid integer for " + name + ": " + value);
  return static_cast<int>(parsed);
}

void require_fraction(double value, const std::string& name) {
  if (value < 0.0 || value > 1.0)
    throw std::runtime_error(name + " must be in [0, 1]");
}

std::string csv_quote(const std::string& value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
  std::string result = "\"";
  for (char ch : value) {
    if (ch == '"') result += '"';
    result += ch;
  }
  return result + '"';
}

std::string json_quote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  out << '"';
  return out.str();
}

std::string optional_number(const std::optional<double>& value) {
  if (!value) return "";
  std::ostringstream out;
  out << std::setprecision(17) << *value;
  return out.str();
}

void check_stream(const std::ofstream& stream, const fs::path& path) {
  if (!stream) throw std::runtime_error("Failed to write " + path.string());
}

fs::path render_visualization(const fs::path& image_path, const cv::Mat& image,
                              const cv::Mat& anomaly_map, const std::string& decision,
                              const std::optional<double>& score, double threshold,
                              const fs::path& directory, int max_side) {
  fs::create_directories(directory);
  cv::Mat canvas;
  const int longest = std::max(image.cols, image.rows);
  if (longest > max_side) {
    const double scale = static_cast<double>(max_side) / longest;
    cv::resize(image, canvas, cv::Size(), scale, scale, cv::INTER_AREA);
  } else {
    canvas = image.clone();
  }
  if (!anomaly_map.empty()) {
    cv::Mat resized_map, normalized, colored;
    cv::resize(anomaly_map, resized_map, canvas.size(), 0, 0, cv::INTER_LINEAR);
    cv::normalize(resized_map, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
    cv::addWeighted(canvas, 0.6, colored, 0.4, 0, canvas);
  }
  const int band = std::min(80, std::max(40, canvas.rows / 6));
  cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, band), cv::Scalar(0, 0, 0), cv::FILLED);
  std::ostringstream caption;
  caption << image_path.filename().string() << "  " << decision << "  score=";
  if (score) caption << std::fixed << std::setprecision(5) << *score;
  else caption << "missing";
  caption << "  threshold=" << std::fixed << std::setprecision(5) << threshold;
  cv::putText(canvas, caption.str(), cv::Point(8, std::min(band - 10, 30)),
              cv::FONT_HERSHEY_SIMPLEX, 0.45,
              decision == "PASS" ? cv::Scalar(80, 230, 80) : cv::Scalar(80, 80, 255),
              1, cv::LINE_AA);
  std::uint64_t path_hash = 14695981039346656037ULL;
  for (unsigned char ch : normalized_path(image_path)) {
    path_hash ^= ch;
    path_hash *= 1099511628211ULL;
  }
  std::ostringstream name;
  name << image_path.stem().string() << '_' << std::hex
       << path_hash << ".png";
  const auto destination = directory / name.str();
  if (!cv::imwrite(destination.string(), canvas))
    throw std::runtime_error("Cannot write visualization: " + destination.string());
  return destination;
}
}  // namespace

std::string normalized_path(const fs::path& path) {
  return fs::weakly_canonical(fs::absolute(path)).string();
}

Config read_config(const fs::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open config: " + path.string());
  Config config;
  std::unordered_set<std::string> seen;
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    const auto equal = line.find('=');
    if (equal == std::string::npos)
      throw std::runtime_error("Expected key=value at config line " +
                               std::to_string(line_number));
    const std::string key = trim(line.substr(0, equal));
    const std::string value = trim(line.substr(equal + 1));
    if (key.empty() || value.empty() || !seen.insert(key).second)
      throw std::runtime_error("Empty or duplicate config key at line " +
                               std::to_string(line_number));
    if (key == "score_threshold" || key == "threshold")
      config.score_threshold = finite_number(value, key);
    else if (key == "provider_type") config.provider_type = value;
    else if (key == "model_path") config.model_path = value;
    else if (key == "model_sha256") config.model_sha256 = value;
    else if (key == "input_width") config.input_width = integer(value, key);
    else if (key == "input_height") config.input_height = integer(value, key);
    else if (key == "normalization") config.normalization = value;
    else if (key == "score_reduction") config.score_reduction = value;
    else if (key == "output_index") config.output_index = integer(value, key);
    else if (key == "warmup_count") config.warmup_count = integer(value, key);
    else if (key == "benchmark_repetitions")
      config.benchmark_repetitions = integer(value, key);
    else if (key == "parity_abs_tolerance")
      config.parity_abs_tolerance = finite_number(value, key);
    else if (key == "parity_rel_tolerance")
      config.parity_rel_tolerance = finite_number(value, key);
    else if (key == "review_margin") config.review_margin = finite_number(value, key);
    else if (key == "brightness_min") config.brightness_min = finite_number(value, key);
    else if (key == "brightness_max") config.brightness_max = finite_number(value, key);
    else if (key == "saturation_max") config.saturation_max = finite_number(value, key);
    else if (key == "shadow_max") config.shadow_max = finite_number(value, key);
    else if (key == "contrast_min") config.contrast_min = finite_number(value, key);
    else if (key == "sharpness_min") config.sharpness_min = finite_number(value, key);
    else if (key == "max_side") config.max_side = integer(value, key);
    else if (key == "gaussian_kernel") config.gaussian_kernel = integer(value, key);
    else if (key == "saturation_level") config.saturation_level = integer(value, key);
    else if (key == "shadow_level") config.shadow_level = integer(value, key);
    else if (key == "visualizations") {
      const int flag = integer(value, key);
      if (flag != 0 && flag != 1) throw std::runtime_error("visualizations must be 0 or 1");
      config.visualizations = flag == 1;
    }
    else if (key == "benchmark_visualizations") {
      const int flag = integer(value, key);
      if (flag != 0 && flag != 1) throw std::runtime_error("benchmark_visualizations must be 0 or 1");
      config.benchmark_visualizations = flag == 1;
    }
    else throw std::runtime_error("Unknown config key: " + key);
  }
  const std::vector<std::string> required = {
      "provider_type", "input_width", "input_height", "normalization",
      "score_reduction", "output_index", "warmup_count",
      "benchmark_repetitions", "parity_abs_tolerance", "parity_rel_tolerance",
      "review_margin", "brightness_min", "brightness_max",
      "saturation_max", "shadow_max", "contrast_min", "sharpness_min",
      "max_side", "gaussian_kernel", "saturation_level", "shadow_level"};
  for (const auto& key : required)
    if (!seen.count(key)) throw std::runtime_error("Missing config key: " + key);
  if (seen.count("threshold") == seen.count("score_threshold"))
    throw std::runtime_error("Config requires exactly one threshold key");
  if (config.provider_type != "csv" && config.provider_type != "onnx" &&
      config.provider_type != "efficientad")
    throw std::runtime_error("provider_type must be csv, onnx, or efficientad");
  if (config.provider_type != "csv" && config.model_path.empty())
    throw std::runtime_error("model_path is required for ONNX provider");
  if (config.provider_type == "efficientad" &&
      (config.input_width != 256 || config.input_height != 256 ||
       config.normalization != "imagenet" || config.output_index != 0 ||
       config.score_reduction != "max"))
    throw std::runtime_error("EfficientAD requires 256x256 ImageNet input, output_index=0, score_reduction=max");
  if (!config.model_path.empty() && config.model_path.is_relative())
    config.model_path = fs::absolute(path.parent_path() / config.model_path);
  if (config.normalization != "none" && config.normalization != "zero_one" &&
      config.normalization != "imagenet")
    throw std::runtime_error("normalization must be none, zero_one, or imagenet");
  if (config.score_reduction != "mean" && config.score_reduction != "max")
    throw std::runtime_error("score_reduction must be mean or max");
  require_fraction(config.brightness_min, "brightness_min");
  require_fraction(config.brightness_max, "brightness_max");
  require_fraction(config.saturation_max, "saturation_max");
  require_fraction(config.shadow_max, "shadow_max");
  require_fraction(config.contrast_min, "contrast_min");
  if (config.input_width < 1 || config.input_height < 1 ||
      config.output_index < 0 || config.warmup_count < 0 ||
      config.benchmark_repetitions < 1 || config.parity_abs_tolerance < 0.0 ||
      config.parity_rel_tolerance < 0.0 ||
      config.brightness_min > config.brightness_max || config.review_margin < 0.0 ||
      config.sharpness_min < 0.0 || config.max_side < 1 ||
      config.gaussian_kernel < 1 || config.gaussian_kernel % 2 == 0 ||
      config.saturation_level < 0 || config.saturation_level > 255 ||
      config.shadow_level < 0 || config.shadow_level > 255)
    throw std::runtime_error("Config contains an invalid limit");
  return config;
}

std::vector<std::string> parse_csv_row(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        field += '"';
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(field);
      field.clear();
    } else if (ch != '\r' || quoted || i + 1 < line.size()) {
      field += ch;
    }
  }
  if (quoted) throw std::runtime_error("Unclosed quoted CSV field");
  fields.push_back(field);
  return fields;
}

std::unordered_map<std::string, double> read_scores(
    const fs::path& csv, const fs::path& image_root) {
  std::ifstream input(csv);
  if (!input) throw std::runtime_error("Cannot open prediction CSV: " + csv.string());
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("Empty prediction CSV");
  const auto header = parse_csv_row(line);
  auto find_column = [&](const std::string& name) {
    const auto it = std::find(header.begin(), header.end(), name);
    if (it == header.end()) throw std::runtime_error("Missing CSV column: " + name);
    return static_cast<std::size_t>(it - header.begin());
  };
  const auto path_col = find_column("image_path");
  const auto score_col = find_column("image_score");
  std::unordered_map<std::string, double> scores;
  std::size_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (trim(line).empty()) continue;
    const auto fields = parse_csv_row(line);
    if (fields.size() != header.size())
      throw std::runtime_error("Wrong CSV column count at line " +
                               std::to_string(line_number));
    fs::path path(fields[path_col]);
    if (path.empty()) throw std::runtime_error("Empty image_path in prediction CSV");
    if (path.is_relative()) path = image_root / path;
    const auto key = normalized_path(path);
    const double score = finite_number(trim(fields[score_col]), "image_score");
    if (!scores.emplace(key, score).second)
      throw std::runtime_error("Duplicate image_path in prediction CSV: " + key);
  }
  if (scores.empty()) throw std::runtime_error("Prediction CSV has no scores");
  return scores;
}

Metrics measure(const cv::Mat& input, const Config& config) {
  if (input.empty()) throw std::runtime_error("Empty image");
  cv::Mat gray;
  if (input.channels() == 3) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  else if (input.channels() == 1) gray = input;
  else throw std::runtime_error("Unsupported image channel count");
  if (gray.depth() != CV_8U) throw std::runtime_error("Expected 8-bit image");

  const int longest = std::max(gray.cols, gray.rows);
  if (longest > config.max_side) {
    const double scale = static_cast<double>(config.max_side) / longest;
    cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_AREA);
  }
  if (config.gaussian_kernel > 1)
    cv::GaussianBlur(gray, gray,
                     cv::Size(config.gaussian_kernel, config.gaussian_kernel), 0);

  cv::Scalar mean, stddev;
  cv::meanStdDev(gray, mean, stddev);
  cv::Mat saturation_mask, shadow_mask;
  cv::compare(gray, config.saturation_level, saturation_mask, cv::CMP_GE);
  cv::compare(gray, config.shadow_level, shadow_mask, cv::CMP_LE);
  const double pixels = static_cast<double>(gray.total());
  cv::Mat laplacian, grad_x, grad_y;
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar lap_mean, lap_std;
  cv::meanStdDev(laplacian, lap_mean, lap_std);
  cv::Sobel(gray, grad_x, CV_64F, 1, 0);
  cv::Sobel(gray, grad_y, CV_64F, 0, 1);
  const double high_frequency =
      (cv::mean(grad_x.mul(grad_x))[0] + cv::mean(grad_y.mul(grad_y))[0]) /
      (255.0 * 255.0);
  return {mean[0] / 255.0,
          cv::countNonZero(saturation_mask) / pixels,
          cv::countNonZero(shadow_mask) / pixels,
          stddev[0] / 255.0,
          lap_std[0] * lap_std[0] / (255.0 * 255.0),
          high_frequency};
}

Record inspect(const fs::path& image_path, ScoreProvider& provider,
               const Config& config, const fs::path& visualization_dir) {
  const auto started = std::chrono::steady_clock::now();
  Record result;
  result.image_path = image_path;
  std::vector<std::string> reasons;
  try {
    const auto decode_started = Clock::now();
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    result.decode_ms = elapsed_ms(decode_started);
    if (image.empty()) {
      reasons.emplace_back("unreadable_image");
      result.error_detail = "decode: cannot read " + image_path.string();
      result.inference_status = "not_run";
    } else {
      const auto quality_started = Clock::now();
      result.metrics = measure(image, config);
      result.quality_ms = elapsed_ms(quality_started);
      const auto& m = *result.metrics;
      if (m.brightness < config.brightness_min ||
          m.brightness > config.brightness_max) reasons.emplace_back("brightness");
      if (m.saturation_fraction > config.saturation_max)
        reasons.emplace_back("saturation");
      if (m.shadow_fraction > config.shadow_max) reasons.emplace_back("shadow");
      if (m.contrast < config.contrast_min) reasons.emplace_back("contrast");
      if (m.laplacian_sharpness < config.sharpness_min)
        reasons.emplace_back("sharpness");
      try {
        const auto score_result = provider.score(image_path, image);
        result.score = score_result.score;
        result.score_preprocess_ms = score_result.preprocessing_ms;
        result.score_inference_ms = score_result.inference_ms;
        result.score_postprocess_ms = score_result.postprocessing_ms;
        result.error_detail = score_result.error.empty() ? "" :
            "inference: " + image_path.string() + ": " + score_result.error;
        result.inference_status = score_result.error.empty() && result.score ? "ok" : "error";
        if (!score_result.anomaly_map.empty()) result.max_anomaly = result.score;
        if (!score_result.error.empty()) reasons.emplace_back("score_error");
        else if (!result.score) reasons.emplace_back("missing_score");
        else if (*result.score >= config.score_threshold)
          reasons.emplace_back("score_threshold");
        if (!visualization_dir.empty() && score_result.error.empty()) {
          const auto visualization_started = Clock::now();
          const std::string visual_decision = reasons.empty() ? "PASS" : "NG";
          try {
            result.visualization_path = render_visualization(
                image_path, image, score_result.anomaly_map, visual_decision,
                result.score, config.score_threshold, visualization_dir,
                config.max_side);
          } catch (const std::exception& error) {
            result.error_detail = "visualization: " + image_path.string() + ": " +
                                  error.what();
            reasons.emplace_back("visualization_error");
          }
          result.visualization_ms = elapsed_ms(visualization_started);
        }
      } catch (const std::exception& error) {
        result.error_detail = image_path.string() + ": " + error.what();
        result.inference_status = "error";
        reasons.emplace_back("score_error");
      }
    }
  } catch (const std::exception& error) {
    result.error_detail = image_path.string() + ": processing: " + error.what();
    reasons.emplace_back("processing_error");
  }
  result.decision = reasons.empty() ? "PASS" : "NG";
  for (const auto& reason : reasons) {
    if (!result.reason.empty()) result.reason += ';';
    result.reason += reason;
  }
  if (result.reason.empty()) result.reason = "within_limits";
  result.review = result.decision == "NG" ||
                  (result.score && std::abs(*result.score - config.score_threshold) <=
                                config.review_margin);
  result.processing_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  return result;
}

std::vector<fs::path> list_images(const fs::path& directory) {
  if (!fs::is_directory(directory))
    throw std::runtime_error("Input is not a directory: " + directory.string());
  std::vector<fs::path> paths;
  for (const auto& entry : fs::recursive_directory_iterator(directory)) {
    if (!entry.is_regular_file()) continue;
    std::string extension = entry.path().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
        extension == ".bmp" || extension == ".tif" || extension == ".tiff")
      paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) throw std::runtime_error("No supported images in input directory");
  return paths;
}

void write_reports(const fs::path& output_dir, const std::vector<Record>& records,
                   const Config& config) {
  fs::create_directories(output_dir);
  const auto csv_path = output_dir / "report.csv";
  const auto review_path = output_dir / "review_queue.csv";
  const auto json_path = output_dir / "report.json";
  const auto summary_path = output_dir / "summary.json";
  std::ofstream csv(csv_path), review(review_path), json(json_path), summary(summary_path);
  if (!csv || !review || !json || !summary)
    throw std::runtime_error("Cannot create reports in " + output_dir.string());
  csv << std::setprecision(17);
  review << std::setprecision(17);
  json << std::setprecision(17);
  summary << std::setprecision(17);
  const std::string header = "image_path,decision,reason,image_score,brightness,"
      "saturation_fraction,shadow_fraction,contrast,laplacian_sharpness,"
      "high_frequency_energy,score_preprocess_ms,score_inference_ms,"
      "processing_ms,error_detail,review,filename,threshold,inference_status,"
      "max_anomaly,localization_output_path,decode_ms,quality_ms,"
      "score_postprocess_ms,visualization_ms\n";
  csv << header;
  review << header;
  std::size_t passed = 0, rejected = 0, queued = 0;
  double total_ms = 0.0;
  json << "{\n  \"records\": [\n";
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto& r = records[i];
    passed += r.decision == "PASS";
    rejected += r.decision == "NG";
    queued += r.review;
    total_ms += r.processing_ms;
    std::ostringstream row;
    row << std::setprecision(17) << csv_quote(r.image_path.string()) << ','
        << r.decision << ',' << csv_quote(r.reason) << ',' << optional_number(r.score)
        << ',';
    if (r.metrics) {
      const auto& m = *r.metrics;
      row << m.brightness << ',' << m.saturation_fraction << ','
          << m.shadow_fraction << ',' << m.contrast << ','
          << m.laplacian_sharpness << ',' << m.high_frequency_energy;
    } else {
      row << ",,,,,";
    }
    row << ',' << r.score_preprocess_ms << ',' << r.score_inference_ms
        << ',' << r.processing_ms << ',' << csv_quote(r.error_detail)
        << ',' << (r.review ? "true" : "false")
        << ',' << csv_quote(r.image_path.filename().string())
        << ',' << config.score_threshold
        << ',' << csv_quote(r.inference_status)
        << ',' << optional_number(r.max_anomaly)
        << ',' << csv_quote(r.visualization_path.string())
        << ',' << r.decode_ms << ',' << r.quality_ms
        << ',' << r.score_postprocess_ms << ',' << r.visualization_ms << '\n';
    csv << row.str();
    if (r.review) review << row.str();

    json << "    {\"image_path\": " << json_quote(r.image_path.string())
         << ", \"decision\": " << json_quote(r.decision)
         << ", \"reason\": " << json_quote(r.reason)
         << ", \"image_score\": ";
    if (r.score) json << *r.score; else json << "null";
    json << ", \"metrics\": ";
    if (r.metrics) {
      const auto& m = *r.metrics;
      json << "{\"brightness\": " << m.brightness
           << ", \"saturation_fraction\": " << m.saturation_fraction
           << ", \"shadow_fraction\": " << m.shadow_fraction
           << ", \"contrast\": " << m.contrast
           << ", \"laplacian_sharpness\": " << m.laplacian_sharpness
           << ", \"high_frequency_energy\": " << m.high_frequency_energy << '}';
    } else {
      json << "null";
    }
    json << ", \"filename\": " << json_quote(r.image_path.filename().string())
         << ", \"threshold\": " << config.score_threshold
         << ", \"inference_status\": " << json_quote(r.inference_status)
         << ", \"max_anomaly\": ";
    if (r.max_anomaly) json << *r.max_anomaly; else json << "null";
    json << ", \"localization_output_path\": "
         << json_quote(r.visualization_path.string())
         << ", \"decode_ms\": " << r.decode_ms
         << ", \"quality_ms\": " << r.quality_ms
         << ", \"score_postprocess_ms\": " << r.score_postprocess_ms
         << ", \"visualization_ms\": " << r.visualization_ms
         << ", \"score_preprocess_ms\": " << r.score_preprocess_ms
         << ", \"score_inference_ms\": " << r.score_inference_ms
         << ", \"processing_ms\": " << r.processing_ms
         << ", \"error_detail\": " << json_quote(r.error_detail)
         << ", \"review\": " << (r.review ? "true" : "false") << '}';
    if (i + 1 < records.size()) json << ',';
    json << '\n';
  }
  json << "  ],\n  \"summary\": ";
  std::ostringstream summary_body;
  summary_body << std::setprecision(17)
               << "{\"images\": " << records.size()
               << ", \"pass\": " << passed
               << ", \"ng\": " << rejected
               << ", \"review_queue\": " << queued
               << ", \"total_processing_ms\": " << total_ms
               << ", \"mean_processing_ms\": "
               << (records.empty() ? 0.0 : total_ms / records.size())
               << ", \"score_threshold\": " << config.score_threshold
               << ", \"provider_type\": " << json_quote(config.provider_type)
               << ", \"model_path\": " << json_quote(config.model_path.string())
               << ", \"model_sha256\": " << json_quote(config.model_sha256)
               << ", \"input_width\": " << config.input_width
               << ", \"input_height\": " << config.input_height
               << ", \"normalization\": " << json_quote(config.normalization)
               << ", \"output_index\": " << config.output_index
               << ", \"score_reduction\": " << json_quote(config.score_reduction)
               << '}';
  json << summary_body.str() << "\n}\n";
  summary << summary_body.str() << '\n';
  csv.flush(); review.flush(); json.flush(); summary.flush();
  check_stream(csv, csv_path);
  check_stream(review, review_path);
  check_stream(json, json_path);
  check_stream(summary, summary_path);
}
}  // namespace aoi
