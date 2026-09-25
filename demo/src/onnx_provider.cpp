#include "aoi/aoi.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

namespace aoi {
namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Torchvision's PIL BILINEAR resize antialiases large downscales. OpenCV's
// INTER_LINEAR samples instead and materially changes scores on 4224px images.
// A separable triangle filter matches the PIL 8-bit resize within one LSB.
struct ResizeWeights {
  int first = 0;
  std::vector<double> values;
};

std::vector<ResizeWeights> triangle_weights(int source, int target) {
  const double scale = static_cast<double>(source) / target;
  const double filter_scale = std::max(1.0, scale);
  std::vector<ResizeWeights> table(static_cast<std::size_t>(target));
  for (int i = 0; i < target; ++i) {
    const double center = (i + 0.5) * scale;
    const int first = std::max(0, static_cast<int>(center - filter_scale + 0.5));
    const int last = std::min(source, static_cast<int>(center + filter_scale + 0.5));
    if (last <= first) throw std::runtime_error("Invalid resize support");
    auto& entry = table[static_cast<std::size_t>(i)];
    entry.first = first;
    double sum = 0.0;
    for (int j = first; j < last; ++j) {
      const double value = std::max(0.0, 1.0 -
          std::abs((j + 0.5 - center) / filter_scale));
      entry.values.push_back(value);
      sum += value;
    }
    for (double& value : entry.values) value /= sum;
  }
  return table;
}

cv::Mat pil_bilinear_resize(const cv::Mat& source, int width, int height) {
  const auto horizontal_weights = triangle_weights(source.cols, width);
  const auto vertical_weights = triangle_weights(source.rows, height);
  cv::Mat horizontal(source.rows, width, CV_8UC3);
  for (int y = 0; y < source.rows; ++y) {
    const auto* input = source.ptr<cv::Vec3b>(y);
    auto* output = horizontal.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width; ++x) {
      const auto& entry = horizontal_weights[static_cast<std::size_t>(x)];
      double sums[3]{};
      for (std::size_t k = 0; k < entry.values.size(); ++k) {
        const auto& pixel = input[entry.first + static_cast<int>(k)];
        for (int channel = 0; channel < 3; ++channel)
          sums[channel] += pixel[channel] * entry.values[k];
      }
      for (int channel = 0; channel < 3; ++channel)
        output[x][channel] = cv::saturate_cast<unsigned char>(
            std::floor(sums[channel] + 0.5));
    }
  }
  cv::Mat resized(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    const auto& entry = vertical_weights[static_cast<std::size_t>(y)];
    auto* output = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width; ++x) {
      double sums[3]{};
      for (std::size_t k = 0; k < entry.values.size(); ++k) {
        const auto& pixel = horizontal.ptr<cv::Vec3b>(
            entry.first + static_cast<int>(k))[x];
        for (int channel = 0; channel < 3; ++channel)
          sums[channel] += pixel[channel] * entry.values[k];
      }
      for (int channel = 0; channel < 3; ++channel)
        output[x][channel] = cv::saturate_cast<unsigned char>(
            std::floor(sums[channel] + 0.5));
    }
  }
  return resized;
}

class OnnxScoreProvider final : public ScoreProvider {
 public:
  explicit OnnxScoreProvider(const Config& config)
      : config_(config), env_(ORT_LOGGING_LEVEL_WARNING, "industrial_aoi") {
    if (!fs::is_regular_file(config_.model_path))
      throw std::runtime_error("ONNX model not found: " + config_.model_path.string());
    options_.SetIntraOpNumThreads(1);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
      session_ = Ort::Session(env_, config_.model_path.string().c_str(), options_);
      validate_metadata();
    } catch (const Ort::Exception& error) {
      throw std::runtime_error(std::string("ONNX model initialization failed: ") +
                               error.what());
    }
  }

  ScoreResult score(const fs::path&, const cv::Mat& bgr_image) override {
    ScoreResult result;
    try {
      if (bgr_image.empty() || bgr_image.type() != CV_8UC3)
        throw std::runtime_error("Expected nonempty 8-bit BGR image");
      const auto pre_started = Clock::now();
      cv::Mat resized, rgb;
      if (config_.provider_type == "efficientad")
        resized = pil_bilinear_resize(bgr_image, config_.input_width,
                                      config_.input_height);
      else
        cv::resize(bgr_image, resized,
                   cv::Size(config_.input_width, config_.input_height),
                   0, 0, cv::INTER_LINEAR);
      cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
      const std::size_t area = static_cast<std::size_t>(config_.input_width) *
                               static_cast<std::size_t>(config_.input_height);
      std::vector<float> tensor(3 * area);
      constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
      constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};
      for (int y = 0; y < rgb.rows; ++y) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < rgb.cols; ++x) {
          const auto pixel_index = static_cast<std::size_t>(y) * rgb.cols + x;
          for (int channel = 0; channel < 3; ++channel) {
            float value = static_cast<float>(row[x][channel]);
            if (config_.normalization != "none") value /= 255.0f;
            if (config_.normalization == "imagenet")
              value = (value - mean[channel]) / stddev[channel];
            tensor[static_cast<std::size_t>(channel) * area + pixel_index] = value;
          }
        }
      }
      result.preprocessing_ms = milliseconds(pre_started);

      const auto infer_started = Clock::now();
      const std::vector<int64_t> shape = {1, 3, config_.input_height,
                                          config_.input_width};
      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      auto input = Ort::Value::CreateTensor<float>(memory, tensor.data(),
                                                   tensor.size(), shape.data(),
                                                   shape.size());
      const char* input_names[] = {input_name_.c_str()};
      const char* output_names[] = {output_name_.c_str()};
      auto output = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                 output_names, 1);
      if (output.size() != 1 || !output[0].IsTensor())
        throw std::runtime_error("ONNX output is not one tensor");
      result.inference_ms = milliseconds(infer_started);
      const auto post_started = Clock::now();
      const auto info = output[0].GetTensorTypeAndShapeInfo();
      const auto count = info.GetElementCount();
      if (count == 0) throw std::runtime_error("ONNX output tensor is empty");
      const float* values = output[0].GetTensorData<float>();
      if (config_.provider_type == "efficientad") {
        const auto map_shape = info.GetShape();
        if (map_shape != std::vector<int64_t>({1, 1, 56, 56}))
          throw std::runtime_error("EfficientAD output must be [1,1,56,56]");
        cv::Mat low_map(56, 56, CV_32F, const_cast<float*>(values));
        cv::Mat padded;
        cv::copyMakeBorder(low_map, padded, 4, 4, 4, 4,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::resize(padded, result.anomaly_map, bgr_image.size(),
                   0, 0, cv::INTER_LINEAR);
        double max_value = -std::numeric_limits<double>::infinity();
        for (int y = 0; y < result.anomaly_map.rows; ++y) {
          const auto* row = result.anomaly_map.ptr<float>(y);
          for (int x = 0; x < result.anomaly_map.cols; ++x) {
            if (!std::isfinite(row[x]))
              throw std::runtime_error("EfficientAD map contains non-finite values");
            max_value = std::max(max_value, static_cast<double>(row[x]));
          }
        }
        result.score = max_value;
        result.postprocessing_ms = milliseconds(post_started);
        return result;
      }
      double reduced = config_.score_reduction == "max"
                           ? -std::numeric_limits<double>::infinity()
                           : 0.0;
      for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]))
          throw std::runtime_error("ONNX output contains non-finite values");
        if (config_.score_reduction == "max")
          reduced = std::max(reduced, static_cast<double>(values[i]));
        else
          reduced += values[i];
      }
      if (config_.score_reduction == "mean") reduced /= count;
      result.score = reduced;
      result.postprocessing_ms = milliseconds(post_started);
    } catch (const std::exception& error) {
      result.error = error.what();
    }
    return result;
  }

  std::string name() const override { return config_.provider_type; }

 private:
  void validate_metadata() {
    if (session_.GetInputCount() != 1)
      throw std::runtime_error("ONNX model must have exactly one input");
    if (config_.output_index >= static_cast<int>(session_.GetOutputCount()))
      throw std::runtime_error("ONNX output_index is outside model outputs");
    const auto input_type = session_.GetInputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("ONNX input must be float32");
    const auto shape = input_info.GetShape();
    if (shape.size() != 4 || (shape[0] != 1 && shape[0] != -1) ||
        shape[1] != 3 ||
        (shape[2] != config_.input_height && shape[2] != -1) ||
        (shape[3] != config_.input_width && shape[3] != -1))
      throw std::runtime_error("ONNX input must match NCHW [1,3,input_height,input_width]");
    const auto output_type = session_.GetOutputTypeInfo(config_.output_index);
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      throw std::runtime_error("ONNX output must be float32");
    if (config_.provider_type == "efficientad" &&
        output_info.GetShape() != std::vector<int64_t>({1, 1, 56, 56}))
      throw std::runtime_error("EfficientAD model metadata must output [1,1,56,56]");
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(config_.output_index, allocator).get();
  }

  Config config_;
  Ort::Env env_;
  Ort::SessionOptions options_;
  Ort::Session session_{nullptr};
  std::string input_name_;
  std::string output_name_;
};
}  // namespace

std::unique_ptr<ScoreProvider> make_onnx_provider(const Config& config) {
  return std::make_unique<OnnxScoreProvider>(config);
}
}  // namespace aoi
