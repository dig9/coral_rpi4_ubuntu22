#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "edgetpu_c.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

/**
 * The following 'string_format' was copied from the address below.
 * https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf?page=1&tab=scoredesc#tab-top
 */
template <typename... Args>
std::string string_format(const std::string &format, Args... args)
{
    size_t size =
        snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(),
                       buf.get() + size - 1); // We don't want the '\0' inside
}

// Precomp option to patch.
#define PATCH_BY_JICHOI
#ifdef PATCH_BY_JICHOI
#define CLOCKS_PER_MSEC (CLOCKS_PER_SEC / 1000)
#define COMPLEXITY_INVOKE_NAME "complexity invoke"
#define ELAPSE_CHECKER_INIT(strName, varName)                              \
    int count_##varName = 0;                                               \
    clock_t startTime_##varName = clock();                                 \
    clock_t lastTime = startTime_##varName;                                \
    clock_t curTime = lastTime;                                            \
    clock_t gapTime;                                                       \
    std::string checkerName = strName;                                     \
    std::cout << "!!! [" << checkerName                                    \
              << "] elapse checker initialized: " << checkerName << " !!!" \
              << std::endl;

#define ELAPSE_CHECKER_CHECK(varName)                                                                                   \
    curTime = clock();                                                                                                  \
    gapTime = curTime - lastTime;                                                                                       \
    std::cout << "!!! [" << checkerName << "] line: " << __LINE__                                                       \
              << ", cnt: " << ++count_##varName << " gap: " << gapTime / CLOCKS_PER_MSEC << "ms, elapsed from init: " \
              << (curTime - startTime_##varName) / CLOCKS_PER_MSEC << " ms !!!"                                         \
              << std::endl;                                                                                             \
    lastTime = curTime;

#define prt_err(str)                                                          \
    do                                                                        \
    {                                                                         \
        std::cerr << "error:[" << str << "]: " << __FILE__ << ":" << __LINE__ \
                  << std::endl;                                               \
    } while (0)
#define prt_info(str)                                     \
    do                                                    \
    {                                                     \
        std::cout << "info:[" << str << "]" << std::endl; \
    } while (0)
#else
#define string_format(x)
#define CLOCKS_PER_MSEC
#define COMPLEXITY_INVOKE_NAME
#define ELAPSE_CHECKER_INIT(strName, varName)
#define ELAPSE_CHECKER_CHECK(varName)
#define prt_err(str)
#define prt_info(str)
#endif

namespace
{
    constexpr size_t kBmpFileHeaderSize = 14;
    constexpr size_t kBmpInfoHeaderSize = 40;
    constexpr size_t kBmpHeaderSize = kBmpFileHeaderSize + kBmpInfoHeaderSize;

    int32_t ToInt32(const char p[4])
    {
        // fixed a critical bug from google.
        return ((p[3] << 24) & (0xFF000000) | (p[2] << 16) & (0x00FF0000) |
                (p[1] << 8) & (0x0000FF00) | p[0] & (0x000000FF)) &
               0xFFFFFFFF;
    }

    std::vector<uint8_t> ReadBmpImage(const char *filename,
                                      int *out_width = nullptr,
                                      int *out_height = nullptr,
                                      int *out_channels = nullptr)
    {
        assert(filename);

        std::ifstream file(filename, std::ios::binary);
        if (!file)
        {
            prt_err("failed to open input stream");
            return {}; // Open failed.
        }

        char header[kBmpHeaderSize];
        memset(header, 0, kBmpFileHeaderSize);
        if (!file.read(header, sizeof(header)))
        {
            prt_err("invalid header size");
            return {}; // Read failed.
        }

        const char *file_header = header;
        const char *info_header = header + kBmpFileHeaderSize;

        if (file_header[0] != 'B' || file_header[1] != 'M')
        {
            prt_err("failed to check bitmap marker");
            return {}; // Invalid file type.
        }

        const int channels = info_header[14] / 8;
        if (channels != 1 && channels != 3)
        {
            prt_err(string_format("no support channel count %d", channels));
            return {}; // invalid channel size
        }

        if (ToInt32(&info_header[16]) != 0)
        {
            prt_err("error: Unsupported compression.");
            return {}; // Unsupported compression.
        }

        const uint32_t offset = ToInt32(&file_header[10]);

        if (offset > kBmpHeaderSize &&
            !file.seekg(offset - kBmpHeaderSize, std::ios::cur))
        {
            prt_err("can't seek to info header");
            return {}; // Seek failed.
        }

        int width = ToInt32(&info_header[4]);
        if (width < 0)
        {
            prt_err("invalid width");
            return {}; // Invalid width.
        }
        int height = ToInt32(&info_header[8]);
        const bool top_down = height < 0;
        if (top_down)
            height = -height;

        const int line_bytes = width * channels;
        prt_info(string_format("bmp info: width = %d, hegith = %d, channels = %d, "
                               "line_bytes= %d, top_down = %d",
                               width, height, channels, line_bytes, top_down));

        const int line_padding_bytes =
            4 * ((8 * channels * width + 31) / 32) - line_bytes;

        std::vector<uint8_t> image(line_bytes * height);
        for (int i = 0; i < height; ++i)
        {

            uint8_t *line = &image[(top_down ? i : (height - 1 - i)) * line_bytes];

            if (!file.read(reinterpret_cast<char *>(line), line_bytes))
            {
                prt_err(string_format("failed to read linebyte %d", line_bytes));
                return {}; // Read failed.
            }
            if (!file.seekg(line_padding_bytes, std::ios::cur))
            {
                prt_err("error: can't seek to padding bytes");
                return {}; // Seek failed.
            }
            if (channels == 3)
            {
                for (int j = 0; j < width; ++j)
                    std::swap(line[3 * j], line[3 * j + 2]);
            }
        }

        if (out_width)
            *out_width = width;
        if (out_height)
            *out_height = height;
        if (out_channels)
            *out_channels = channels;
        return image;
    }

    std::vector<std::string> ReadLabels(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file)
            return {}; // Open failed.

        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);)
            lines.emplace_back(line);
        return lines;
    }

    std::string GetLabel(const std::vector<std::string> &labels, int label)
    {
        if (label >= 0 && label < labels.size())
            return labels[label];
        return std::to_string(label);
    }

    std::vector<float> Dequantize(const TfLiteTensor &tensor)
    {
        const auto *data = reinterpret_cast<const uint8_t *>(tensor.data.data);
        std::vector<float> result(tensor.bytes);
        for (int i = 0; i < tensor.bytes; ++i)
            result[i] = tensor.params.scale * (data[i] - tensor.params.zero_point);
        return result;
    }

    std::vector<std::pair<int, float>> Sort(const std::vector<float> &scores,
                                            float threshold)
    {
        std::vector<const float *> ptrs(scores.size());
        std::iota(ptrs.begin(), ptrs.end(), scores.data());
        auto end = std::partition(ptrs.begin(), ptrs.end(),
                                  [=](const float *v)
                                  { return *v >= threshold; });
        std::sort(ptrs.begin(), end,
                  [](const float *a, const float *b)
                  { return *a > *b; });

        std::vector<std::pair<int, float>> result;
        result.reserve(end - ptrs.begin());
        for (auto it = ptrs.begin(); it != end; ++it)
            result.emplace_back(*it - scores.data(), **it);
        return result;
    }
} // namespace

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        std::cerr << argv[0]
                  << " <model_file> <label_file> <image_file> <threshold:float> <tpu "
                     "on:1,off:0> <loopCount:decimal> "
                  << std::endl;

        std::cerr << argc << std::endl;
        return 1;
    }

    const std::string model_file = argv[1];
    const std::string label_file = argv[2];
    const std::string image_file = argv[3];
    const float threshold = std::stof(argv[4]);
    const int edgetpuOn = std::stoi(argv[5]);
    const int loopCount = std::stoi(argv[6]) > 0 ? std::stoi(argv[6]) : 1;

    // Load labels.
    auto labels = ReadLabels(label_file);
    if (labels.empty())
    {
        std::cerr << "Cannot read labels from " << label_file << std::endl;
        return 1;
    }
    // Load image.
    int image_bpp, image_width, image_height;
    auto image =
        ReadBmpImage(image_file.c_str(), &image_width, &image_height, &image_bpp);
    if (image.empty())
    {
        std::cerr << "Cannot read image from " << image_file << std::endl;
        return 1;
    }
    prt_info(string_format("jichoi: readImage done: height/width = %d / %d", image_height, image_width));

    // Load model.
    auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model)
    {
        std::cerr << "Cannot read model from " << model_file << std::endl;
        return 1;
    }

    prt_info("jichoi: read model Done");
    // Create interpreter.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk)
    {
        std::cerr << "Cannot create interpreter" << std::endl;
        return 1;
    }

    // Find TPU device.
    // if(edgetpuOn){
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices == 0)
    {
        std::cerr << "No connected TPU found" << std::endl;
        return 1;
    }
    auto &device = devices.get()[0];

    prt_info("jichoi: build interpreter");
    TfLiteDelegate *delegate =
        edgetpu_create_delegate(device.type, device.path, nullptr, 0);

    prt_info(string_format("jichoi: create delegator done: [0x%p]", delegate));
    // interpreter->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate});
    interpreter->ModifyGraphWithDelegate(delegate);
    prt_info("jichoi: set delegator done");
    // }
    // Allocate tensors.
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Cannot allocate interpreter tensors" << std::endl;
        return 1;
    }
    prt_info("jichoi: allocate tensors done");
    // Set interpreter input.
    const auto *input_tensor = interpreter->input_tensor(0);
    prt_info(string_format("!!! this model requires heigth/width = %d / %d",
                           input_tensor->dims->data[1],
                           input_tensor->dims->data[2]));
    prt_info(string_format("!!! this model requires dimension: %d",
                           input_tensor->dims->data[3]));

    if (input_tensor->type != kTfLiteUInt8 ||          //
        input_tensor->dims->data[0] != 1 ||            //
        input_tensor->dims->data[1] != image_height || //
        input_tensor->dims->data[2] != image_width ||  //
        input_tensor->dims->data[3] != image_bpp)
    {
        std::cerr << "Input tensor shape does not match input image" << std::endl;

        return 1;
    }
    prt_info("jichoi: check image and tensors size  done");
    std::copy(image.begin(), image.end(),
              interpreter->typed_input_tensor<uint8_t>(0));

    prt_info("jichoi: copy image to tensors done");

    ELAPSE_CHECKER_INIT(COMPLEXITY_INVOKE_NAME, ck1)
    // Run inference.
    for (int l = 0; l < loopCount; ++l)
    {
        if (interpreter->Invoke() != kTfLiteOk)
        {
            std::cerr << "Cannot invoke interpreter" << std::endl;
            return 1;
        }
        prt_info(string_format("jichoi: Invoke done cnt: %d", l));
        ELAPSE_CHECKER_CHECK(ck1)
    }
#if 1
    auto rects = interpreter->typed_output_tensor<float>(0);
    auto classes = interpreter->typed_output_tensor<float>(1);

    auto scores = interpreter->typed_output_tensor<float>(2);
    auto numDetect = interpreter->typed_output_tensor<float>(3);
    const auto size = interpreter->output_tensor(0)->dims->size;

    prt_info("");
    prt_info("_______________ Output __________________");
    prt_info(string_format("number of detection: %f [%p]", *numDetect, numDetect));
    prt_info(string_format("output dims .size: %d, .bytes: %d", size, interpreter->output_tensor(0)->bytes));

    for (int idx = 0; idx < *numDetect; idx++)
    {
        if (scores[idx] < threshold)
            continue;

        int ymin = (int)(rects[0] * image_height);
        int xmin = (int)(rects[1] * image_width);
        int h = (int)(rects[2] * image_height) - ymin;
        int w = (int)(rects[3] * image_width) - xmin;
        std::cout << "classes(" << idx << "): " << classes[idx] << ", label(" << idx
                  << "): " << GetLabel(labels, classes[idx]) << ", scores(" << idx
                  << "): " << scores[idx] << string_format(" y,x,,h,w = [%d,%d,%d,%d]", ymin, xmin, h, w) << std::endl;
    }

#else
    // jichoi. what are you doing?
    // Get interpreter output.
    auto results = Sort(Dequantize(*interpreter->output_tensor(0)), threshold);
    for (auto &result : results)
        std::cout << std::setw(7) << std::fixed << std::setprecision(5)
                  << result.second << GetLabel(labels, result.first) << std::endl;
#endif
    return 0;
}
