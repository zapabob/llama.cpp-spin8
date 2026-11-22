#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using json = nlohmann::ordered_json;

struct options {
    std::string model;
    std::string output_prefix;
    std::string device;
    int ngl = 0;
    int threads = 2;
    uint32_t context = 128;
    ggml_type cache_type = GGML_TYPE_F16;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO;
    bool no_alloc = false;
    std::vector<llama_token> prompt = {1, 4, 7, 9, 13};
    std::vector<llama_token> decode = {17, 21, 25};
};

struct observations {
    std::ofstream log;
    std::mutex mutex;
    std::map<std::tuple<std::string, std::string, std::string>, uint64_t> placements;
    json examples = json::array();
};

static void print_usage(const char * executable) {
    std::cout << "Usage: " << executable << " --model model.gguf --output-prefix result [options]\n"
              << "  --ngl N                 GPU layers; 0 forces CPU (default: 0)\n"
              << "  --device NAME           Select a registered backend device\n"
              << "  --threads N             CPU threads (default: 2)\n"
              << "  --ctx N                 Context size (default: 128)\n"
              << "  --cache-type f16|f32    K/V cache storage type (default: f16)\n"
              << "  --flash-attn off|on|auto (default: off)\n"
              << "  --mmap | --no-mmap      Force mapped or ordinary model loading\n"
              << "  --no-alloc              Shape-only memory-fit graph; no inference\n"
              << "  --prompt-tokens CSV     Prefill IDs (default: 1,4,7,9,13)\n"
              << "  --decode-tokens CSV     Forced decode IDs (default: 17,21,25)\n"
              << "Writes .f32 full logits, .json metadata, and .log native messages.\n"
              << "This is a fixed-token execution test, not a model quality evaluation.\n";
}

static int parse_int(const std::string & value, int minimum, int maximum) {
    size_t used = 0;
    const long long result = std::stoll(value, &used);
    if (used != value.size() || result < minimum || result > maximum) {
        throw std::runtime_error("integer out of range: " + value);
    }
    return static_cast<int>(result);
}

static std::vector<llama_token> parse_tokens(const std::string & value) {
    std::vector<llama_token> result;
    std::istringstream input(value);
    std::string token;
    if (value.empty() || value.back() == ',') {
        throw std::runtime_error("token list must not be empty or end with a comma");
    }
    while (std::getline(input, token, ',')) {
        result.push_back(parse_int(token, 0, std::numeric_limits<int32_t>::max()));
    }
    return result;
}

static options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-alloc") {
            result.no_alloc = true;
            continue;
        }
        if (arg == "--mmap" || arg == "--no-mmap") {
            result.load_mode = arg == "--mmap" ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--model" || arg == "-m") {
            result.model = value;
        } else if (arg == "--output-prefix" || arg == "-o") {
            result.output_prefix = value;
        } else if (arg == "--ngl" || arg == "-ngl") {
            result.ngl = parse_int(value, -1, std::numeric_limits<int32_t>::max());
        } else if (arg == "--device") {
            result.device = value;
        } else if (arg == "--threads") {
            result.threads = parse_int(value, 1, 1024);
        } else if (arg == "--ctx") {
            result.context = parse_int(value, 1, std::numeric_limits<int32_t>::max());
        } else if (arg == "--cache-type") {
            if (value == "f16") {
                result.cache_type = GGML_TYPE_F16;
            } else if (value == "f32") {
                result.cache_type = GGML_TYPE_F32;
            } else {
                throw std::runtime_error("invalid cache type: " + value);
            }
        } else if (arg == "--prompt-tokens") {
            result.prompt = parse_tokens(value);
        } else if (arg == "--decode-tokens") {
            result.decode = parse_tokens(value);
        } else if (arg == "--flash-attn") {
            if (value == "off") {
                result.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            } else if (value == "on") {
                result.flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            } else if (value == "auto") {
                result.flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
            } else {
                throw std::runtime_error("invalid flash-attn mode: " + value);
            }
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (result.model.empty() || result.output_prefix.empty()) {
        throw std::runtime_error("--model and --output-prefix are required");
    }
    if (result.prompt.size() + result.decode.size() > result.context) {
        throw std::runtime_error("token sequence exceeds the requested context");
    }
    return result;
}

static void log_callback(ggml_log_level, const char * message, void * userdata) {
    auto & state = *static_cast<observations *>(userdata);
    std::lock_guard<std::mutex> lock(state.mutex);
    state.log << message;
    state.log.flush();
    std::fputs(message, stderr);
}

static bool eval_callback(ggml_tensor * tensor, bool ask, void * userdata) {
    const std::string op = ggml_op_name(tensor->op);
    const bool watch = op.find("SPIN8") != std::string::npos || op == "MUL_MAT" || op == "GET_ROWS";
    if (ask) {
        return watch;
    }
    if (!watch) {
        return true;
    }
    auto & state = *static_cast<observations *>(userdata);
    auto buffer = tensor->buffer;
    if (!buffer && tensor->view_src) {
        buffer = tensor->view_src->buffer;
    }
    std::string buffer_name = "unknown";
    std::string device_name = "unknown";
    if (buffer) {
        buffer_name = ggml_backend_buffer_name(buffer);
        const auto buft = ggml_backend_buffer_get_type(buffer);
        const auto device = ggml_backend_buft_get_device(buft);
        if (device) {
            device_name = ggml_backend_dev_name(device);
        } else {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                const auto candidate = ggml_backend_dev_get(i);
                if (ggml_backend_dev_buffer_type(candidate) == buft) {
                    device_name = ggml_backend_dev_name(candidate);
                    break;
                }
            }
        }
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.placements[std::make_tuple(op, device_name, buffer_name)];
    if (state.examples.size() < 32) {
        state.examples.push_back({
            {"name", tensor->name}, {"op", op}, {"device", device_name}, {"buffer", buffer_name},
            {"shape_ggml", {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}},
        });
    }
    return true;
}

static const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
    }
    return "unknown";
}

static json loaded_runtime_libraries() {
    std::set<std::string> paths;
#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        const auto start = line.find('/');
        if (start == std::string::npos) {
            continue;
        }
        const std::string path = line.substr(start);
        const auto name = std::filesystem::path(path).filename().string();
        if (name.find("libllama") == 0 || name.find("libggml") == 0) {
            paths.insert(path);
        }
    }
#endif
    return paths;
}

static void set_batch(llama_batch_ext * batch, const std::vector<llama_token> & tokens, llama_pos start) {
    llama_batch_ext_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos position = start + static_cast<llama_pos>(i);
        if (index < 0 || !llama_batch_ext_set_pos(batch, index, &position) ||
            !llama_batch_ext_set_output_logits(batch, index, true)) {
            throw std::runtime_error("failed to populate the input batch");
        }
    }
}

static void write_row(llama_context * ctx, int index, int n_vocab, std::ofstream & output,
                      json & rows, const char * phase, llama_token token, llama_pos position) {
    const float * values = llama_get_logits_ith(ctx, index);
    if (!values) {
        throw std::runtime_error("requested logits are unavailable");
    }
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    double sum_squared = 0;
    int top1 = 0;
    for (int i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(values[i])) {
            throw std::runtime_error("non-finite logits at input position " + std::to_string(position));
        }
        minimum = std::min(minimum, values[i]);
        if (values[i] > maximum) {
            maximum = values[i];
            top1 = i;
        }
        sum_squared += static_cast<double>(values[i]) * values[i];
    }
    output.write(reinterpret_cast<const char *>(values), static_cast<std::streamsize>(n_vocab) * sizeof(float));
    if (!output) {
        throw std::runtime_error("failed to write logits");
    }
    rows.push_back({
        {"phase", phase}, {"input_token", token}, {"position", position},
        {"top1_token", top1}, {"minimum", minimum}, {"maximum", maximum},
        {"l2_norm", std::sqrt(sum_squared)},
    });
}

static void run(const options & opts, observations & state, json & report) {
    ggml_backend_load_all();
    llama_backend_init();
    report["system_info"] = llama_print_system_info();
    report["available_devices"] = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto device = ggml_backend_dev_get(i);
        report["available_devices"].push_back({
            {"name", ggml_backend_dev_name(device)},
            {"description", ggml_backend_dev_description(device)},
            {"type", device_type_name(ggml_backend_dev_type(device))},
        });
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = opts.ngl;
    model_params.load_mode = opts.load_mode;
    model_params.load_mtp = false;
    model_params.no_alloc = opts.no_alloc;
    std::vector<ggml_backend_dev_t> devices;
    if (opts.ngl == 0) {
        if (!opts.device.empty()) {
            auto requested = ggml_backend_dev_by_name(opts.device.c_str());
            if (!requested || ggml_backend_dev_type(requested) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::runtime_error("--ngl 0 forces CPU; do not select a non-CPU device");
            }
        }
        devices = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
    } else if (!opts.device.empty()) {
        auto requested = ggml_backend_dev_by_name(opts.device.c_str());
        if (!requested) {
            throw std::runtime_error("requested device is unavailable: " + opts.device);
        }
        devices = {requested, nullptr};
    }
    if (!devices.empty()) {
        if (!devices.front()) {
            throw std::runtime_error("CPU backend is unavailable");
        }
        model_params.devices = devices.data();
    }
    llama_model_ptr model(llama_model_load_from_file(opts.model.c_str(), model_params));
    if (!model) {
        throw std::runtime_error("model loading failed; see native log");
    }
    const auto vocab = llama_model_get_vocab(model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("invalid vocabulary size");
    }
    for (const auto & tokens : {opts.prompt, opts.decode}) {
        for (const auto token : tokens) {
            if (token >= n_vocab) {
                throw std::runtime_error("input token exceeds vocabulary size");
            }
        }
    }
    report["n_vocab"] = n_vocab;
    report["n_embd"] = llama_model_n_embd(model.get());
    report["n_layer"] = llama_model_n_layer(model.get());
    char text[512] = {};
    llama_model_desc(model.get(), text, sizeof(text));
    report["model_description"] = text;
    for (const auto key : {"general.architecture", "general.name", "general.file_type"}) {
        if (llama_model_meta_val_str(model.get(), key, text, sizeof(text)) >= 0) {
            report["model_metadata"][key] = text;
        }
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = opts.context;
    context_params.n_batch = static_cast<uint32_t>(opts.prompt.size());
    context_params.n_ubatch = context_params.n_batch;
    context_params.n_seq_max = 1;
    context_params.n_threads = opts.threads;
    context_params.n_threads_batch = opts.threads;
    context_params.type_k = opts.cache_type;
    context_params.type_v = opts.cache_type;
    context_params.flash_attn_type = opts.flash_attn;
    context_params.no_perf = false;
    context_params.cb_eval = eval_callback;
    context_params.cb_eval_user_data = &state;
    if (opts.ngl == 0) {
        context_params.offload_kqv = false;
        context_params.op_offload = false;
    }
    llama_context_ptr ctx(llama_init_from_model(model.get(), context_params));
    if (!ctx) {
        throw std::runtime_error("context creation failed; see native log");
    }
    report["n_ctx_actual"] = llama_n_ctx(ctx.get());
    report["cache_type_k"] = ggml_type_name(context_params.type_k);
    report["cache_type_v"] = ggml_type_name(context_params.type_v);
    if (opts.no_alloc) {
        report["scope"] = "shape-only model and context graph construction; no tensor execution or numerical validation";
        report["no_alloc_graph_created"] = true;
        report["logits_file"] = nullptr;
        report["loaded_runtime_libraries"] = loaded_runtime_libraries();
        report["status"] = "PASS";
        return;
    }
    llama_batch_ext_ptr batch(llama_batch_ext_init(ctx.get()));
    if (!batch) {
        throw std::runtime_error("batch allocation failed");
    }
    std::ofstream output(opts.output_prefix + ".f32", std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open logits output");
    }
    report["rows"] = json::array();
    set_batch(batch.get(), opts.prompt, 0);
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        throw std::runtime_error("prompt evaluation failed; see native log");
    }
    for (size_t i = 0; i < opts.prompt.size(); ++i) {
        write_row(ctx.get(), static_cast<int>(i), n_vocab, output, report["rows"], "prefill",
                  opts.prompt[i], static_cast<llama_pos>(i));
    }
    for (size_t i = 0; i < opts.decode.size(); ++i) {
        const auto position = static_cast<llama_pos>(opts.prompt.size() + i);
        set_batch(batch.get(), {opts.decode[i]}, position);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("forced decode failed; see native log");
        }
        write_row(ctx.get(), 0, n_vocab, output, report["rows"], "forced_decode", opts.decode[i], position);
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finish logits output");
    }
    report["logits_shape"] = {report["rows"].size(), n_vocab};
    report["finite_logits"] = true;
    report["loaded_runtime_libraries"] = loaded_runtime_libraries();
    report["observed_output_buffer_placements"] = json::array();
    for (const auto & item : state.placements) {
        report["observed_output_buffer_placements"].push_back({
            {"op", std::get<0>(item.first)}, {"device", std::get<1>(item.first)},
            {"buffer", std::get<2>(item.first)}, {"completed_observations", item.second},
        });
    }
    report["placement_examples"] = state.examples;
    report["placement_note"] = "Observed node output buffers after execution; not a GPU kernel profiler. The callback can reduce fusion and throughput.";
    report["status"] = "PASS";
    llama_perf_context_print(ctx.get());
}

int main(int argc, char ** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    options opts;
    try {
        opts = parse_options(argc, argv);
        opts.model = std::filesystem::absolute(opts.model).string();
        opts.output_prefix = std::filesystem::absolute(opts.output_prefix).string();
        std::filesystem::create_directories(std::filesystem::path(opts.output_prefix).parent_path());
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        print_usage(argv[0]);
        return 2;
    }
    observations state;
    state.log.open(opts.output_prefix + ".log", std::ios::trunc);
    if (!state.log) {
        std::cerr << "cannot open native log\n";
        return 2;
    }
    llama_log_set(log_callback, &state);
    const uint16_t endian = 1;
    json report = {
        {"schema", "spin8-native-smoke-v1"}, {"status", "FAIL"},
        {"scope", "fixed-token native execution; no quality or speed claim"},
        {"model", opts.model}, {"logits_file", opts.output_prefix + ".f32"},
        {"log_file", opts.output_prefix + ".log"},
        {"logits_dtype", *reinterpret_cast<const uint8_t *>(&endian) == 1 ? "<f4" : ">f4"},
        {"prompt_tokens", opts.prompt}, {"decode_tokens", opts.decode},
        {"n_gpu_layers_requested", opts.ngl}, {"device_requested", opts.device},
        {"threads", opts.threads}, {"n_ctx_requested", opts.context},
        {"flash_attn_requested", static_cast<int>(opts.flash_attn)}, {"finite_logits", false},
        {"load_mode_requested", llama_load_mode_name(opts.load_mode)},
        {"no_alloc_requested", opts.no_alloc},
    };
    try {
        run(opts, state, report);
    } catch (const std::exception & error) {
        report["error"] = error.what();
        std::cerr << error.what() << '\n';
    }
    llama_backend_free();
    llama_log_set(nullptr, nullptr);
    std::ofstream metadata(opts.output_prefix + ".json", std::ios::trunc);
    metadata << report.dump(2) << '\n';
    metadata.close();
    if (!metadata) {
        std::cerr << "failed to write result metadata\n";
        return 2;
    }
    std::cout << report["status"].get<std::string>() << ": " << opts.output_prefix << ".json\n";
    return report["status"] == "PASS" ? 0 : 1;
}
