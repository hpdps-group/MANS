#pragma once

#ifdef ENABLE_TIMING

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace mans {

struct TimingStats {
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::infinity();
    double max_ms = 0.0;
    std::uint64_t count = 0;
};

class TimingCollector {
public:
    static TimingCollector& instance() {
        static TimingCollector inst;
        return inst;
    }

    void begin_run() {
        runs_.emplace_back();
        current_ = &runs_.back().stats;
    }

    void end_run() {
        current_ = nullptr;
    }

    void add(const char* name, double ms) {
        if (!current_) {
            begin_run();
        }
        auto& stats = (*current_)[name];
        stats.total_ms += ms;
        stats.count += 1;
        stats.min_ms = std::min(stats.min_ms, ms);
        stats.max_ms = std::max(stats.max_ms, ms);
    }

    void dump_csv(const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            return;
        }
        out << std::fixed << std::setprecision(6);
        out << "run_index,"
               "total_ms,total_count,total_avg_ms,sum_total_ms,"
               "io_read_ms,io_read_count,io_read_avg_ms,"
               "io_write_ms,io_write_count,io_write_avg_ms,"
               "decide_adm_ms,decide_adm_count,decide_adm_avg_ms,"
               "adm_ms,adm_count,adm_avg_ms,"
               "ans_ms,ans_count,ans_avg_ms,"
               "alloc_ms,alloc_count,alloc_avg_ms,"
               "adm_alloc_ms,adm_alloc_count,adm_alloc_avg_ms,"
               "io_ratio,adm_ratio,ans_ratio,adm_ans_ratio,alloc_ratio,adm_alloc_ratio\n";

        for (std::size_t i = 0; i < runs_.size(); ++i) {
            const auto& stats = runs_[i].stats;
            const auto total = get_stats(stats, "total");
            const auto io_read = get_stats(stats, "io_read");
            const auto io_write = get_stats(stats, "io_write");
            const auto decide_adm = get_stats(stats, "decide_adm");
            const auto adm_comp = get_stats(stats, "adm_compress");
            const auto adm_decomp = get_stats(stats, "adm_decompress");
            const auto ans_comp = get_stats(stats, "ans_compress");
            const auto ans_decomp = get_stats(stats, "ans_decompress");
            const auto alloc = sum_prefix(stats, "alloc_");
            const auto adm_alloc = sum_prefix(stats, "adm_alloc_");

            const double sum_total_ms = sum_all_ms(stats);
            const double total_ms = total.total_ms > 0.0 ? total.total_ms : sum_total_ms;
            const double denom = total_ms > 0.0 ? total_ms : 1.0;

            const double io_ms = io_read.total_ms + io_write.total_ms;
            const std::uint64_t io_count = io_read.count + io_write.count;
            const double adm_ms = adm_comp.total_ms + adm_decomp.total_ms;
            const std::uint64_t adm_count = adm_comp.count + adm_decomp.count;
            const double ans_ms = ans_comp.total_ms + ans_decomp.total_ms;
            const std::uint64_t ans_count = ans_comp.count + ans_decomp.count;

            out << (i + 1) << ","
                << total_ms << "," << total.count << "," << avg_ms(total_ms, total.count) << ","
                << sum_total_ms << ","
                << io_read.total_ms << "," << io_read.count << "," << avg_ms(io_read.total_ms, io_read.count) << ","
                << io_write.total_ms << "," << io_write.count << "," << avg_ms(io_write.total_ms, io_write.count) << ","
                << decide_adm.total_ms << "," << decide_adm.count << "," << avg_ms(decide_adm.total_ms, decide_adm.count) << ","
                << adm_ms << "," << adm_count << "," << avg_ms(adm_ms, adm_count) << ","
                << ans_ms << "," << ans_count << "," << avg_ms(ans_ms, ans_count) << ","
                << alloc.total_ms << "," << alloc.count << "," << avg_ms(alloc.total_ms, alloc.count) << ","
                << adm_alloc.total_ms << "," << adm_alloc.count << "," << avg_ms(adm_alloc.total_ms, adm_alloc.count) << ","
                << (io_ms / denom) << ","
                << (adm_ms / denom) << ","
                << (ans_ms / denom) << ","
                << ((adm_ms + ans_ms) / denom) << ","
                << (alloc.total_ms / denom) << ","
                << (adm_alloc.total_ms / denom)
                << "\n";
        }
    }

private:
    struct RunData {
        std::unordered_map<std::string, TimingStats> stats;
    };

    TimingCollector() = default;

    static TimingStats get_stats(const std::unordered_map<std::string, TimingStats>& stats,
                                 const char* name) {
        auto it = stats.find(name);
        if (it == stats.end()) {
            return TimingStats{};
        }
        return it->second;
    }

    struct SumStats {
        double total_ms = 0.0;
        std::uint64_t count = 0;
    };

    static SumStats sum_prefix(const std::unordered_map<std::string, TimingStats>& stats,
                               const char* prefix) {
        const std::string prefix_str(prefix);
        SumStats total{};
        for (const auto& kv : stats) {
            if (kv.first.rfind(prefix_str, 0) == 0) {
                total.total_ms += kv.second.total_ms;
                total.count += kv.second.count;
            }
        }
        return total;
    }

    static double sum_all_ms(const std::unordered_map<std::string, TimingStats>& stats) {
        double total = 0.0;
        for (const auto& kv : stats) {
            total += kv.second.total_ms;
        }
        return total;
    }

    static double avg_ms(double total_ms, std::uint64_t count) {
        return count ? (total_ms / static_cast<double>(count)) : 0.0;
    }

    std::vector<RunData> runs_;
    std::unordered_map<std::string, TimingStats>* current_ = nullptr;
};

class ScopedTimer {
public:
    explicit ScopedTimer(const char* name)
        : name_(name),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration<double, std::milli>(end - start_).count();
        TimingCollector::instance().add(name_, ms);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* name_;
    std::chrono::steady_clock::time_point start_;
};

class RunScope {
public:
    RunScope() { TimingCollector::instance().begin_run(); }
    ~RunScope() { TimingCollector::instance().end_run(); }

    RunScope(const RunScope&) = delete;
    RunScope& operator=(const RunScope&) = delete;
};

} // namespace mans

#define MANS_TIMING_SCOPE(name) mans::ScopedTimer _mans_timer_##__LINE__(name)
#define MANS_TIMING_DUMP(path) mans::TimingCollector::instance().dump_csv(path)
#define MANS_TIMING_RUN_SCOPE() mans::RunScope _mans_run_##__LINE__{}

#else

#define MANS_TIMING_SCOPE(name) do {} while (0)
#define MANS_TIMING_DUMP(path) do {} while (0)
#define MANS_TIMING_RUN_SCOPE() do {} while (0)

#endif
