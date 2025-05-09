// =============================================================
// Copyright(C) 2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#ifndef CVOI_COMMON_STIMER_HPP_
#define CVOI_COMMON_STIMER_HPP_

#include <chrono>

/**
 * @class CvoiTimer
 * @brief A base class, which is used to record some code's exection duration.
 *
 * A typical use:
 *   CvoiTimer atimer;
 *   func_or_code_to_test();
 *   const auto duration = atimer.Elapsed();
 *   another_func_or_code_to_test();
 *   const auto duration2 = atimer.Elapsed();
 * NOTE: duration2 covers the elapsed duration of both func_or_code_to_test()
 * and another_func_or_code_to_test();
 */

class CvoiTimer
{
public:
    CvoiTimer() : start_(std::chrono::steady_clock::now()) {}

    /**
     * @brief Compute the elapsed time since the timer is started.
     *
     * @return The elapsed time in microseconds.
     */
    uint64_t Elapsed()
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<Duration>(now - start_).count();
    }

    /**
     * @brief print elapsed time (in milliseconds) with the givein print info.
     *
     * @param[in] print_info A string to be printed with the elapsed time.
     * @param[in] individer The iterations in which the duration is elapsed.
     */
    void printElapsed(const std::string& print_info, int individer = 1)
    {
        slog::verbose << print_info << ": " << (double)Elapsed() / (individer * 1000) << "ms."
                      << slog::endl;
    }

private:
    using Duration = std::chrono::duration<uint64_t, std::micro>;
    std::chrono::steady_clock::time_point start_;
};

#endif // CVOI_COMMON_STIMER_HPP_
