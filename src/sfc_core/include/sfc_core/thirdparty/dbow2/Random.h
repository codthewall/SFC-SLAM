/**
 * File: Random.h
 * Project: DUtils library
 * Author: Dorian Galvez-Lopez
 * Description: manages pseudo-random numbers
 * License: see LICENSE.txt
 * Modified: simplified for ORB vocabulary loading only
 */

#ifndef __D_RANDOM__
#define __D_RANDOM__

#include <cstdlib>
#include <ctime>

namespace DUtils {

class Random
{
public:
    static void SeedRand() { std::srand((unsigned)std::time(nullptr)); }
    static void SeedRandOnce() { static bool seeded = false; if(!seeded) { SeedRand(); seeded = true; } }
    static void SeedRand(int seed) { std::srand((unsigned)seed); }
    static void SeedRandOnce(int seed) { static bool seeded = false; if(!seeded) { SeedRand(seed); seeded = true; } }

    static int RandomInt(int min_val, int max_val) {
        return min_val + std::rand() % (max_val - min_val + 1);
    }

    static double RandomValue(double min_val, double max_val) {
        return min_val + (max_val - min_val) * (std::rand() / (double)RAND_MAX);
    }
};

} // namespace DUtils

#endif
