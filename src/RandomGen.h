/*
 * RandomGen.h
 *
 *  Created on: Feb 8, 2013
 *      Author: Christoph
 */

#ifndef RANDOMGEN_H_
#define RANDOMGEN_H_

#include <sys/time.h>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

struct RandomGen
{
	RandomGen() : rng(42){};
	RandomGen(unsigned int lseed): seed(lseed) {
		rng.seed(seed);
	}

	void setSeed(unsigned int lseed) {
		seed = lseed;
		rng.seed(seed);
	}

	int getRandomNumber(int maxExcluded) {
		boost::uniform_int<> intdist(0, maxExcluded-1);
		boost::variate_generator< boost::mt19937&, boost::uniform_int<> >
		    GetRand(rng, intdist);
		return GetRand();
	}


    int operator()(int maxExcluded)
    {
        return boost::uniform_int<int>(0, maxExcluded-1)(rng);
    }
unsigned int seed;
boost::mt19937 rng;
};



#endif /* RANDOMGEN_H_ */
