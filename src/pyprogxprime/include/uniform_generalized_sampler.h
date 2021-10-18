// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include "uniform_random_generator.h"
#include "sampler.h"

namespace gcransac
{
	namespace sampler
	{
		class UniformGeneralizedSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			std::unique_ptr<utils::UniformRandomGenerator<size_t>> blockSelector;
			std::vector<std::unique_ptr<utils::UniformRandomGenerator<size_t>>> random_generators;
			const std::vector<std::pair<size_t, size_t>> blockBorders;
			const std::vector<size_t> sampleSplit;
			const size_t blockNumber;
			const bool orderedSplits;

		public:
			explicit UniformGeneralizedSampler(
				const cv::Mat * const container_,
				const std::vector<std::pair<size_t, size_t>> &blockBorders_,
				const std::vector<size_t> &sampleSplit_,
				const bool &orderedSplits_ = false) :
					Sampler(container_),
					blockNumber(blockBorders_.size()),
					sampleSplit(sampleSplit_),
					orderedSplits(orderedSplits_),
					blockBorders(blockBorders_)
			{
				//if (sampleSplit.size() != blockNumber)
				//	fprintf(stderr, "Error when initializing the generalized uniform sampler. The split and block numbers are different.\n");

				initialized = initialize(container_);
			}

			~UniformGeneralizedSampler() {}

			const std::string getName() const { return "Uniform Sampler"; }

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat * const container_)
			{
				random_generators.reserve(blockNumber);
				for (const auto &block : blockBorders)
				{
					random_generators.emplace_back(std::make_unique<utils::UniformRandomGenerator<size_t>>());
					random_generators.back()->resetGenerator(
						block.first,
						block.second);
				}

				if (!orderedSplits)
				{
					blockSelector = std::make_unique<utils::UniformRandomGenerator<size_t>>();
					blockSelector->resetGenerator(
						0,
						blockNumber - 1);
				}

				return true;
			}

			void reset()
			{
				for (size_t blockIdx = 0; blockIdx < blockNumber; ++blockIdx)
				for (const auto &block : blockBorders)
				{
					random_generators[blockIdx]->resetGenerator(
						blockBorders[blockIdx].first,
						blockBorders[blockIdx].second);
				}
			}

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE bool sample(const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_);
		};

		OLGA_INLINE bool UniformGeneralizedSampler::sample(
			const std::vector<size_t> &pool_,
			size_t * const subset_,
			size_t sample_size_)
		{
			// If there are not enough points in the pool, interrupt the procedure.
			if (sample_size_ > pool_.size())
				return false;

			/*size_t currentBlock = 0;
			if (!orderedSplits)
				// Generate a random starting block
				currentBlock = blockSelector->getRandomNumber();*/

			size_t sampleIdx = 0;
			int currentBlock = 0,
				prevBlock = -1;

			for (size_t splitIdx = 0; splitIdx < sampleSplit.size(); ++splitIdx)
			{
				const size_t& pointNumber = sampleSplit[splitIdx];

				do
				{
					currentBlock = blockSelector->getRandomNumber();
				} while (currentBlock == prevBlock);
				prevBlock = currentBlock;

				for (size_t localSampleIdx = 0; localSampleIdx < pointNumber; ++localSampleIdx)
				{
					subset_[sampleIdx++] = pool_[random_generators[currentBlock]->getRandomNumber()];
				}
			}
	
			/*for (size_t blockIdx = 0; blockIdx < blockNumber; ++blockIdx)
			{
				for (size_t localSampleIdx = 0; localSampleIdx < sampleSplit[blockIdx]; ++localSampleIdx)
					subset_[sampleIdx++] = pool_[random_generators[currentBlock]->getRandomNumber()];

				currentBlock = (currentBlock + 1) % blockNumber;
			}*/
			return true;
		}
	}
}