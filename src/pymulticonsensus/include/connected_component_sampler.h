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

#include <queue>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

#include "uniform_random_generator.h"
#include "neighborhood/grid_neighborhood_graph.h"
#include "samplers/sampler.h"

#include <iostream>

namespace gcransac
{
	namespace sampler
	{
		class ConnectedComponentSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			const double
				maximumDistance,
				minimumDistance,
				stepSize;
			const size_t
				stepNumber,
				minimumStructureSize,
				pointNumber;
				
			size_t currentIterations,
				maxIterations,
				iterationsPerStructure;
			double currentDistance;

			std::vector<size_t> growthFunction; // The growth function.

			std::unique_ptr<utils::UniformRandomGenerator<size_t>> randomGenerator; // The random number generator
			std::unique_ptr<utils::UniformRandomGenerator<size_t>> structureRandomGenerator; // The random number generator

			std::vector<std::vector<size_t>> currentStructures;
			std::vector<std::pair<size_t, size_t>> largeStructureIndices;
			size_t currentStructureIdx;

			std::vector<std::vector<std::pair<double, size_t>>> neighborhoodGraph;
			bool usingSpatialModel;

		public:
			explicit ConnectedComponentSampler(
				const cv::Mat * const container_,
				const size_t& minimumStructureSize_,
				const double minimumDistance_ = 20.0,
				const double maximumDistance_ = 500.0,
				const size_t stepNumber_ = 10)
				: Sampler(container_),
				minimumDistance(minimumDistance_),
				currentDistance(minimumDistance_),
				stepSize((maximumDistance_ - minimumDistance_) / stepNumber_),
				stepNumber(stepNumber_),
				pointNumber(container_->rows),
				minimumStructureSize(minimumStructureSize_),
				maximumDistance(maximumDistance_),
				usingSpatialModel(true),
				iterationsPerStructure(10),
				currentIterations(0)
			{
				initialized = initialize(container_);
			}

			~ConnectedComponentSampler() {}
			
			void update(
				const size_t* const subset_,
				const size_t& sample_size_,
				const size_t& iteration_number_,
				const double& inlier_ratio_) 
			{
				
			}

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat * const container_)
			{
				if (pointNumber != container_->rows)
				{
					fprintf(stderr, "The point number differs in the initialization (%d) from what it is used in the constructor (%d).\n",
						static_cast<int>(container_->rows),
						static_cast<int>(pointNumber));
					return false;
				}

				if (pointNumber < minimumStructureSize)
					return false;

				// Find the widest neighborhood structure
				std::vector< std::vector< cv::DMatch >> matches;

				cv::Ptr<cv::BFMatcher> matcher =
					cv::BFMatcher::create();

				if (container_->depth() == CV_64F)
				{
					cv::Mat tmpContainer;
					container_->convertTo(tmpContainer, CV_32F);

					matcher->radiusMatch(tmpContainer,
						tmpContainer,
						matches,
						maximumDistance);
				}
				else
					matcher->radiusMatch(*container_,
						*container_,
						matches,
						maximumDistance);

				// Build neighborhood-graph structure
				neighborhoodGraph.resize(pointNumber);
				for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
				{
					// Occupying the required memory early. The size is neighbor number minus one since
					// the point is always amongst its own neighbors.
					neighborhoodGraph[pointIdx].reserve(matches[pointIdx].size() - 1);

					// Adding the neighbor and its distance to the structure.
					// In OpenCV, they are already ordered according to their distances.
					for (const auto& match : matches[pointIdx])
						neighborhoodGraph[pointIdx].emplace_back(
							std::make_pair(match.distance, match.trainIdx));
				}

				// Initialize the random generator as well which will be used if there are no 
				// connected components left.
				randomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
				randomGenerator->resetGenerator(0,
					static_cast<size_t>(pointNumber) - 1);
				structureRandomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();

				// Inititalize the CC growth function
				maxIterations = static_cast<size_t>(20 * pointNumber);

				growthFunction.resize(pointNumber);
				double T_n = maxIterations;
				for (size_t i = 0; i < minimumStructureSize; ++i) {
					T_n *= static_cast<double>(minimumStructureSize - i) /
						(pointNumber - i);
				}

				unsigned int T_n_prime = 1;
				for (size_t i = 0; i < pointNumber; ++i) {
					if (i + 1 <= minimumStructureSize) {
						growthFunction[i] = T_n_prime;
						continue;
					}
					double Tn_plus1 = static_cast<double>(i + 1) * T_n /
						(i + 1 - minimumStructureSize);
					growthFunction[i] = T_n_prime + static_cast<size_t>(ceil(Tn_plus1 - T_n));
					T_n = Tn_plus1;
					T_n_prime = growthFunction[i];
				}
				
				findConnectedComponents(currentDistance);
				return true;
			}

			void findConnectedComponents(
				const double& currentMaximumDistance_)
			{
				if (neighborhoodGraph.size() == 0)
					return;

				// Clear the previous structures
				currentStructures.clear();
				largeStructureIndices.clear();
				currentStructureIdx = 0;

				// The number of connected components in the data
				size_t structureNumber = 0;
				// The label of each point assigning the point to a connected component
				std::vector<size_t> structureIndices(pointNumber, 0);
				// The mask determining if a point has been already visited
				std::vector<bool> visitedMask(pointNumber, false);

				// Find an unvisited point
				for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
				{
					if (visitedMask[pointIdx])
						continue;

					++structureNumber;
					currentStructures.resize(structureNumber);
					auto& currentStructure = currentStructures.back();
					currentStructure.reserve(pointNumber);

					std::queue<size_t> processQueue;
					processQueue.push(pointIdx);

					while (!processQueue.empty())
					{
						const size_t& idx = processQueue.front();
						if (visitedMask[idx])
						{
							processQueue.pop();
							continue;
						}

						structureIndices[idx] = structureNumber;
						visitedMask[idx] = true;
						currentStructure.emplace_back(idx);
						for (const auto& neighbor : neighborhoodGraph[idx])
						{
							// If the point is farther than the current maximum distance, do not add it
							// as a neighbor.
							if (neighbor.first >= currentMaximumDistance_)
								break;

							const size_t& neighborIdx =
								neighbor.second;

							if (visitedMask[neighborIdx])
								continue;
							processQueue.push(neighborIdx);
						}
						processQueue.pop();
					}

					const size_t& structureSize = currentStructure.size();
					if (structureSize >= minimumStructureSize)
						largeStructureIndices.emplace_back(
							std::make_pair(structureSize, structureNumber - 1));
				}

				// Sort the connected components according to their sizes
				std::sort(std::rbegin(largeStructureIndices), std::rend(largeStructureIndices));	

				if (largeStructureIndices.size() > 0)
				{
					structureRandomGenerator->resetGenerator(0, largeStructureIndices[0].first - 1);
					iterationsPerStructure = growthFunction[largeStructureIndices[0].first];
				}
			}

			void reset() {}

			bool isUsingSpatialModel() const { return usingSpatialModel; }

			const std::string getName() const { return "Connected Component Sampler"; }

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE bool sample(
				const std::vector<size_t>& pool_,
				size_t* const subset_,
				size_t sample_size_);

			OLGA_INLINE bool sample(
				std::vector<size_t>& subset_,
				size_t sample_size_);
		};

		OLGA_INLINE bool ConnectedComponentSampler::sample(
			const std::vector<size_t>& pool_,
			size_t* const subset_,
			size_t sampleSize_)
		{
			// If there are no more structure in the current scale, go for a bigger one.
			if (currentStructureIdx >= largeStructureIndices.size())
				currentDistance += stepSize;

			// If we exceeded the maximum size and still need sampling,
			// let's go with uniform sampling.
			if (currentDistance >= maximumDistance ||
				neighborhoodGraph.size() == 0)
			{
				randomGenerator->generateUniqueRandomSet(
					subset_,
					sampleSize_);
				usingSpatialModel = true;
				return true;
			}

			// If there are no more structure in the current scale, go for a bigger one.
			if (currentStructureIdx >= largeStructureIndices.size())
				findConnectedComponents(currentDistance);

			if (currentStructureIdx >= largeStructureIndices.size())
			{
				randomGenerator->generateUniqueRandomSet(
					subset_,
					sampleSize_);
				usingSpatialModel = true;
				return true;
			}

			const size_t& structureIdx =
				largeStructureIndices[currentStructureIdx++].second;
			const auto& structure =
				currentStructures[structureIdx];

			for (size_t pointIdx = 0; pointIdx < sampleSize_; ++pointIdx)
				subset_[pointIdx] = structure[pointIdx];

			std::cout << "Error" << std::endl;
			return true;
		}

		OLGA_INLINE bool ConnectedComponentSampler::sample(
			std::vector<size_t>& subset_,
			size_t sampleSize_)
		{
			// If there are no more structure in the current scale, go for a bigger one.
			if (currentStructureIdx >= largeStructureIndices.size())
				currentDistance += stepSize;

			// If we exceeded the maximum size and still need sampling,
			// let's go with uniform sampling.
			if (currentDistance >= maximumDistance)
			{
				subset_.resize(sampleSize_);
				randomGenerator->generateUniqueRandomSet(
					&(subset_[0]),
					sampleSize_);
				usingSpatialModel = true;
				return true;
			}

			// If there are no more structure in the current scale, go for a bigger one.
			if (currentStructureIdx >= largeStructureIndices.size())
				findConnectedComponents(currentDistance);

			if (currentIterations++ >= iterationsPerStructure)
			{
				currentIterations = 0;
				++currentStructureIdx;
				if (currentStructureIdx < largeStructureIndices.size())
				{
					iterationsPerStructure = growthFunction[largeStructureIndices[currentStructureIdx].first];
					structureRandomGenerator->resetGenerator(0,
						largeStructureIndices[currentStructureIdx].first - 1);
				}
			}

			if (currentStructureIdx >= largeStructureIndices.size())
			{
				subset_.resize(sampleSize_);
				randomGenerator->generateUniqueRandomSet(
					&(subset_[0]),
					sampleSize_);
				usingSpatialModel = true;
				return true;
			}

			const size_t& structureIdx =
				largeStructureIndices[currentStructureIdx].second;
			const auto& structure =
				currentStructures[structureIdx];

			subset_.resize(sampleSize_);
			structureRandomGenerator->generateUniqueRandomSet(
				&(subset_[0]),
				sampleSize_);
			for (size_t pointIdx = 0; pointIdx < sampleSize_; ++pointIdx)
				subset_[pointIdx] = structure[subset_[pointIdx]];

			// subset_ = structure;
			return true;
		}
	}
}
