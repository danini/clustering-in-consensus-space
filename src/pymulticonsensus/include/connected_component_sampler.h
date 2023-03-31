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
#include "samplers/prosac_sampler.h"
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

			double currentDistance;

			std::unique_ptr<utils::UniformRandomGenerator<size_t>> randomGenerator; // The random number generator

			std::vector<std::vector<size_t>> currentStructures;
			std::vector<std::pair<size_t, size_t>> largeStructureIndices;
			size_t currentStructureIdx;

			std::vector<std::vector<std::pair<double, size_t>>> neighborhoodGraph;
			bool usingSpatialModel;

		public:
			explicit ConnectedComponentSampler(
				const cv::Mat* container_, // The pointer pointing to the data points
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
				usingSpatialModel(true)
			{
				initialized = initialize(container_);
			}

			~ConnectedComponentSampler() {}

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat* container_);

			void findConnectedComponents(
				const double& currentMaximumDistance_);

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

		bool ConnectedComponentSampler::initialize(
			const cv::Mat* container_)
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
				static_cast<size_t>(pointNumber));

			findConnectedComponents(currentDistance);

			return true;
		}

		void ConnectedComponentSampler::findConnectedComponents(
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
		}

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
				largeStructureIndices[currentStructureIdx++].second;
			const auto& structure =
				currentStructures[structureIdx];

			subset_ = structure;
			return true;
		}

		template <size_t _DimensionNumber>
		class GridBasedConnectedComponentSampler : public Sampler < cv::Mat, size_t >
		{
		protected:
			const size_t
				layerNumber,
				sampleSize,
				minimumStructureSize,
				pointNumber;

			double currentDistance;

			std::unique_ptr<utils::UniformRandomGenerator<size_t>> randomGenerator; // The random number generator

			std::vector<neighborhood::GridNeighborhoodGraph<_DimensionNumber>> gridLayers; // The overlapping neighborhood grids
			std::vector<size_t> layerData; // The sizes of the grids

			std::vector<double> sizes;
			std::vector<const std::vector<size_t> *> currentStructures;
			std::vector<std::pair<size_t, size_t>> largeStructureIndices;
			size_t currentStructureIdx,
				currentLayer;

			std::vector<std::vector<std::pair<double, size_t>>> neighborhoodGraph;

		public:
			explicit GridBasedConnectedComponentSampler(
				const cv::Mat* container_, // The pointer pointing to the data points
				const size_t& minimumStructureSize_,
				const std::vector<size_t> layerData_, // The number of cells for each neighborhood grid. This must be in descending order.
				const size_t sampleSize_, // The size of a minimal sample.
				const std::vector<double>& sizes_) // The width of the source image
				: Sampler(container_),
				pointNumber(container_->rows),
				currentLayer(0),
				sizes(sizes_),
				minimumStructureSize(minimumStructureSize_),
				layerData(layerData_),
				sampleSize(sampleSize_),
				layerNumber(layerData_.size())
			{
				initialized = initialize(container_);
			}

			~GridBasedConnectedComponentSampler() {}

			// Initializes any non-trivial variables and sets up sampler if
			// necessary. Must be called before sample is called.
			bool initialize(const cv::Mat* container_);

			void findConnectedComponents();

			void reset() {} 

			const std::string getName() const { return "Connected Component Sampler"; }

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			OLGA_INLINE bool sample(const std::vector<size_t>& pool_,
				size_t* const subset_,
				size_t sample_size_);

			OLGA_INLINE bool sample(
				std::vector<size_t>& subset_,
				size_t sample_size_);
		};

		template <size_t _DimensionNumber>
		bool GridBasedConnectedComponentSampler<_DimensionNumber>::initialize(
			const cv::Mat* container_)
		{
			// Initialize the random generator
			randomGenerator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
			randomGenerator->resetGenerator(0,
				static_cast<size_t>(pointNumber));

			// Initialize the grid layers. We do not need to add the last layer
			// since it contains the whole image.
			gridLayers.reserve(layerNumber);

			// Check layer data
			for (size_t layerIdx = 0; layerIdx < layerNumber; ++layerIdx)
			{
				if (layerIdx > 0 &&
					layerData[layerIdx - 1] <= layerData[layerIdx])
				{
					fprintf(stderr, "Error when initializing the Progressive NAPSAC sampler. The layers must be in descending order. The current order is \"");
					for (size_t layerIdx = 0; layerIdx < layerNumber - 1; ++layerIdx)
						fprintf(stderr, "%d ", layerData[layerIdx]);
					fprintf(stderr, "%d\"\n", layerData.back());
					return false;
				}

				const size_t cellNumberInGrid = layerData[layerIdx];

				std::vector<double> cellSizes(_DimensionNumber);
				for (size_t dimensionIdx = 0; dimensionIdx < _DimensionNumber; ++dimensionIdx)
					cellSizes[dimensionIdx] = sizes[dimensionIdx] / cellNumberInGrid;

				gridLayers.emplace_back(neighborhood::GridNeighborhoodGraph<_DimensionNumber>(container_,
					cellSizes,
					cellNumberInGrid));
			}

			findConnectedComponents();

			return true;
		}

		template <size_t _DimensionNumber>
		void GridBasedConnectedComponentSampler<_DimensionNumber>::findConnectedComponents()
		{
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

			for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
			{
				const auto &cell = gridLayers[currentLayer].getNeighbors(pointIdx);

				if (cell.size() < minimumStructureSize)
					continue;

				currentStructures.emplace_back(&cell);
				largeStructureIndices.emplace_back(
					std::make_pair(cell.size(), structureNumber));
				++structureNumber;
			}

			// Sort the connected components according to their sizes
			std::sort(std::rbegin(largeStructureIndices), std::rend(largeStructureIndices));
		}

		template <size_t _DimensionNumber>
		OLGA_INLINE bool GridBasedConnectedComponentSampler<_DimensionNumber>::sample(
			const std::vector<size_t>& pool_,
			size_t* const subset_,
			size_t sampleSize_)
		{
			// If there are no more structure in the current scale, go for a bigger one.
			if (currentStructureIdx >= largeStructureIndices.size())
				++currentLayer;

			// If we exceeded the maximum size and still need sampling,
			// let's go with uniform sampling.
			if (currentLayer >= layerData.size())
			{
				randomGenerator->generateUniqueRandomSet(
					subset_,
					sampleSize_,
					pointNumber);
				return true;
			}

			// If there are no more structure in the current scale, go for a bigger one.
			if (currentStructureIdx >= largeStructureIndices.size())
			{
				do
				{
					findConnectedComponents();

					// If there are no more structure in the current scale, go for a bigger one.
					if (currentStructureIdx >= largeStructureIndices.size())
						++currentLayer;
				} while (currentStructureIdx >= largeStructureIndices.size() &&
					currentLayer < layerData.size());

				if (currentLayer >= layerData.size())
				{
					randomGenerator->generateUniqueRandomSet(
						subset_,
						sampleSize_,
						pointNumber);
					return true;
				}
			}

			const size_t& structureIdx =
				largeStructureIndices[currentStructureIdx++].second;
			const auto& structure =
				currentStructures[structureIdx];

			for (size_t pointIdx = 0; pointIdx < sampleSize_; ++pointIdx)
				subset_[pointIdx] = structure->at(pointIdx);
			return true;
		}

		template <size_t _DimensionNumber>
		OLGA_INLINE bool GridBasedConnectedComponentSampler<_DimensionNumber>::sample(
			std::vector<size_t>& subset_,
			size_t sampleSize_)
		{
			// If there are no more structure in the current scale, go for a bigger one.
			if (currentStructureIdx >= largeStructureIndices.size())
				++currentLayer;

			// If we exceeded the maximum size and still need sampling,
			// let's go with uniform sampling.
			if (currentLayer >= layerData.size())
			{
				subset_.resize(sampleSize_);
				randomGenerator->generateUniqueRandomSet(
					&(subset_[0]),
					sampleSize_);
				return true;
			}

			// If there are no more structure in the current scale, go for a bigger one.
			if (currentStructureIdx >= largeStructureIndices.size())
				findConnectedComponents();

			const size_t& structureIdx =
				largeStructureIndices[currentStructureIdx++].second;
			const auto& structure =
				currentStructures[structureIdx];

			subset_ = structure;
			return true;
		}
	}
}
