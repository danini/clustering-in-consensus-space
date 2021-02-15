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

#include "clustering.h"
#include "dbscan_clustering.h"
#include <vector>
#include <unordered_map>
#include <Eigen/Eigen> 

namespace clustering
{
	namespace density
	{
		template <typename _DataType,
			typename _Distance>
			class MedianShiftClustering : public DensityClustring<std::vector<_DataType>, _Distance>
		{
		protected:
			const _Distance distanceObject;
			std::unordered_map<std::pair<size_t, size_t>, double, PairHash> distanceMap;
			Eigen::MatrixXd hyperPlanes;

		public:
			MedianShiftClustering() :
				distanceObject(_Distance())
			{

			}

			static const char* getName()
			{
				return "median shift";
			}

			void run(
				const double& threshold_,
				const std::vector<_DataType>& data_,
				std::vector<std::vector<size_t>>& clusters_)
			{
				if (data_.size() == 0)
					return;

				const size_t kDimensionNumber = data_[0].size();
				const size_t kPointNumber = data_.size();

				hyperPlanes = Eigen::MatrixXd::Identity(kDimensionNumber, kDimensionNumber);
				bool running = true;
				size_t clusterSize = kPointNumber;
				std::vector<_DataType> points = data_;
				std::vector<_DataType> newPoints(kPointNumber);
				std::vector<size_t> neighbors;
				neighbors.reserve(kPointNumber);
				std::vector<std::vector<size_t>> indexChain(kPointNumber);

				do
				{
					// Calculate the pair-wise distances
					distanceMap.clear();
					distanceMap.reserve(kPointNumber * (kPointNumber - 1) / 2);

					calculateDistances(points,
						distanceMap);

					for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
					{
						const auto& currentPoint = points[pointIdx];

						// Select the neighbors of the current point
						neighbors.clear();
						selectNeighbors(points,
							pointIdx,
							distanceMap,
							threshold_,
							neighbors);

						for (const auto& neighborIdx : neighbors)
							indexChain[neighborIdx].emplace_back(pointIdx);

						if (neighbors.size() == 1)
							newPoints[pointIdx] = currentPoint;
						else if (neighbors.size() <= 3)
							mean(newPoints[pointIdx],
								points,
								neighbors);
						else
							tukeyMedian(newPoints[pointIdx],
								points,
								neighbors);
					}

					points.swap(newPoints);
					newPoints.clear();
					newPoints.resize(kPointNumber);

					std::vector<std::vector<size_t>> tmpClusters;

					cluster(
						data_,
						points,
						tmpClusters);

					if (clusterSize == tmpClusters.size())
					{
						clusters_ = tmpClusters;
						break;
					}
					clusterSize = tmpClusters.size();
				} while (running);
			}

			void cluster(
				const std::vector<_DataType>& data_,
				const std::vector<_DataType>& shiftedData_,
				std::vector<std::vector<size_t>>& clusters_)
			{
				constexpr double CLUSTER_EPSILON = 0.01;
				std::vector<_DataType> modes;
				modes.reserve(shiftedData_.size());
				clusters_.reserve(shiftedData_.size());

				for (int i = 0; i < shiftedData_.size(); i++)
				{
					int c = 0;
					for (; c < clusters_.size(); c++) {
						double distance = distanceObject.distance(shiftedData_[i], modes[c]);
						if (distance <= CLUSTER_EPSILON) {
							break;
						}
					}

					if (c == clusters_.size()) {
						modes.emplace_back(shiftedData_[i]);
						clusters_.resize(clusters_.size() + 1);
					}

					clusters_[c].emplace_back(i);
				}

			}

			inline void tukeyMedian(
				_DataType& newPoint_,
				const std::vector<_DataType>& data_,
				const std::vector<size_t>& neighbors_)
			{
				const size_t kDimensionNumber = data_[0].size();
				const size_t kNeighborNumber = neighbors_.size();
				std::vector<size_t> depths(kNeighborNumber, 0);

				newPoint_.consensusVector.resize(kDimensionNumber, 0);

				// Iterate through the points
				for (const auto& neighborIdx : neighbors_)
				{
					const auto& point = data_[neighborIdx];

					// Iterate through the hyperplanes
					for (size_t planeIdx = 0; planeIdx < kDimensionNumber; planeIdx += kDimensionNumber / 100)
					{
						// Compute last hyperplane parameter
						double w = 0;
						for (size_t dimensionIdx = 0; dimensionIdx < kDimensionNumber; ++dimensionIdx)
							w = w - hyperPlanes(planeIdx, dimensionIdx) * point.consensusVector[dimensionIdx];

						// Count the points on each side
						int c1 = 0, 
							c2 = 0;
						std::vector<size_t> pts1, pts2;
						double distance;
						for (size_t k = 0; k < kNeighborNumber; ++k)
						{
							distance = 0;
							for (size_t dimensionIdx = 0; dimensionIdx < kDimensionNumber; ++dimensionIdx)
								distance += hyperPlanes(planeIdx, dimensionIdx) * data_[k].consensusVector[dimensionIdx];
							distance += w;

							if (distance < 0)
							{
								pts1.emplace_back(k);
								++c1;
							}
							else
							{
								pts2.emplace_back(k);
								++c2;
							}
						}

						if (c1 < c2)
							for (const auto& pointIdx : pts1)
								++depths[pointIdx];
						else
							for (const auto& pointIdx : pts2)
								++depths[pointIdx];
					}
				}

				size_t minDepth = std::numeric_limits<size_t>::max();
				size_t medianIdx;
				for (size_t k = 0; k < kNeighborNumber; ++k)
				{
					if (depths[k] < minDepth)
					{
						minDepth = depths[k];
						medianIdx = k;
					}
				}

				newPoint_ = data_[neighbors_[medianIdx]];
			}

			inline void mean(
				_DataType& newPoint_,
				const std::vector<_DataType>& data_,
				const std::vector<size_t>& neighbors)
			{
				const size_t kDimensionNumber = data_[0].size();

				newPoint_.consensusVector.resize(kDimensionNumber, 0);

				for (const auto& neighborIdx : neighbors)
					for (size_t dimension = 0; dimension < kDimensionNumber; ++dimension)
						newPoint_.consensusVector[dimension] += data_[neighborIdx].consensusVector[dimension];

				const size_t& neighborNumber = neighbors.size();
				for (size_t dimension = 0; dimension < kDimensionNumber; ++dimension)
					newPoint_.consensusVector[dimension] /= neighborNumber;
			}

			void selectNeighbors(
				const std::vector<_DataType>& data_,
				const size_t &pointIdx_,
				const std::unordered_map<std::pair<size_t, size_t>, double, PairHash>& distanceMap_,
				const double &threshold_,
				std::vector<size_t> &neighbors)
			{
				const size_t kPointNumber = data_.size();
				std::pair<size_t, size_t> pairIdx;

				for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
				{
					if (pointIdx == pointIdx_)
					{
						neighbors.emplace_back(pointIdx);
						continue;
					}

					pairIdx.first = MIN(pointIdx, pointIdx_);
					pairIdx.second = MAX(pointIdx, pointIdx_);

					const double& distance =
						distanceMap_.find(pairIdx)->second;

					if (distance <= threshold_)
						neighbors.emplace_back(pointIdx);
				}
			}

			void calculateDistances(const std::vector<_DataType>& data_,
				std::unordered_map<std::pair<size_t, size_t>, double, PairHash>& distanceMap_) const
			{
				const size_t& pointNumber = data_.size();

				for (size_t i = 0; i < pointNumber - 1; i++)
					for (size_t j = i + 1; j < pointNumber; j++)
						distanceMap_[std::make_pair(i, j)] =
							distanceObject.distance(data_[i], data_[j]);
			}
		};
	}
}