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
#include <vector>
#include <unordered_map>

namespace clustering
{
	namespace density
	{
		// Only for pairs of std::hash-able types for simplicity.
		// You can of course template this struct to allow other hash functions
		struct PairHash {
			static size_t clusterNumber;

			template <class T1, class T2>
			std::size_t operator () (const std::pair<T1, T2>& p) const {
				// Mainly for demonstration purposes, i.e. works but is overly simple
				// In the real world, use sth. like boost.hash_combine
				return clusterNumber * p.first + p.second;
			}
		};

		size_t PairHash::clusterNumber{ 1000 };

		template <typename _DataType,
			typename _Distance>
		class DBScanClustering : public DensityClustring<std::vector<_DataType>, _Distance>
		{
		protected:
			enum PointState { NOISE = -2, NOT_CLASSIFIED = -1 };

			const _Distance distanceObject;

			std::unordered_map<std::pair<size_t, size_t>, double, PairHash> distanceMap;
			std::vector<std::vector<size_t>> adjacentPoints;
			std::vector<size_t> pointNumbers;
			std::vector<int> pointToCluster;

			int minimumPointNumber;

		public:
			DBScanClustering(
				int minimumPointNumber_ = 1) :
				distanceObject(_Distance()),
				minimumPointNumber(minimumPointNumber_)
			{

			}

			static const char* getName()
			{
				return "dbscan";
			}

			void run(
				const double &threshold_,
				const std::vector<_DataType>& data_,
				std::vector<std::vector<size_t>>& clusters_)
			{
				const size_t& pointNumber = data_.size();
				if (pointNumber == 0)
					return;
				if (pointNumber == 1)
				{
					clusters_.emplace_back(std::vector<size_t>(1, 0));
					return;
				}

				int clusterIdx = -1;

				adjacentPoints.clear();
				pointToCluster.clear();
				pointNumbers.clear();

				adjacentPoints.resize(pointNumber);
				pointToCluster.resize(pointNumber, NOT_CLASSIFIED);
				pointNumbers.resize(pointNumber, 0);
				clusters_.reserve(pointNumber);

				//PairHash::clusterNumber = pointNumber;
				distanceMap.clear();
				distanceMap.reserve(pointNumber * (pointNumber - 1) / 2);

				calculateDistances(data_,
					distanceMap);

				checkNearPoints(data_,
					threshold_,
					pointNumbers,
					adjacentPoints,
					distanceMap);

				for (int i = 0; i < pointNumber; i++) {
					if (pointToCluster[i] != NOT_CLASSIFIED) continue;

					if (isCoreObject(i,
						pointNumbers)) {
						dfs(i, 
							++clusterIdx,
							pointNumbers,
							pointToCluster,
							adjacentPoints);
					}
					else {
						pointToCluster[i] = NOISE;
					}
				}

				clusters_.resize(clusterIdx + 1);
				for (int i = 0; i < pointNumber; i++) {
					if (pointToCluster[i] 
						!= NOISE) {
						clusters_[pointToCluster[i]].push_back(i);
					}
				}

				for (int i = 0; i < pointNumber; i++) {
					if (pointToCluster[i] == NOISE) {
						clusters_.resize(clusters_.size() + 1);
						clusters_.back().emplace_back(i);
					}
				}
			}

			void dfs(
				int now,
				int c,
				std::vector<size_t>& pointNumbers_,
				std::vector<int>& pointToCluster_,
				const std::vector<std::vector<size_t>>& adjacentPoints_) const
			{
				pointToCluster_[now] = c;
				if (!isCoreObject(now,
					pointNumbers_))
					return;

				for (auto& next : adjacentPoints_[now])
				{
					if (pointToCluster_[next] != NOT_CLASSIFIED)
						continue;
					dfs(next, 
						c,
						pointNumbers_,
						pointToCluster_,
						adjacentPoints_);
				}
			}

			void calculateDistances(const std::vector<_DataType>& data_,
				std::unordered_map<std::pair<size_t, size_t>, double, PairHash> &distanceMap_) const
			{
				const size_t& pointNumber = data_.size();

				for (size_t i = 0; i < pointNumber - 1; i++)
					for (size_t j = i + 1; j < pointNumber; j++)
						distanceMap_[std::make_pair(i, j)] =
							distanceObject.distance(data_[i], data_[j]);
			}

			void checkNearPoints(const std::vector<_DataType>& data_,
				const double& threshold_,
				std::vector<size_t>& pointNumbers_,
				std::vector<std::vector<size_t>>& adjacentPoints_,
				const std::unordered_map<std::pair<size_t, size_t>, double, PairHash>& distanceMap_) const
			{
				const size_t &pointNumber = data_.size();
				std::pair<size_t, size_t> index;

				for (int i = 0; i < pointNumber; i++)
				{
					for (int j = 0; j < pointNumber; j++)
					{
						if (i == j) 
							continue;

						if (i < j)
						{
							index.first = i;
							index.second = j;
						} else
						{
							index.first = j;
							index.second = i;
						}

						const double &distance = 
							distanceMap_.find(index)->second;

						if (distance <= threshold_) {
							++pointNumbers_[i];
							adjacentPoints_[i].push_back(j);
						}
					}
				}
			}

			bool isCoreObject(int idx,
				const std::vector<size_t>& pointNumbers_) const
			{
				return pointNumbers_[idx] >= minimumPointNumber;
			}
		};
	}
}