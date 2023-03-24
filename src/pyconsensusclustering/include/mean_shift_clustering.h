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
		template <typename _DataType,
			typename _Distance>
		class MeanShiftClustering : public DensityClustring<std::vector<_DataType>, _Distance>
		{
		protected:
			const _Distance distanceObject;
			cv::RNG randomGenerator;

		public:
			MeanShiftClustering() :
				distanceObject(_Distance())
			{

			}

			static const char* getName()
			{
				return "mean shift";
			}

			void shiftPoint(
				const _DataType& point,
				const std::vector<_DataType>& points,
				double kernel_bandwidth,
				_DataType& shifted_point) 
			{
				shifted_point.consensusVector.resize(point.size(), 0);

				double total_weight = 0;
				for (int i = 0; i < points.size(); i++) {
					const _DataType& temp_point = points[i];
					double distance = distanceObject.distance(point, temp_point);
					double weight = 0;
					if (distance <= kernel_bandwidth)
						weight = 1.0 - distance;
					for (int j = 0; j < shifted_point.size(); j++) 
						shifted_point.consensusVector[j] += temp_point.consensusVector[j] * weight;
					total_weight += weight;
				}

				const double total_weight_inv = 
					1.0 / total_weight;
				for (int i = 0; i < shifted_point.size(); i++) {
					shifted_point.consensusVector[i] *= total_weight_inv;
				}
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

			void meanShift(
				const double& threshold_,
				const std::vector<_DataType>& data_,
				std::vector<_DataType> &shiftedPoints_)
			{
				constexpr double EPSILON = 1e-5;
				const size_t kDimensionNumber = data_[0].size();
				const size_t kPointNumber = data_.size();
				size_t clusterNumber = 0;
				std::vector<bool> stoppedMoving(kPointNumber, false);
				shiftedPoints_ = data_;
				double maximumShiftDistance;
				_DataType newPoint;
				newPoint.consensusVector.reserve(kDimensionNumber);
				do {
					maximumShiftDistance = 0;
					for (int i = 0; i < kPointNumber; i++) {
						newPoint.consensusVector.clear();
						if (!stoppedMoving[i]) {
							shiftPoint(shiftedPoints_[i], data_, threshold_, newPoint);
							double shiftDistance = distanceObject.distance(newPoint, shiftedPoints_[i]);
							if (shiftDistance > maximumShiftDistance) {
								maximumShiftDistance = shiftDistance;
							}
							if (shiftDistance <= EPSILON) {
								stoppedMoving[i] = true;
							}
							shiftedPoints_[i] = newPoint;
						}
					}
				} while (maximumShiftDistance > EPSILON);
			}

			void run(
				const double &threshold_,
				const std::vector<_DataType>& data_,
				std::vector<std::vector<size_t>>& clusters_)
			{
				if (data_.size() == 0)
					return;

				std::vector<_DataType> shiftedPoints;
				meanShift(threshold_, data_, shiftedPoints);
				cluster(data_, shiftedPoints, clusters_);
			}		
		};
	}
}