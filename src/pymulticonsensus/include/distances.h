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
#include <unordered_set>

namespace clustering
{
	namespace distances
	{
		template <class _DataType>
		class Distance 
		{
		public:
			virtual double distance(
				const _DataType& point1_,
				const _DataType& point2_) const = 0;
		};

		template <class _DataType,
			bool _LargeScaleData = false>
		class TanimotoDistance : public Distance<_DataType>
		{
		public:
			double distance(
				const _DataType& point1_,
				const _DataType& point2_) const
			{
				if constexpr (_LargeScaleData)
				{
					double dotProduct = 0,
						length1 = 0,
						length2 = 0;

					std::unordered_set<size_t> indices;
					indices.reserve(point1_.consensusVector.size());

					for (const auto& inlierIdx : point1_.inliers)
					{
						const double& a = point1_.consensusVector[inlierIdx],
							& b = point2_.consensusVector[inlierIdx];

						dotProduct += a * b;
						length1 += a * a;
						length2 += b * b;
						indices.emplace(inlierIdx);
					}

					for (const auto& inlierIdx : point2_.inliers)
					{
						if (indices.find(inlierIdx) != indices.end())
							continue;

						const double& a = point1_.consensusVector[inlierIdx],
							& b = point2_.consensusVector[inlierIdx];

						dotProduct += a * b;
						length1 += a * a;
						length2 += b * b;
					}

					double distance =
						dotProduct / (length1 + length2 - dotProduct);
				}
				else
				{
					double dotProduct = 0,
						length1 = 0,
						length2 = 0;

					for (size_t i = 0; i < point1_.consensusVector.size(); ++i)
					{
						const double& a = point1_.consensusVector[i],
							& b = point2_.consensusVector[i];

						dotProduct += a * b;
						length1 += a * a;
						length2 += b * b;
					}

					double distance =
						dotProduct / (length1 + length2 - dotProduct);

					return 1 - distance;
				}
			}
		};

		template <class _DataType,
			bool _LargeScaleData = false>
		class JaccardDistance : public Distance<_DataType>
		{
		public:
			double distance(
				const _DataType &point1_,
				const _DataType &point2_) const
			{
				int nzsum = 0,
					nesum = 0,
					nz, ne;

				for (size_t i = 0; i < point1_.consensusVector.size(); ++i)
				{
					nz = (point1_.consensusVector[i] != 0 | point2_.consensusVector[i] != 0);
					ne = (point1_.consensusVector[i] != point2_.consensusVector[i]);

					nzsum += nz;
					nesum += (nz & ne);
				}

				return static_cast<double>(nesum) / static_cast<double>(nzsum);
			}
		};
	}
}