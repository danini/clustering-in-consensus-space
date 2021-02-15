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
#include <Eigen/Eigen> 

namespace clustering
{
	namespace density
	{
		template <typename _DataType,
			typename _Distance>
			class kCliqueClustering : public DensityClustring<std::vector<_DataType>, _Distance>
		{
		protected:
			const _Distance distanceObject;
			std::unordered_map<std::pair<size_t, size_t>, double, PairHash> distanceMap;
			Eigen::MatrixXd hyperPlanes;

		public:
			kCliqueClustering() :
				distanceObject(_Distance())
			{

			}

			void transferNodes(
				const size_t &pointIdx_,
				const std::vector<size_t> &neighbors_,
				const size_t &cliqueSize_,
				const std::vector<std::vector<size_t>>& allNeighbors_,
				std::vector<size_t>& C_,
				std::vector<std::vector<size_t>>& clusters_)
			{
				bool found_s12 = false;
				bool found_s1 = false;

				for (size_t c = 0; c < clusters_.size(); ++c)
				{
					for (size_t cc = 0; cc < clusters_[c].size(); ++cc)
					{
						if all(ismember(S1, cliques{ c }(cc, :)))
							found_s1 = true;
						end
							if all(ismember(union(S1, S2), cliques{ c }(cc, :)))
								found_s12 = true;
						break;
						end
					}
				}

				/*transfer_nodes(S1, S2, clique_size, C)
					% Recursive function to transfer nodes from set B to set A(as
						% defined above)

					% Check if the union of S1and S2 or S1 is inside an already found larger
					% clique
				for c = 1:length(cliques)
					for cc = 1 : size(cliques{ c }, 1)
						if all(ismember(S1, cliques{ c }(cc, :)))
							found_s1 = true;
				end
					if all(ismember(union(S1, S2), cliques{ c }(cc, :)))
						found_s12 = true;
				break;
				end
					end
					end

					if found_s12 || (length(S1) ~= clique_size && isempty(S2))
						% If the union of the sets A and B can be included in an
						% already found(larger) clique, the recursion is stepped back
						% to check other possibilities
						R = [];
				elseif length(S1) == clique_size;
				% The size of A reaches s, a new clique is found
					if found_s1
						R = [];
					else
						R = S1;
				end
					else
						% Check the remaining possible combinations of the neighbors
						% indices
						if isempty(find(S2 >= max(S1), 1))
							R = [];
						else
							R = [];
				for w = find(S2 >= max(S1), 1) :length(S2)
					S2_aux = S2;
				S1_aux = S1;
				S1_aux = [S1_aux S2_aux(w)];
				S2_aux = setdiff(S2_aux(C(S2(w), S2_aux) == 1), S2_aux(w));
				R = [R; transfer_nodes(S1_aux, S2_aux, clique_size, C)];
				end
					end
					end
					end*/
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

				calculateDistances(
					data_,
					distanceMap);

				std::vector<std::vector<size_t>> neighbors(kPointNumber);
				std::vector<std::pair<size_t, size_t>> neighborNumbers(kPointNumber);

				for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
				{
					auto& currentNeighbors = neighbors[pointIdx];
					currentNeighbors.reserve(kPointNumber);
					selectNeighbors(points,
						pointIdx,
						distanceMap,
						threshold_,
						currentNeighbors);

					neighborNumbers[pointIdx].first = currentNeighbors.size();
					neighborNumbers[pointIdx].second = pointIdx;
				}

				// Sort the points according to the ranks in a descending order
				std::sort(std::rbegin(neighborNumbers), std::rend(neighborNumbers));

				size_t maxS = 0;

				for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
				{
					if (neighborNumbers[pointIdx].first >= pointIdx)
						maxS = pointIdx;
					else
						break;
				}

				// Find all s - size kliques in the graph
				for (int s = maxS; s >= 3; --s)
				{
					auto tempNeighbors = neighbors;

					// Looping over nodes
					for (size_t n = 0; n < kPointNumber; ++n)
					{
						const size_t &A = n;
						const auto& B = tempNeighbors[n];
						std::vector<size_t> C;
						transferNodes(A, B, s, tempNeighbors, C, clusters_);
					}
				}

				/*
					cliques = cell(0);
				% Find all s - size kliques in the graph
					for s = max_s:-1 : 3
						M_aux = M;
					% Looping over nodes
					for n = 1:nb_nodes
						A = n; % Set of nodes all linked to each other
						B = setdiff(find(M_aux(n, :) == 1), n);% Set of nodes that are linked to each node in A, but not necessarily to the nodes in B
						C = transfer_nodes(A, B, s, M_aux);% Enlarging A by transferring nodes from B
						if ~isempty(C)
							for i = size(C, 1)
								cliques = [cliques; {C(i, :)}];
				end
					end
					M_aux(n, :) = 0;% Remove the processed node
					M_aux(:, n) = 0;
				end
					end

					% Generating the clique - clique overlap matrix
					CC = zeros(length(cliques));
				for c1 = 1:length(cliques)
					for c2 = c1 : length(cliques)
						if c1 == c2
							CC(c1, c2) = numel(cliques{ c1 });
						else
							CC(c1, c2) = numel(intersect(cliques{ c1 }, cliques{ c2 }));
				CC(c2, c1) = CC(c1, c2);
				end
					end
					end

					% Extracting the k - clique matrix from the clique - clique overlap matrix
					% Off - diagonal elements <= k - 1 -- > 0
					% Diagonal elements <= k-- > 0
					CC(eye(size(CC)) == 1) = CC(eye(size(CC)) == 1) - k;
				CC(eye(size(CC))~= 1) = CC(eye(size(CC))~= 1) - k + 1;
				CC(CC >= 0) = 1;
				CC(CC < 0) = 0;

				% Extracting components(or k - clique communities) from the k - clique matrix
					components = [];
				for i = 1:length(cliques)
					linked_cliques = find(CC(i, :) == 1);
				new_component = [];
				for j = 1:length(linked_cliques)
					new_component = union(new_component, cliques{ linked_cliques(j) });
				end
					found = false;
				if ~isempty(new_component)
					for j = 1:length(components)
						if all(ismember(new_component, components{ j }))
							found = true;
				end
					end
					if ~found
						components = [components; {new_component}];
				end
					end
					end


					function R = transfer_nodes(S1, S2, clique_size, C)
					% Recursive function to transfer nodes from set B to set A(as
						% defined above)

					% Check if the union of S1and S2 or S1 is inside an already found larger
					% clique
					found_s12 = false;
				found_s1 = false;
				for c = 1:length(cliques)
					for cc = 1 : size(cliques{ c }, 1)
						if all(ismember(S1, cliques{ c }(cc, :)))
							found_s1 = true;
				end
					if all(ismember(union(S1, S2), cliques{ c }(cc, :)))
						found_s12 = true;
				break;
				end
					end
					end

					if found_s12 || (length(S1) ~= clique_size && isempty(S2))
						% If the union of the sets A and B can be included in an
						% already found(larger) clique, the recursion is stepped back
						% to check other possibilities
						R = [];
				elseif length(S1) == clique_size;
				% The size of A reaches s, a new clique is found
					if found_s1
						R = [];
					else
						R = S1;
				end
					else
						% Check the remaining possible combinations of the neighbors
						% indices
						if isempty(find(S2 >= max(S1), 1))
							R = [];
						else
							R = [];
				for w = find(S2 >= max(S1), 1) :length(S2)
					S2_aux = S2;
				S1_aux = S1;
				S1_aux = [S1_aux S2_aux(w)];
				S2_aux = setdiff(S2_aux(C(S2(w), S2_aux) == 1), S2_aux(w));
				R = [R; transfer_nodes(S1_aux, S2_aux, clique_size, C)];
				end
					end
					end
					end*/
			}

			void selectNeighbors(
				const std::vector<_DataType>& data_,
				const size_t& pointIdx_,
				const std::unordered_map<std::pair<size_t, size_t>, double, PairHash>& distanceMap_,
				const double& threshold_,
				std::vector<size_t>& neighbors)
			{
				std::pair<size_t, size_t> pairIdx;

				for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
				{
					if (pointIdx == pointIdx_)
						continue;

					pairIdx.first = MIN(pointIdx, pointIdx_);
					pairIdx.second = MAX(pointIdx, pointIdx_);

					const double& distance =
						distanceMap_[pairIdx];

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