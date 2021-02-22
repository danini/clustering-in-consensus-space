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

#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
		class Subspace4Estimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			Subspace4Estimator() : 
				minimal_solver(std::make_shared<_MinimalSolverEngine>()), // Minimal solver engine used for estimating a model from a minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>()) // Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			{}
			
			~Subspace4Estimator() {}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			_MinimalSolverEngine& getMutableMinimalSolver()
			{
				return *minimal_solver;
			}

			const _MinimalSolverEngine& getMinimalSolver() const
			{
				return *minimal_solver;
			}

			_NonMinimalSolverEngine& getMutableNonMinimalSolver()
			{
				return *non_minimal_solver;
			}

			const _NonMinimalSolverEngine& getNonMinimalSolver() const
			{
				return *non_minimal_solver;
			}

			// The size of a sample when doing inner RANSAC on a non-minimal sample
			OLGA_INLINE size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Estimating the model from a minimal sample
			OLGA_INLINE bool estimateModel(
				const cv::Mat& data_, // The data points
				const size_t* sample_, // The sample usd for the estimation
				std::vector<Model>* models_) const // The estimated model parameters
			{
				const bool success = minimal_solver->estimateModel(data_, // The data points
					sample_, // The sample used for the estimation
					sampleSize(), // The size of a minimal sample
					*models_);

				return success; // The estimated model parameters
			}

			// Estimating the model from a non-minimal sample
			OLGA_INLINE bool estimateModelNonminimal(const cv::Mat& data_, // The data points
				const size_t* sample_, // The sample used for the estimation
				const size_t& sample_number_, // The size of a minimal sample
				std::vector<Model>* models_,
				const double* weights_ = nullptr) const // The estimated model parameters
			{
				// Return of there are not enough points for the estimation
				if (sample_number_ < nonMinimalSampleSize())
					return false;

				bool success = non_minimal_solver->estimateModel(data_,
					sample_,
					sample_number_,
					*models_,
					weights_);

				// The four point fundamental matrix fitting algorithm
				if (!success)
					return false;
				return true;
			}

			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double* const pointPtr = reinterpret_cast<double*>(point_.data);
				
				double residual = 0;
				for (size_t r = 0; r < 5; ++r)
					residual += descriptor_(r) * pointPtr[r];

				return residual * residual;

				/*double squaredResidual = 0, tempValue;
				for (size_t r = 0; r < 5; ++r)
				{
					tempValue = 0;
					for (size_t c = 0; c < 5; ++c)
						tempValue += descriptor_(r, c) * pointPtr[c];
					squaredResidual += tempValue * tempValue;
				}

				return squaredResidual;*/
			}

			OLGA_INLINE double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			OLGA_INLINE double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return sqrt(squaredResidual(point_, descriptor_));
			}

			// before calculating the model parameters
			OLGA_INLINE bool isValidSample(
				const cv::Mat& data_, // All data points
				const size_t* sample_) const // The indices of the selected points
			{
				return true;
			}
		};
	}
}