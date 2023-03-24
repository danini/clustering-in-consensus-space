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

#include "estimators/solver_engine.h"
#include "estimators/homography_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class HomographyThreePointSolver : public SolverEngine
			{
			protected:
				Eigen::Matrix3d fundamentalMatrix;
				Eigen::Vector3d epipoleLeft,
					epipoleRight;
				bool isInitialized;

			public:
				HomographyThreePointSolver() : isInitialized(false)
				{
				}

				~HomographyThreePointSolver()
				{
				}

				void setFundamentalMatrix(const Eigen::Matrix3d &fundamentalMatrix_)
				{
					fundamentalMatrix = fundamentalMatrix_;
					isInitialized = true;

					const Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr2(
						fundamentalMatrix.transpose() * fundamentalMatrix);
					const Eigen::MatrixXd& Q2 = qr2.matrixQ();
					epipoleLeft = Q2.rightCols<1>();

					const Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr1(
						fundamentalMatrix * fundamentalMatrix.transpose());
					const Eigen::MatrixXd& Q1 = qr1.matrixQ();
					epipoleRight = Q1.rightCols<1>();
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				static constexpr size_t maximumSolutions()
				{
					return 1;
				}
				
				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

				OLGA_INLINE bool HomographyThreePointSolver::estimateModel(
					const cv::Mat& data_,
					const size_t* sample_,
					size_t sample_number_,
					std::vector<Model>& models_,
					const double* weights_,
					const Eigen::Matrix3d& normalizingTransformationSource_,
					const Eigen::Matrix3d& normalizingTransformationDestination_) const;
			};

			OLGA_INLINE bool HomographyThreePointSolver::estimateModel(
				const cv::Mat& data_,
				const size_t* sample_,
				size_t sample_number_,
				std::vector<Model>& models_,
				const double* weights_) const
			{
				static const Eigen::Matrix3d normalizingTransformations =
					Eigen::Matrix3d::Identity();

				return estimateModel(
					data_,
					sample_,
					sample_number_,
					models_,
					weights_,
					normalizingTransformations,
					normalizingTransformations);
			}

			OLGA_INLINE bool HomographyThreePointSolver::estimateModel(
				const cv::Mat& data_,
				const size_t* sample_,
				size_t sample_number_,
				std::vector<Model>& models_,
				const double* weights_,
				const Eigen::Matrix3d& normalizingTransformationSource_,
				const Eigen::Matrix3d& normalizingTransformationDestination_) const
			{
				if (!isInitialized)
				{
					fprintf(stderr, "Homography estimator is not initialized.\n");
					return false;
				}

				Eigen::Matrix3d normalizedFundamentalMatrix =
					normalizingTransformationDestination_.inverse().transpose() * fundamentalMatrix * normalizingTransformationSource_.inverse();
;
				Eigen::Matrix3d covarianceMatrix = 
					normalizedFundamentalMatrix * normalizedFundamentalMatrix.transpose();
				const Eigen::FullPivLU<Eigen::MatrixXd> lu(covarianceMatrix.eval());

				if (lu.rank() != 2)
					return false;

				Eigen::Vector3d epipoleRight = lu.kernel();
				epipoleRight = epipoleRight / epipoleRight(2);

				constexpr size_t kEquationNumber = 2;
				const size_t &kColumnNumber = data_.cols;
				const size_t &kPointNumber = data_.rows;
				const size_t kRowNumber = kEquationNumber * sample_number_;
				Eigen::MatrixXd coefficients(kRowNumber, 3);
				Eigen::MatrixXd inhomogeneous(kRowNumber, 1);
 
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				size_t row_idx = 0;
				double weight = 1.0;

				for (size_t i = 0; i < sample_number_; ++i)
				{
					const size_t idx = 
						sample_ == nullptr ? i : sample_[i];

					const double *point_ptr = 
						data_ptr + idx * kColumnNumber;

					const double
						&x1 = point_ptr[0],
						&y1 = point_ptr[1],
						&x2 = point_ptr[2],
						&y2 = point_ptr[3];

					if (weights_ != nullptr)
						weight = weights_[idx];
										
					coefficients(row_idx, 0) = weight * (epipoleRight(0) * x1 - x2 * x1); 
					coefficients(row_idx, 1) = weight * (epipoleRight(0) * y1 - x2 * y1);
					coefficients(row_idx, 2) = weight * (epipoleRight(0) - x2); 
					inhomogeneous(row_idx) = -weight * (x1 * normalizedFundamentalMatrix(1, 0) + y1 * normalizedFundamentalMatrix(1, 1) + normalizedFundamentalMatrix(1, 2));
					++row_idx;

					coefficients(row_idx, 0) = weight * (epipoleRight(1) * x1 - y2 * x1); 
					coefficients(row_idx, 1) = weight * (epipoleRight(1) * y1 - y2 * y1);
					coefficients(row_idx, 2) = weight * (epipoleRight(1) - y2);
					inhomogeneous(row_idx) = weight * (x1 * normalizedFundamentalMatrix(0, 0) + y1 * normalizedFundamentalMatrix(0, 1) + normalizedFundamentalMatrix(0, 2));
					++row_idx;
				}

				Eigen::Matrix<double, 3, 1> h;

				// If we have a minimal sample, it is usually enough to solve the problem with not necessarily
				// the most accurate solver. Therefore, we use normal equations for this
				if (sample_number_ == sampleSize())
				{
					const Eigen::Matrix<double, 3, 6> coefficientsTransposed =
						coefficients.transpose();
					h = (coefficientsTransposed * coefficients).llt().solve(coefficientsTransposed * inhomogeneous);
				} 
				else // Otherwise, we want the results to be very accurate.
					h = coefficients.colPivHouseholderQr().solve(inhomogeneous);

				const double &h31 = h(0);
				const double &h32 = h(1);
				const double &h33 = h(2);

				const double h21  = epipoleRight(1) * h31 - normalizedFundamentalMatrix(0, 0);
				const double h22  = epipoleRight(1) * h32 - normalizedFundamentalMatrix(0, 1);
				const double h23  = epipoleRight(1) * h33 - normalizedFundamentalMatrix(0, 2);
				const double h11 = epipoleRight(0) * h31 + normalizedFundamentalMatrix(1, 0);
				const double h12 = epipoleRight(0) * h32 + normalizedFundamentalMatrix(1, 1);
				const double h13 = epipoleRight(0) * h33 + normalizedFundamentalMatrix(1, 2);

				Homography model;
				model.descriptor << 
					h11, h12, h13,
					h21, h22, h23,
					h31, h32, h33;
				models_.emplace_back(model);
				return true;
			}
		}
	}
}