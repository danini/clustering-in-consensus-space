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

#include "solver_engine.h"
#include "generalized_homography_estimator.h"
#include <ceres/ceres.h>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// Non-linear optimization.
			template <size_t _EstimateFocalLength = 0>
			struct SampsonError {
				SampsonError(double x, double y, double x_db, double y_db,
					const Eigen::Matrix3d& K_, const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_)
					: p2D_q_x(x),
					p2D_q_y(y),
					p2D_db_x(x_db),
					p2D_db_y(y_db) {
					K = K_;
					R = R_;
					c = c_;
				}

				template <typename T>
				bool operator()(const T* const data, T* residuals) const {
					// The first four entries encode the rotation matrix as a quaternion, the
					// next three entries correspond to the translation, the last three entries
					// correspond to N.
					Eigen::Quaternion<T> q(data[0], data[1], data[2], data[3]);
					q.normalize();
					Eigen::Matrix<T, 3, 3> Rcurr(q);
					Eigen::Matrix<T, 3, 1> t;
					t << data[4], data[5], data[6];

					Eigen::Matrix<T, 3, 1> cameraTranslation =
						R.template cast<T>() * c.template cast<T>() + t;

					T focalLength = static_cast<T>(1.0);
					if constexpr (_EstimateFocalLength)
						focalLength = static_cast<T>(data[7]);

					const T zero = static_cast<T>(0);

					// The cross product matrix of the translation vector
					Eigen::Matrix<T, 3, 3> cross_prod_t;
					cross_prod_t << zero, -cameraTranslation(2), cameraTranslation(1),
						cameraTranslation(2), zero, -cameraTranslation(0),
						-cameraTranslation(1), cameraTranslation(0), zero;

					Eigen::Matrix<T, 3, 3> E =
						cross_prod_t *
						R.template cast<T>() * Rcurr;
					
					Eigen::Matrix<T, 3, 1> p;
					p << static_cast<T>(p2D_q_x), static_cast<T>(p2D_q_y), static_cast<T>(1.0);

					const T
						x1 = static_cast<T>(p2D_q_x) / focalLength,
						y1 = static_cast<T>(p2D_q_y) / focalLength,
						x2 = static_cast<T>(p2D_db_x),
						y2 = static_cast<T>(p2D_db_y);

					const T
						&e11 = E(0, 0),
						&e12 = E(0, 1),
						&e13 = E(0, 2),
						&e21 = E(1, 0),
						&e22 = E(1, 1),
						&e23 = E(1, 2),
						&e31 = E(2, 0),
						&e32 = E(2, 1),
						&e33 = E(2, 2);

					T rxc = e11 * x2 + e21 * y2 + e31;
					T ryc = e12 * x2 + e22 * y2 + e32;
					T rwc = e13 * x2 + e23 * y2 + e33;
					T r = (x1 * rxc + y1 * ryc + rwc);
					T rx = e11 * x1 + e12 * y1 + e13;
					T ry = e21 * x1 + e22 * y1 + e23;

					residuals[0] = r * r /
						(rxc * rxc + ryc * ryc + rx * rx + ry * ry);

					T a = rxc * rxc + ryc * ryc;
					T b = rx * rx + ry * ry;

					residuals[1] = r * r * (a + b) / (a * b);
					return true;
				}

				// Factory function
				static ceres::CostFunction* CreateCost(const double x, const double y,
					const double x_db, const double y_db,
					const Eigen::Matrix3d& K_,
					const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_) {
					if constexpr (_EstimateFocalLength)
						return (new ceres::AutoDiffCostFunction<SampsonError, 2, 8>(
							new SampsonError(x, y, x_db, y_db, K_, R_, c_)));
					return (new ceres::AutoDiffCostFunction<SampsonError, 2, 7>(
						new SampsonError(x, y, x_db, y_db, K_, R_, c_)));
				}

				// Assumes that the measurement is centered around the principal point.
				double p2D_q_x;
				double p2D_q_y;
				double p2D_db_x;
				double p2D_db_y;
				Eigen::Matrix3d K;
				Eigen::Matrix3d R;
				Eigen::Vector3d c;
			};

			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			template <size_t _EstimateFocalLength = 0>
			class GeneralizedEssentialMatrixCeresSolver : public SolverEngine
			{
			public:
				GeneralizedEssentialMatrixCeresSolver(
					const std::vector<Eigen::Matrix<double, 3, 4>> &generalizedCameraPoses_,
					const size_t cameraNumber_) :
					generalizedCameraPoses(generalizedCameraPoses_),
					cameraNumber(cameraNumber_)
				{
				}

				~GeneralizedEssentialMatrixCeresSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return false;
				}

				static constexpr size_t maximumSolutions()
				{
					return 1;
				}
				
				static constexpr bool needInitialModel()
				{
					return true;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 6;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				const std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses;
				const size_t cameraNumber;
			};

			template <size_t _EstimateFocalLength>
			OLGA_INLINE bool GeneralizedEssentialMatrixCeresSolver<_EstimateFocalLength>::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				const size_t kColumns = data_.cols;
				const double *data_ptr = reinterpret_cast<double *>(data_.data);

				const Eigen::Matrix3d &R = 
					models_[0].descriptor.block<3, 3>(0, 0);
				const Eigen::Vector3d &t =
					models_[0].descriptor.col(3);
				
				double E[8];
				Eigen::Quaterniond q(R);
				q.normalize();
				E[0] = q.w();
				E[1] = q.x();
				E[2] = q.y();
				E[3] = q.z();
				E[4] = t[0];
				E[5] = t[1];
				E[6] = t[2];
				if constexpr (_EstimateFocalLength)
					E[7] = models_[0].descriptor(0, 4);
				
				ceres::Problem refinement_problem;
				const int kSampleSize = static_cast<int>(sample_number_);
				for (int i = 0; i < kSampleSize; ++i) {
					const int kIdx = sample_[i];

					const size_t idx =
						sample_ == nullptr ? i : sample_[i];

					const double *point_ptr =
						data_ptr + idx * kColumns;

					const int cameraIdx =
						point_ptr[8];

					ceres::CostFunction* cost_function =
						SampsonError<_EstimateFocalLength>::CreateCost(
							//point_ptr[9], point_ptr[10], point_ptr[0], point_ptr[1],
							point_ptr[0], point_ptr[1], point_ptr[9], point_ptr[10],
							Eigen::Matrix3d::Identity(),
							generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0),
							generalizedCameraPoses[cameraIdx].rightCols<1>());

					refinement_problem.AddResidualBlock(cost_function, nullptr, E);
				}

				ceres::Solver::Options options;
				options.linear_solver_type = ceres::DENSE_QR;
				options.minimizer_progress_to_stdout = false;
				ceres::Solver::Summary summary;
				ceres::Solve(options, &refinement_problem, &summary);

				if (summary.IsSolutionUsable() &&
					summary.termination_type == ceres::CONVERGENCE) {

					Eigen::Quaterniond qq(E[0], E[1], E[2], E[3]);
					Eigen::Matrix3d newR;
					newR = qq;
					Eigen::Vector3d newt;
					newt << E[4], E[5], E[6];


					Model model;
					if constexpr (_EstimateFocalLength)
						model.descriptor = Eigen::MatrixXd(3, 5);
					else
						model.descriptor = Eigen::MatrixXd(3, 4);
					
					model.descriptor.block<3, 4>(0, 0) << newR, newt;
					if constexpr (_EstimateFocalLength)
						model.descriptor(0, 4) = E[7];
					models_.clear();
					models_.emplace_back(model);
				}
				else
					return false;

				return models_.size() > 0;
			}
		}
	}
}