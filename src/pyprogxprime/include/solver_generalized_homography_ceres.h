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
			struct ReprojectionError {
				ReprojectionError(double x, double y, double x_db, double y_db,
					const Eigen::Matrix3d& K_, const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_)
					: p2D_q_x(x),
					p2D_q_y(y),
					p2D_db_x(x_db),
					p2D_db_y(y_db) {
					K = K_;
					R = R_;
					c = -c_;
				}

				template <typename T>
				bool operator()(const T* const data, T* residuals) const {
					// The first four entries encode the rotation matrix as a quaternion, the
					// next three entries correspond to the translation, the last three entries
					// correspond to N.
					/*Eigen::Quaternion<T> q(data[0], data[1], data[2], data[3]);
					q.normalize();
					Eigen::Matrix<T, 3, 3> R_H(q);
					Eigen::Matrix<T, 3, 1> t;
					t << data[4], data[5], data[6];*/

					Eigen::Matrix<T, 3, 1> N;
					N << data[3], data[7], data[11];

					// Recreates the homography.
					Eigen::Matrix<T, 3, 3> H;
					//H = R_H - t * N.transpose();*/
					H << data[0], data[1], data[2],
						data[4], data[5], data[6],
						data[8], data[9], data[10];

					T focalLength = static_cast<T>(1.0);
					if constexpr (_EstimateFocalLength)
						focalLength = static_cast<T>(data[12]);

					Eigen::Matrix<T, 3, 3> H_cam = K.template cast<T>() * R.template cast<T>() * (H + c.template cast<T>() * N.transpose());
					Eigen::Matrix<T, 3, 1> p;
					p << static_cast<T>(p2D_q_x) / focalLength, 
						static_cast<T>(p2D_q_y) / focalLength, 
						static_cast<T>(1.0);
					Eigen::Matrix<T, 3, 1> p2 = H_cam * p;

					T x_proj = p2[0] / p2[2];
					T y_proj = p2[1] / p2[2];

					residuals[0] = /*pow(*/static_cast<T>(p2D_db_x) - x_proj;//, static_cast<T>(2));
					residuals[1] = /*pow(*/static_cast<T>(p2D_db_y) - y_proj;// , static_cast<T>(2));
					return true;
				}

				// Factory function
				static ceres::CostFunction* CreateCost(const double x, const double y,
					const double x_db, const double y_db,
					const Eigen::Matrix3d& K_,
					const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_) {
					if constexpr (_EstimateFocalLength)
						return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 13>(
							new ReprojectionError(x, y, x_db, y_db, K_, R_, c_)));
					return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 12>(
						new ReprojectionError(x, y, x_db, y_db, K_, R_, c_)));
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
			class GeneralizedHomographyCeresSolver : public SolverEngine
			{
			public:
				GeneralizedHomographyCeresSolver(
					const std::vector<Eigen::Matrix<double, 3, 4>> &generalizedCameraPoses_,
					const size_t cameraNumber_) :
					generalizedCameraPoses(generalizedCameraPoses_),
					cameraNumber(cameraNumber_)
				{
				}

				~GeneralizedHomographyCeresSolver()
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
					return 5;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
				
				void GeneralizedHomographyCeresSolver::decomposeGenHomography(
					const cv::Mat &data_,
					const size_t *sample_,
					const size_t &sampleSize_,
					const Eigen::Matrix3d &homography_,
					Eigen::Vector3d &N_,
					Eigen::Matrix3d &R_,
					Eigen::Vector3d &t_) const;

			protected:
				const std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses;
				const size_t cameraNumber;
			};
			
			template <size_t _EstimateFocalLength>
			void GeneralizedHomographyCeresSolver<_EstimateFocalLength>::decomposeGenHomography(
				const cv::Mat &data_,
				const size_t *sample_,
				const size_t &sampleSize_,
				const Eigen::Matrix3d &homography_,
				Eigen::Vector3d &N_,
				Eigen::Matrix3d &R_,
				Eigen::Vector3d &t_) const 
			{
				std::vector<Eigen::Matrix3d> rotations;
				std::vector<Eigen::Vector3d> translations;
				std::vector<Eigen::Vector3d> normals;

				gcransac::utils::decomposeHomographyMatrix(
					homography_.block<3, 3>(0, 0),
					Eigen::Matrix3d::Identity(),
					Eigen::Matrix3d::Identity(),
					rotations,
					translations,
					normals);

				//extract_original_pose(N_, homography_, &rotations, &translations);
				int num_solutions = static_cast<int>(rotations.size());
				if (num_solutions == 0) return;

				double best_score = std::numeric_limits<double>::max();
				int best_num_consistent = 0;
				int best_pose = 0;

				for (int i = 0; i < num_solutions; ++i) 
				{
					int num_consistent = 0;
					double score = 0.0;

					// Performs a cheirality check.
					for (int j = 0; j < sampleSize_; ++j)
					{
						const int kIdx = sample_[j];

						const double
							&x1 = data_.at<double>(kIdx, 0),
							&y1 = data_.at<double>(kIdx, 1),
							&x2 = data_.at<double>(kIdx, 9),
							&y2 = data_.at<double>(kIdx, 10);
						
						const int cameraIdx =
							data_.at<double>(kIdx, 8);
						
						Eigen::Vector2d p2D_db, p2D_q;
						p2D_db << x2, y2;
						p2D_q << x1, y1;

						// Computes the closest point on one line to the second one, then checks
						// if the point is in front of both cameras.
						// See https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points for details.
						Eigen::Vector3d p1 = translations[i];
						Eigen::Vector3d d1 = rotations[i] * p2D_q.homogeneous();
						d1.normalize();
						Eigen::Vector3d p2;
						p2 << data_.at<double>(kIdx, 5), data_.at<double>(kIdx, 6), data_.at<double>(kIdx, 7);

						Eigen::Vector3d d2;
						d2 << data_.at<double>(kIdx, 2), data_.at<double>(kIdx, 3), data_.at<double>(kIdx, 4);
						d2.normalize();

						Eigen::Vector3d n1 = d1.cross(d2);
						Eigen::Vector3d n2 = d2.cross(n1);

						Eigen::Vector3d c1 = p1 + (p2 - p1).dot(n2) / (d1.dot(n2)) * d1;
						double error = 3*3;

						if ((d1.dot(c1 - p1) > 0.0) && (d2.dot(c1 - p2) > 0.0)) {
							Eigen::Vector3d p = (generalizedCameraPoses[cameraIdx].block<3,3>(0,0) * (c1 - generalizedCameraPoses[cameraIdx].col(3)));
							if (p[2] > 0.0) {
								error = MIN((p.hnormalized() - p2D_db).squaredNorm(), error);
								++num_consistent;
							}
						}

						score += error;
					}

					if (score < best_score) {
						best_num_consistent = num_consistent;
						best_pose = i;
						best_score = score;
					}
				}

			//	N_ = normals[best_pose];
				R_ = rotations[best_pose];
				t_ = translations[best_pose];
			}

			template <size_t _EstimateFocalLength>
			OLGA_INLINE bool GeneralizedHomographyCeresSolver<_EstimateFocalLength>::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				const size_t kColumns = data_.cols;
				const double *data_ptr = reinterpret_cast<double *>(data_.data);

				/*Eigen::Matrix3d R;
				Eigen::Vector3d t;
				Eigen::Vector3d N =
					models_[0].descriptor.col(3);
				
				decomposeGenHomography(
					data_,
					sample_,
					sample_number_,
					models_[0].descriptor.block<3, 3>(0, 0),
					N,
					R,
					t);

				double H[10];
				Eigen::Quaterniond q(R);
				q.normalize();
				H[0] = q.w();
				H[1] = q.x();
				H[2] = q.y();
				H[3] = q.z();

				H[4] = t[0];
				H[5] = t[1];
				H[6] = t[2];
				H[7] = N[0];
				H[8] = N[1];
				H[9] = N[2];*/

				double H2[13];
				H2[0] = models_[0].descriptor(0, 0);
				H2[1] = models_[0].descriptor(0, 1);
				H2[2] = models_[0].descriptor(0, 2);
				H2[3] = models_[0].descriptor(0, 3);
				H2[4] = models_[0].descriptor(1, 0);
				H2[5] = models_[0].descriptor(1, 1);
				H2[6] = models_[0].descriptor(1, 2);
				H2[7] = models_[0].descriptor(1, 3);
				H2[8] = models_[0].descriptor(2, 0);
				H2[9] = models_[0].descriptor(2, 1);
				H2[10] = models_[0].descriptor(2, 2);
				H2[11] = models_[0].descriptor(2, 3);
				if constexpr (_EstimateFocalLength)
					H2[12] = models_[0].descriptor(0, 4);
				
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
						ReprojectionError<_EstimateFocalLength>::CreateCost(
							//point_ptr[9], point_ptr[10], point_ptr[0], point_ptr[1],
							point_ptr[0], point_ptr[1], point_ptr[9], point_ptr[10],
							Eigen::Matrix3d::Identity(),
							generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0),
							generalizedCameraPoses[cameraIdx].rightCols<1>());

					refinement_problem.AddResidualBlock(cost_function, nullptr, H2);
				}

				ceres::Solver::Options options;
				options.linear_solver_type = ceres::DENSE_QR;
				options.minimizer_progress_to_stdout = false;
				ceres::Solver::Summary summary;
				ceres::Solve(options, &refinement_problem, &summary);

				if (summary.IsSolutionUsable() && 
					summary.termination_type == ceres::CONVERGENCE) {
					/*Eigen::Vector3d newN = Eigen::Vector3d(H[7], H[8], H[9]);

					Eigen::Quaterniond qq(H[0], H[1], H[2], H[3]);
					Eigen::Matrix3d newR;
					newR = qq;
					Eigen::Vector3d newt;
					newt << H[4], H[5], H[6];

					Eigen::Matrix3d newH = newR - newt * newN.transpose();*/

					Homography model;
					if constexpr (_EstimateFocalLength)
						model.descriptor = Eigen::MatrixXd(3, 5);
					else
						model.descriptor = Eigen::MatrixXd(3, 4);
					//model.descriptor << newH, newN;
					
					model.descriptor(0, 0)	= H2[0];
					model.descriptor(0, 1)	= H2[1];
					model.descriptor(0, 2)	= H2[2];
					model.descriptor(0, 3)	= H2[3];
					model.descriptor(1, 0)	= H2[4];
					model.descriptor(1, 1)	= H2[5];
					model.descriptor(1, 2)	= H2[6];
					model.descriptor(1, 3)	= H2[7];
					model.descriptor(2, 0)	= H2[8];
					model.descriptor(2, 1)	= H2[9];
					model.descriptor(2, 2)	= H2[10];
					model.descriptor(2, 3)	= H2[11];
					if constexpr (_EstimateFocalLength)
						model.descriptor(0, 4) = H2[12];

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