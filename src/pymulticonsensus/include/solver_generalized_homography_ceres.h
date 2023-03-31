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
#include "pose_utils.h"
#include "generalized_homography_estimator.h"
#include <ceres/ceres.h>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// Non-linear optimization.
			struct ReprojectionError {
				// Assumes that the measurement is centered around the principal point.
				double p2D_q_x;
				double p2D_q_y;
				double p2D_db_x;
				double p2D_db_y;
				Eigen::Matrix3d K;
				Eigen::Matrix3d R;
				Eigen::Vector3d c;

				ReprojectionError(double x, double y, double x_db, double y_db,
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
					Eigen::Matrix<T, 3, 3> R_H(q);
					Eigen::Matrix<T, 3, 1> t;
					t << data[4], data[5], data[6];

					Eigen::Matrix<T, 3, 1> N;
					N << data[7], data[8], data[9];

					// Recreates the homography.
					Eigen::Matrix<T, 3, 3> H;
					H = R_H - t * N.transpose();

					Eigen::Matrix<T, 3, 3> H_cam = K.template cast<T>() * R.template cast<T>() * (H + c.template cast<T>() * N.transpose());
					Eigen::Matrix<T, 3, 1> p;
					p << static_cast<T>(p2D_q_x), static_cast<T>(p2D_q_y), static_cast<T>(1.0);
					Eigen::Matrix<T, 3, 1> p2 = H_cam * p;

					T x_proj = p2[0] / p2[2];
					T y_proj = p2[1] / p2[2];

					residuals[0] = static_cast<T>(p2D_db_x) - x_proj;
					residuals[1] = static_cast<T>(p2D_db_y) - y_proj;

					return true;
				}

				// Factory function
				static ceres::CostFunction* CreateCost(const double x, const double y,
					const double x_db, const double y_db,
					const Eigen::Matrix3d& K_,
					const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_) {
					return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 10>(
						new ReprojectionError(x, y, x_db, y_db, K_, R_, c_)));
				}
			};

			// Non-linear optimization.
			struct ReprojectionTriangulationError {
				ReprojectionTriangulationError(double x, double y, double x_db, double y_db,
					// const Eigen::Matrix3d& K_,
					const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_) :
					p2D_q_x(x),
					p2D_q_y(y),
					p2D_db_x(x_db),
					p2D_db_y(y_db)
				{
					R = R_;
					c = c_;
				}

				/*ReprojectionError(double x, double y, double x_db, double y_db,
					const Eigen::Matrix3d& K_, const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_)
					: p2D_q_x(x),
					p2D_q_y(y),
					p2D_db_x(x_db),
					p2D_db_y(y_db) {
					K = K_;
					R = R_;
					c = -c_;
				}*/


				template <typename T>
				bool operator()(const T* const data, T* residuals) const {
					// The first four entries encode the rotation matrix as a quaternion, the
					// next three entries correspond to the translation, the last three entries
					// correspond to N.
					Eigen::Quaternion<T> q(data[0], data[1], data[2], data[3]);
					q.normalize();
					Eigen::Matrix<T, 3, 3> R_H(q);
					Eigen::Matrix<T, 3, 1> t;
					t << data[4], data[5], data[6];

					// Eigen::Matrix<T, 3, 1> N;
					// N << data[7], data[8], data[9];

					Eigen::Matrix<T, 3, 1> p_q, p_db;
					p_q << static_cast<T>(p2D_q_x), static_cast<T>(p2D_q_y), static_cast<T>(1.0);
					// p_db << static_cast<T>(p2D_db_x / K(0, 0)),
							// static_cast<T>(p2D_db_y / K(1, 1)), static_cast<T>(1.0);
					p_db << static_cast<T>(p2D_db_x), static_cast<T>(p2D_db_y), static_cast<T>(1.0);

					// Computes the closest point on one line to the second one, then checks
					// if the point is in front of both cameras.
					// See https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points for details.
					Eigen::Matrix<T, 3, 1> p1 = t;
					Eigen::Matrix<T, 3, 1> d1 = R_H * p_q;
					d1.normalize();
					Eigen::Matrix<T, 3, 1> p2 = c.template cast<T>();
					Eigen::Matrix<T, 3, 1> d2 = R.transpose().template cast<T>() * p_db;
					d2.normalize();

					Eigen::Matrix<T, 3, 1> n = d1.cross(d2);
					n.normalize();
					Eigen::Matrix<T, 3, 1> n1 = d1.cross(n);
					n1.normalize();
					Eigen::Matrix<T, 3, 1> n2 = d2.cross(n);
					n2.normalize();

					Eigen::Matrix<T, 3, 1> c1 = p1 + (p2 - p1).dot(n2) / (d1.dot(n2)) * d1;
					Eigen::Matrix<T, 3, 1> c2 = p2 + (p1 - p2).dot(n1) / (d2.dot(n1)) * d2;

					Eigen::Matrix<T, 3, 1> proj = R.template cast<T>() * (c1 - c.template cast<T>());

					T x_proj = proj[0] / proj[2];
					T y_proj = proj[1] / proj[2];

					residuals[0] = static_cast<T>(p2D_db_x) - x_proj;
					residuals[1] = static_cast<T>(p2D_db_y) - y_proj;

					proj = R_H.transpose().template cast<T>() * (c2 - t);
					x_proj = proj[0] / proj[2];
					y_proj = proj[1] / proj[2];
					residuals[2] = static_cast<T>(p2D_q_x) - x_proj;
					residuals[3] = static_cast<T>(p2D_q_y) - y_proj;

					return true;
				}

				/*template <typename T>
				bool operator()(const T* const data, T* residuals) const {
					// The first four entries encode the rotation matrix as a quaternion, the
					// next three entries correspond to the translation, the last three entries
					// correspond to N.

					Eigen::Matrix<T, 3, 1> N;
					N << data[3], data[7], data[11];

					// Recreates the homography.
					Eigen::Matrix<T, 3, 3> H;
					H << data[0], data[1], data[2],
						data[4], data[5], data[6],
						data[8], data[9], data[10];

					Eigen::Matrix<T, 3, 3> H_cam = K.template cast<T>() * R.template cast<T>() * (H + c.template cast<T>() * N.transpose());
					Eigen::Matrix<T, 3, 1> p;
					p << static_cast<T>(p2D_q_x), 
						static_cast<T>(p2D_q_y), 
						static_cast<T>(1.0);
					Eigen::Matrix<T, 3, 1> p2 = H_cam * p;

					T x_proj = p2[0] / p2[2];
					T y_proj = p2[1] / p2[2];

					residuals[0] = static_cast<T>(p2D_db_x) - x_proj;//, static_cast<T>(2));
					residuals[1] = static_cast<T>(p2D_db_y) - y_proj;// , static_cast<T>(2));
					return true;
				}*/

				// Factory function
				/*static ceres::CostFunction* CreateCost(const double x, const double y,
					const double x_db, const double y_db,
					const Eigen::Matrix3d& K_,
					const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_) {
					return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 12>(
						new ReprojectionError(x, y, x_db, y_db, K_, R_, c_)));
				}*/

				// Factory function
				static ceres::CostFunction* CreateCost(const double x, const double y,
					const double x_db, const double y_db,
					// const Eigen::Matrix3d& K_,
					const Eigen::Matrix3d& R_,
					const Eigen::Vector3d& c_) {
					return (new ceres::AutoDiffCostFunction<ReprojectionTriangulationError, 4, 7>(
						// new ReprojectionTriangulationError(x, y, x_db, y_db, K_, R_, c_)));
						new ReprojectionTriangulationError(x, y, x_db, y_db, R_, c_)));
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
					const Eigen::Matrix<double, 3, 4>& homography_,
					Eigen::Vector3d &N_,
					Eigen::Matrix3d &R_,
					Eigen::Vector3d &t_) const;

			protected:
				const std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses;
				const size_t cameraNumber;
			};
			
			void GeneralizedHomographyCeresSolver::decomposeGenHomography(
				const cv::Mat &data_,
				const size_t *sample_,
				const size_t &sampleSize_,
				const Eigen::Matrix<double, 3, 4> &homography_,
				Eigen::Vector3d &N_,
				Eigen::Matrix3d &R_,
				Eigen::Vector3d &t_) const 
			{
				std::vector<Eigen::Matrix3d> Rs;
				std::vector<Eigen::Vector3d> ts;
				std::vector<Eigen::Vector3d> ns;
				/*pose::decomposeHomographyMatrix(
					homography_.block<3, 3>(0, 0),
					Eigen::Matrix3d::Identity(),
					Eigen::Matrix3d::Identity(), 
					Rs, ts, ns);*/
				pose::DecomposeHomography(homography_.block<3, 3>(0, 0),
					&Rs,
					&ts,
					&ns);

				// int num_solutions = static_cast<int>(rotations.size());
				int num_solutions = static_cast<int>(Rs.size());
				if (num_solutions == 0) 
					return;

				int best_num_consistent = 0;
				double best_score = std::numeric_limits<double>::max();
				int best_pose = -1;

				Eigen::Vector3d N = homography_.col(3);
				double N_norm = N.norm();
				N.normalize();

				const int kNumInliers = static_cast<int>(sampleSize_);
				for (int i = 0; i < num_solutions; ++i) {
					Eigen::Quaterniond q(Rs[i]);
					q.normalize();
					Rs[i] = q;
					// if (std::fabs(N.dot(ns[i])) < 0.9) continue;
					if (Rs[i].determinant() < 0.0) continue;
					// Rs[i] = cameras_[0].R.transpose() * Rs[i];
					// ts[i] = cameras_[0].R.transpose() * ts[i] / pose->N.norm() + cameras_[0].c;
					ts[i] /= homography_.col(3).norm();
					int num_consistent = 0;
					// Performs a cheirality check.
					double score = 0.0;
					for (int j = 0; j < kNumInliers; ++j) {
						const int kIdx = sample_[j];
						const int kCamIdx = data_.at<double>(kIdx, 8); //  matches_[kIdx].camera_id;
						// Computes the closest point on one line to the second one, then checks
						// if the point is in front of both cameras.
						// See https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points for details.
						// Eigen::Vector3d p1 = translations[i];
						Eigen::Vector3d p1 = ts[i];
						// Eigen::Vector3d d1 = rotations[i] * matches_[kIdx].p2D_query.homogeneous();
						Eigen::Vector3d d1 = Rs[i] * Eigen::Vector3d(data_.at<double>(kIdx, 0), data_.at<double>(kIdx, 1), 1); //  matches_[kIdx].p2D_query.homogeneous();
						d1.normalize();
						Eigen::Vector3d p2(data_.at<double>(kIdx, 5), data_.at<double>(kIdx, 6), data_.at<double>(kIdx, 7)); // = cameras_[kCamIdx].c;
						Eigen::Vector3d d2(data_.at<double>(kIdx, 2), data_.at<double>(kIdx, 3), data_.at<double>(kIdx, 4)); // matches_[kIdx].ref_ray_dir;
						d2.normalize();

						Eigen::Vector3d n1 = d1.cross(d2);
						n1.normalize();
						Eigen::Vector3d n2 = d2.cross(n1);
						n2.normalize();

						Eigen::Vector3d c1 = p1 + (p2 - p1).dot(n2) / (d1.dot(n2)) * d1;

						double error = 0.05 * 0.05;

						if ((d1.dot(c1 - p1) > 0.0) && (d2.dot(c1 - p2) > 0.0)) {
							Eigen::Vector3d p = (/*cameras_[kCamIdx].K * */generalizedCameraPoses[kCamIdx].block<3, 3>(0, 0) * (c1 - p2)); // cameras_[kCamIdx].c));
							if (p[2] > 0.0) {
								error = std::min((p.hnormalized() - Eigen::Vector2d(data_.at<double>(kIdx, 9), data_.at<double>(kIdx, 10)) /*matches_[kIdx].p2D_db*/).squaredNorm(), error);
								++num_consistent;
							}
						}

						score += error;
					}

					// std::cout << i << " " << num_consistent << " " << score << std::endl;
					// std::cout << Rs[i] << std::endl;
					// std::cout << ts[i].transpose() << std::endl;

					// if (num_consistent > best_num_consistent) {
					if (score < best_score) {
						best_num_consistent = num_consistent;
						best_pose = i;
						best_score = score;
					}
				}
				if (best_pose < 0) {
					return;
					/*R_ = Eigen::Matrix3d::Identity();
					t_ = Eigen::Vector3d(0.0, 0.0, 0.0);
					N_ = pose->c;*/
				}
				else {
					R_ = Rs[best_pose];
					t_ = ts[best_pose];
					N_ = ns[best_pose] * N_norm;
				}
				//pose->H = pose->R - pose->c * pose->N.transpose();
				//return (best_pose >= 0);
				// std::cout << best_pose << " " << best_num_consistent << " " << kNumInliers << std::endl;
				// pose->R = Rs[best_pose];
				// pose->c = ts[best_pose];
				// std::cout << std::fabs(N.dot(ns[best_pose])) << std::endl;
				// pose->R = rotations[best_pose];
				// pose->c = translations[best_pose];
				// std::cout << pose->R << std::endl;
				// std::cout << pose->c.transpose() << std::endl << std::endl;
			}

			OLGA_INLINE bool GeneralizedHomographyCeresSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				Eigen::Vector3d N, c;
				Eigen::Matrix3d R;

				decomposeGenHomography(data_,
					sample_,
					sample_number_,
					models_.back().descriptor.block<3, 4>(0, 0),
					N,
					R,
					c);

				Eigen::Matrix3d E;

				double H[10];
				Eigen::Quaterniond q(R);
				q.normalize();
				H[0] = q.w();
				H[1] = q.x();
				H[2] = q.y();
				H[3] = q.z();

				H[4] = c[0];
				H[5] = c[1];
				H[6] = c[2];
				H[7] = N[0];
				H[8] = N[1];
				H[9] = N[2];

				ceres::Problem refinement_problem;
				const int kSampleSize = static_cast<int>(sample_number_);
				double score_before = 0.0;
				int num_epipolar = 0;
				int num_homography = 0;
				for (int i = 0; i < kSampleSize; ++i) 
				{
					const int kIdx = sample_[i];

					size_t cameraIdx = data_.at<double>(kIdx, 8);
					Eigen::Vector2d p_q(data_.at<double>(kIdx, 0), data_.at<double>(kIdx, 1));
					Eigen::Vector2d p_db(data_.at<double>(kIdx, 9), data_.at<double>(kIdx, 10));

					ceres::CostFunction* cost_function =
						ReprojectionTriangulationError::CreateCost(
							p_q[0], p_q[1], p_db[0], p_db[1],
							generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0),
							-generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0).transpose() * generalizedCameraPoses[cameraIdx].col(3));
					refinement_problem.AddResidualBlock(cost_function, nullptr, H);
					++num_epipolar;
				}

				ceres::Solver::Options options;
				options.linear_solver_type = ceres::DENSE_QR;
				options.minimizer_progress_to_stdout = false;
				ceres::Solver::Summary summary;
				ceres::Solve(options, &refinement_problem, &summary);

				double score_after = 0.0;
				if (summary.IsSolutionUsable() && (summary.termination_type == ceres::CONVERGENCE)) 
				{
					models_[0].descriptor.resize(3, 8);
					models_[0].descriptor.col(3) << H[7], H[8], H[9];

					Eigen::Quaterniond qq(H[0], H[1], H[2], H[3]);
					Eigen::Matrix3d R(qq);
					Eigen::Vector3d c(H[4], H[5], H[6]);

					models_[0].descriptor.block<3, 3>(0, 4) << R;
					models_[0].descriptor.col(7) << c;

					models_[0].descriptor.block<3, 3>(0, 0) <<
						R + c * models_[0].descriptor.col(3).transpose();
				}
				else
					models_.resize(0);
				return models_.size() > 0;

				// std::cout << score_before << " " << score_after << std::endl;
				// std::cout << " finished least squares" << std::endl;

				/*const size_t kColumns = data_.cols;
				const double *data_ptr = reinterpret_cast<double *>(data_.data);

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
						ReprojectionError::CreateCost(
							//point_ptr[9], point_ptr[10], point_ptr[0], point_ptr[1],
							point_ptr[0], point_ptr[1], point_ptr[9], point_ptr[10],
							//Eigen::Matrix3d::Identity(),
							generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0),
							generalizedCameraPoses[cameraIdx].rightCols<1>());

					refinement_problem.AddResidualBlock(cost_function, nullptr, H2);
				}

				ceres::Solver::Options options;

				options.linear_solver_type = ceres::DENSE_QR;
				options.minimizer_progress_to_stdout = false;
				options.max_num_iterations = 50;
				ceres::Solver::Summary summary;
				ceres::Solve(options, &refinement_problem, &summary);

				if (summary.IsSolutionUsable() && 
					summary.termination_type == ceres::CONVERGENCE) {

					Homography model;
					model.descriptor.resize(3, 4);
					
					models_[0].descriptor(0, 0)	= H2[0];
					models_[0].descriptor(0, 1)	= H2[1];
					models_[0].descriptor(0, 2)	= H2[2];
					models_[0].descriptor(0, 3)	= H2[3];
					models_[0].descriptor(1, 0)	= H2[4];
					models_[0].descriptor(1, 1)	= H2[5];
					models_[0].descriptor(1, 2)	= H2[6];
					models_[0].descriptor(1, 3)	= H2[7];
					models_[0].descriptor(2, 0)	= H2[8];
					models_[0].descriptor(2, 1)	= H2[9];
					models_[0].descriptor(2, 2)	= H2[10];
					models_[0].descriptor(2, 3)	= H2[11];
				}
				else
					return false;

				return models_.size() > 0;*/
			}
		}
	}
}