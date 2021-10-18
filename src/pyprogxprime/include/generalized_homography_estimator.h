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

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"

#include "solver_homography_four_point.h"

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine, // The solver used for estimating the model from a non-minimal sample
			size_t _EstimateFocalLength = 0> 
			class GeneralizedHomographyEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

			std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses;
			size_t cameraNumber;
			bool initialized;

			void decomposeGenHomography(
				const cv::Mat& data_,
				const size_t* sample_,
				const size_t& sampleSize_,
				const Eigen::Matrix<double, 3, 4>& homography_,
				Eigen::Vector3d& N_,
				Eigen::Matrix3d& R_,
				Eigen::Vector3d& t_) const
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
			}

		public:
			GeneralizedHomographyEstimator() :
				initialized(false)
			{}
			~GeneralizedHomographyEstimator() {}

			void setRigPoses(const std::vector<Eigen::Matrix<double, 3, 4>>& generalizedCameraPoses_)
			{
				initialized = true;
				generalizedCameraPoses = generalizedCameraPoses_;
				cameraNumber = generalizedCameraPoses_.size();
				minimal_solver = std::make_shared<_MinimalSolverEngine>(generalizedCameraPoses, cameraNumber);
				non_minimal_solver = std::make_shared<_NonMinimalSolverEngine>(generalizedCameraPoses, cameraNumber);
			}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample required for the estimation
			static constexpr bool needInitialModel() {
				return _NonMinimalSolverEngine::needInitialModel();
			}
			
			// The size of a minimal sample required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			static constexpr bool isGeneralizedHomography()
			{
				return true;
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			_MinimalSolverEngine &getMutableMinimalSolver()
			{
				return *minimal_solver;
			}

			_NonMinimalSolverEngine &getMutableNonMinimalSolver()
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
				const size_t *sample_, // The sample usd for the estimation
				std::vector<Model>* models_) const // The estimated model parameters
			{
				if (!initialized)
				{
					fprintf(stderr, "The generalized camera's poses have not been set yet.");
					return false;
				}

				std::vector<Model> models;
				const bool success = minimal_solver->estimateModel(data_, // The data points
					sample_, // The sample used for the estimation
					sampleSize(), // The size of a minimal sample
					models); // The estimated model parameters

				if (!success)
					return false;

				models_->reserve(models.size());
				for (const auto &model : models)
				{
					Model homography;
					if constexpr (_EstimateFocalLength)
					{
						constexpr size_t offset = 9;
						homography.descriptor = Eigen::MatrixXd(3, cameraNumber * 3 + offset);
						homography.descriptor.block<3, 5>(0, 0) << model.descriptor;
						const double &focalLength = homography.descriptor(0, 4);
						Eigen::Matrix3d scaling = Eigen::Matrix3d::Identity();
						scaling(0, 0) = 1.0 / focalLength;
						scaling(1, 1) = 1.0 / focalLength;

						for (size_t cameraIdx = 0; cameraIdx < cameraNumber; ++cameraIdx)
						{
							const Eigen::Matrix3d &rotation = generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0);
							const Eigen::Vector3d &translation = generalizedCameraPoses[cameraIdx].rightCols<1>();

							// TODO(danini): translation won't work since c is needed. The only reason why it works now is that
							// the camera rotation is identity.
							homography.descriptor.block<3, 3>(0, offset + cameraIdx * 3) <<
								rotation * (model.descriptor.block<3, 3>(0, 0) +
								(-rotation.transpose() * translation) * model.descriptor.col(3).transpose()) *
								scaling;
						}
					}
					else
					{
						constexpr size_t offset = 8;
						homography.descriptor = Eigen::MatrixXd(3, cameraNumber * 3 + offset);
						homography.descriptor.block<3, 4>(0, 0) =
							model.descriptor;

						Eigen::Matrix3d R;
						Eigen::Vector3d N, t;

						std::vector<size_t> sample(data_.rows);
						std::iota(sample.begin(), sample.end(), 0);

						decomposeGenHomography(
							data_,
							&(sample[0]),
							data_.rows,
							homography.descriptor.block<3, 4>(0, 0),
							N,
							R,
							t);

						homography.descriptor.block<3, 8>(0, 0) << 
							model.descriptor.block<3, 3>(0, 0), N, R, t;

						for (size_t cameraIdx = 0; cameraIdx < cameraNumber; ++cameraIdx)
						{
							const Eigen::Matrix3d &rotation = generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0);
							const Eigen::Vector3d &translation = generalizedCameraPoses[cameraIdx].rightCols<1>();

							homography.descriptor.block<3, 3>(0, offset + cameraIdx * 3) <<
								rotation * (model.descriptor.block<3, 3>(0, 0) +
								(-rotation.transpose() * translation) * model.descriptor.col(3).transpose());
						}
					}

					models_->emplace_back(homography);
				}

				return true;
			}

			// Estimating the model from a non-minimal sample
			OLGA_INLINE bool estimateModelNonminimal(const cv::Mat& data_, // The data points
				const size_t *sample_, // The sample used for the estimation
				const size_t &sample_number_, // The size of a minimal sample
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const // The estimated model parameters
			{
				//return true;

				if (!initialized)
				{
					fprintf(stderr, "The generalized camera's poses have not been set yet.");
					return false;
				}

				// Return of there are not enough points for the estimation
				if (sample_number_ < nonMinimalSampleSize())
					return false;
								
				std::vector<Model> models;
				if constexpr (needInitialModel())
					models.emplace_back(models_->back());
				
				/*double sum1 = 0;
				for (int i = 0; i < sample_number_; ++i)
					sum1 += residual(data_.row(sample_[i]), models[0].descriptor);*/

				// The four point fundamental matrix fitting algorithm
				if (!non_minimal_solver->estimateModel(data_,
					sample_,
					sample_number_,
					models,
					weights_))
					return false;

				size_t offset = 8;
				if constexpr (_EstimateFocalLength)
					offset = 9;

				for (const auto &model : models)
				{
					Model homography;
					homography.descriptor = Eigen::MatrixXd(3, cameraNumber * 3 + offset);
					if constexpr (_EstimateFocalLength)
						homography.descriptor.block<3, 9>(0, 0) << model.descriptor.block<3, 9>(0, 0);
					else
						homography.descriptor.block<3, 8>(0, 0) << model.descriptor.block<3, 8>(0, 0);

					Eigen::Matrix3d scaling = Eigen::Matrix3d::Identity();
					if constexpr (_EstimateFocalLength)
					{
						const double &focalLength = homography.descriptor(0, 4);
						scaling(0, 0) = 1.0 / focalLength;
						scaling(1, 1) = 1.0 / focalLength;
					}

					for (size_t cameraIdx = 0; cameraIdx < cameraNumber; ++cameraIdx)
					{
						const Eigen::Matrix3d &rotation = generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0);
						const Eigen::Vector3d &translation = generalizedCameraPoses[cameraIdx].rightCols<1>();

						// TODO(danini): translation won't work since c is needed. The only reason why it works now is that
						// the camera rotation is identity.
						homography.descriptor.block<3, 3>(0, offset + cameraIdx * 3) <<
							rotation * (model.descriptor.block<3, 3>(0, 0) +
							(-rotation.transpose() * translation) * model.descriptor.col(3).transpose()) *
							scaling;

					}

					/*double sum2 = 0;
					for (int i = 0; i < sample_number_; ++i)
						sum2 += residual(data_.row(sample_[i]), homography.descriptor);*/

					//printf("%f %f\n", sum1, sum2);

					models_->back() = homography;

					//models_->emplace_back(homography);
				}
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
				const double* s = reinterpret_cast<double *>(point_.data);

				const size_t cameraIdx = 
					static_cast<size_t>(s[8]);

				if (cameraIdx == 1)
				{
					//std::cout << "asd\n";
				}

				if constexpr (_EstimateFocalLength)
				{
					const Eigen::Matrix3d &homography =
						descriptor_.block<3, 3>(0, 9 + 3 * cameraIdx);
					const double &focalLength = descriptor_(0, 4);

					const double
						&x1 = s[0],// / focalLength,
						&y1 = s[1],// / focalLength,
						&x2 = s[9],
						&y2 = s[10];

					const double t1 = homography(0, 0) * x1 + homography(0, 1) * y1 + homography(0, 2);
					const double t2 = homography(1, 0) * x1 + homography(1, 1) * y1 + homography(1, 2);
					const double t3 = homography(2, 0) * x1 + homography(2, 1) * y1 + homography(2, 2);

					const double d1 = x2 - (t1 / t3);
					const double d2 = y2 - (t2 / t3);

					return d1 * d1 + d2 * d2;
				}
				else
				{
					double hError = 1e10;

					const double
						& x1 = s[0],
						& y1 = s[1],
						& x2 = s[9],
						& y2 = s[10];

					{
						// Homography error 
						const Eigen::Matrix3d& homography =
							descriptor_.block<3, 3>(0, 8 + 3 * cameraIdx);

						const double t1 = homography(0, 0) * x1 + homography(0, 1) * y1 + homography(0, 2);
						const double t2 = homography(1, 0) * x1 + homography(1, 1) * y1 + homography(1, 2);
						const double t3 = homography(2, 0) * x1 + homography(2, 1) * y1 + homography(2, 2);

						const double d1 = x2 - (t1 / t3);
						const double d2 = y2 - (t2 / t3);

						hError = d1 * d1 + d2 * d2;
					}

					// Triangulation error
					double tError = std::numeric_limits<double>::max();

					if constexpr (true) {
						const Eigen::Matrix3d& R_H = descriptor_.block<3, 3>(0, 4);
						const Eigen::Vector3d& t = descriptor_.col(7);
						Eigen::Vector3d c(s[5], s[6], s[7]);
						const Eigen::Matrix3d& R = generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0);

						Eigen::Matrix<double, 3, 1> p_q, p_db;
						p_q << static_cast<double>(x1), static_cast<double>(y1), static_cast<double>(1.0);
						// p_db << static_cast<double>(p2D_db_x / K(0, 0)),
								// static_cast<double>(p2D_db_y / K(1, 1)), static_cast<double>(1.0);
						p_db << static_cast<double>(x2), static_cast<double>(y2), static_cast<double>(1.0);

						// Computes the closest point on one line to the second one, then checks
						// if the point is in front of both cameras.
						// See https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points for details.
						Eigen::Matrix<double, 3, 1> p1 = t;
						Eigen::Matrix<double, 3, 1> d1 = R_H * p_q;
						d1.normalize();
						Eigen::Matrix<double, 3, 1> p2 = c.template cast<double>();
						Eigen::Matrix<double, 3, 1> d2 = R.transpose().template cast<double>() * p_db;
						d2.normalize();

						Eigen::Matrix<double, 3, 1> n = d1.cross(d2);
						n.normalize();
						Eigen::Matrix<double, 3, 1> n1 = d1.cross(n);
						n1.normalize();
						Eigen::Matrix<double, 3, 1> n2 = d2.cross(n);
						n2.normalize();

						Eigen::Matrix<double, 3, 1> c1 = p1 + (p2 - p1).dot(n2) / (d1.dot(n2)) * d1;
						Eigen::Matrix<double, 3, 1> c2 = p2 + (p1 - p2).dot(n1) / (d2.dot(n1)) * d2;

						Eigen::Matrix<double, 3, 1> proj = R.template cast<double>() * (c1 - c.template cast<double>());

						double x_proj = proj[0] / proj[2];
						double y_proj = proj[1] / proj[2];

						double error1 =
							std::pow(static_cast<double>(x2) - x_proj, 2) +
							std::pow(static_cast<double>(y2) - y_proj, 2);

						proj = R_H.transpose().template cast<double>() * (c2 - t);
						x_proj = proj[0] / proj[2];
						y_proj = proj[1] / proj[2];
						double error2 =
							std::pow(static_cast<double>(x1) - x_proj, 2) +
							std::pow(static_cast<double>(y1) - y_proj, 2);

						tError = 0.5 * (error1 + error2);
					}


					/*const Eigen::Vector3d &p1 = descriptor_.col(7);// pose.c;
					// Eigen::Vector3d d1 = rotations[i] * matches_[kIdx].p2D_query.homogeneous();
					Eigen::Vector3d d1 = descriptor_.block<3, 3>(0, 4) * Eigen::Vector3d(x1, y1, 1);// matches_[i].p2D_query.homogeneous();
					d1.normalize();
					Eigen::Vector3d p2(s[5], s[6], s[7]); //= ;// cameras_[kCamIdx].c;
					Eigen::Vector3d d2(s[2], s[3], s[4]); // = matches_[i].ref_ray_dir;
					d2.normalize();

					Eigen::Vector3d n1 = d1.cross(d2);
					n1.normalize();
					Eigen::Vector3d n2 = d2.cross(n1);
					n2.normalize();

					Eigen::Vector3d c1 = p1 + (p2 - p1).dot(n2) / (d1.dot(n2)) * d1;
					Eigen::Vector3d c2 = p2 + (p1 - p2).dot(n1) / (d2.dot(n1)) * d2;

					// c1 = 0.5 * (c1 + c2);

					tError = std::numeric_limits<double>::max();

					if ((d1.dot(c1 - p1) > 0.0) && (d2.dot(c1 - p2) > 0.0)) {
						Eigen::Vector3d p = generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0) * (c1 - p2); // generalizedCameraPoses[cameraIdx].col(3));
						if (p[2] > 0.0) {
							tError = (p.hnormalized() - Eigen::Vector2d(x2, y2)).squaredNorm();
						}
					}*/
					return std::min(hError, tError);
				}
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

			OLGA_INLINE bool normalizePoints(
				const cv::Mat& data_, // The data points
				const size_t *sample_, // The points to which the model will be fit
				const size_t &sample_number_,// The number of points
				cv::Mat &normalized_points_, // The normalized point coordinates
				Eigen::Matrix3d &normalizing_transform_source_, // The normalizing transformation in the first image
				Eigen::Matrix3d &normalizing_transform_destination_) const // The normalizing transformation in the second image
			{
				const size_t cols = data_.cols;
				double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data);
				const double *points_ptr = reinterpret_cast<double *>(data_.data);

				double mass_point_src[2], // Mass point in the first image
					mass_point_dst[2]; // Mass point in the second image

				// Initializing the mass point coordinates
				mass_point_src[0] =
					mass_point_src[1] =
					mass_point_dst[0] =
					mass_point_dst[1] =
					0.0;

				// Calculating the mass points in both images
				for (size_t i = 0; i < sample_number_; ++i)
				{
					// Get pointer of the current point
					const double *d_idx = points_ptr + cols * sample_[i];

					// Add the coordinates to that of the mass points
					mass_point_src[0] += *(d_idx);
					mass_point_src[1] += *(d_idx + 1);
					mass_point_dst[0] += *(d_idx + 2);
					mass_point_dst[1] += *(d_idx + 3);
				}

				// Get the average
				mass_point_src[0] /= sample_number_;
				mass_point_src[1] /= sample_number_;
				mass_point_dst[0] /= sample_number_;
				mass_point_dst[1] /= sample_number_;

				// Get the mean distance from the mass points
				double average_distance_src = 0.0,
					average_distance_dst = 0.0;
				for (size_t i = 0; i < sample_number_; ++i)
				{
					const double *d_idx = points_ptr + cols * sample_[i];

					const double &x1 = *(d_idx);
					const double &y1 = *(d_idx + 1);
					const double &x2 = *(d_idx + 2);
					const double &y2 = *(d_idx + 3);

					const double dx1 = mass_point_src[0] - x1;
					const double dy1 = mass_point_src[1] - y1;
					const double dx2 = mass_point_dst[0] - x2;
					const double dy2 = mass_point_dst[1] - y2;

					average_distance_src += sqrt(dx1 * dx1 + dy1 * dy1);
					average_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
				}

				average_distance_src /= sample_number_;
				average_distance_dst /= sample_number_;

				// Calculate the sqrt(2) / MeanDistance ratios
				const double ratio_src = M_SQRT2 / average_distance_src;
				const double ratio_dst = M_SQRT2 / average_distance_dst;

				// Compute the normalized coordinates
				for (size_t i = 0; i < sample_number_; ++i)
				{
					const double *d_idx = points_ptr + cols * sample_[i];

					const double &x1 = *(d_idx);
					const double &y1 = *(d_idx + 1);
					const double &x2 = *(d_idx + 2);
					const double &y2 = *(d_idx + 3);

					*normalized_points_ptr++ = (x1 - mass_point_src[0]) * ratio_src;
					*normalized_points_ptr++ = (y1 - mass_point_src[1]) * ratio_src;
					*normalized_points_ptr++ = (x2 - mass_point_dst[0]) * ratio_dst;
					*normalized_points_ptr++ = (y2 - mass_point_dst[1]) * ratio_dst;

					for (size_t i = 4; i < normalized_points_.cols; ++i)
						*normalized_points_ptr++ = *(d_idx + i);
				}

				// Creating the normalizing transformations
				normalizing_transform_source_ << ratio_src, 0, -ratio_src * mass_point_src[0],
					0, ratio_src, -ratio_src * mass_point_src[1],
					0, 0, 1;

				normalizing_transform_destination_ << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
					0, ratio_dst, -ratio_dst * mass_point_dst[1],
					0, 0, 1;
				return true;
			}

			// Calculates the cross-product of two vectors
			OLGA_INLINE void cross_product(
				Eigen::Vector3d &result_,
				const double *vector1_,
				const double *vector2_,
				const unsigned int st_) const
			{
				result_[0] = vector1_[st_] - vector2_[st_];
				result_[1] = vector2_[0] - vector1_[0];
				result_[2] = vector1_[0] * vector2_[st_] - vector1_[st_] * vector2_[0];
			}

			// A function to decide if the selected sample is degenerate or not
			// before calculating the model parameters
			OLGA_INLINE bool isValidSample(
				const cv::Mat& data_, // All data points
				const size_t *sample_) const // The indices of the selected points
			{
				return true;

				// The size of a minimal sample
				constexpr size_t sample_size = sampleSize();

				// Check oriented constraints
				Eigen::Vector3d p, q;

				const double *a = reinterpret_cast<const double *>(data_.row(sample_[0]).data),
					*b = reinterpret_cast<const double *>(data_.row(sample_[1]).data),
					*c = reinterpret_cast<const double *>(data_.row(sample_[2]).data),
					*d = reinterpret_cast<const double *>(data_.row(sample_[3]).data);

				cross_product(p, a, b, 1);
				cross_product(q, a + 2, b + 2, 1);

				if ((p[0] * c[0] + p[1] * c[1] + p[2])*(q[0] * c[2] + q[1] * c[3] + q[2]) < 0)
					return false;
				if ((p[0] * d[0] + p[1] * d[1] + p[2])*(q[0] * d[2] + q[1] * d[3] + q[2]) < 0)
					return false;

				cross_product(p, c, d, 1);
				cross_product(q, c + 2, d + 2, 1);

				if ((p[0] * a[0] + p[1] * a[1] + p[2])*(q[0] * a[2] + q[1] * a[3] + q[2]) < 0)
					return false;
				if ((p[0] * b[0] + p[1] * b[1] + p[2])*(q[0] * b[2] + q[1] * b[3] + q[2]) < 0)
					return false;

				return true;
			}
		};
	}
}