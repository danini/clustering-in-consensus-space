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
						homography.descriptor = Eigen::MatrixXd(3, cameraNumber * 3 + 5);
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
							homography.descriptor.block<3, 3>(0, 5 + cameraIdx * 3) <<
								rotation * (model.descriptor.block<3, 3>(0, 0) +
								(-translation) * model.descriptor.col(3).transpose()) *
								scaling;
						}
					}
					else
					{
						homography.descriptor = Eigen::MatrixXd(3, cameraNumber * 3 + 4);
						homography.descriptor.block<3, 4>(0, 0) << model.descriptor;

						for (size_t cameraIdx = 0; cameraIdx < cameraNumber; ++cameraIdx)
						{
							const Eigen::Matrix3d &rotation = generalizedCameraPoses[cameraIdx].block<3, 3>(0, 0);
							const Eigen::Vector3d &translation = generalizedCameraPoses[cameraIdx].rightCols<1>();

							// TODO(danini): translation won't work since c is needed. The only reason why it works now is that
							// the camera rotation is identity.
							homography.descriptor.block<3, 3>(0, 4 + cameraIdx * 3) <<
								rotation * (model.descriptor.block<3, 3>(0, 0) +
								(-translation) * model.descriptor.rightCols<1>().transpose());
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
					models.emplace_back(models_->at(0));
				
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

				size_t offset = 4;
				if constexpr (_EstimateFocalLength)
					offset = 5;

				models_->clear();
				models_->reserve(models.size());
				for (const auto &model : models)
				{
					Model homography;
					homography.descriptor = Eigen::MatrixXd(3, cameraNumber * 3 + offset);
					if constexpr (_EstimateFocalLength)
						homography.descriptor.block<3, 5>(0, 0) << model.descriptor;
					else
						homography.descriptor.block<3, 4>(0, 0) << model.descriptor;

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
							(-translation) * model.descriptor.col(3).transpose()) * 
							scaling;

					}

					/*double sum2 = 0;
					for (int i = 0; i < sample_number_; ++i)
						sum2 += residual(data_.row(sample_[i]), homography.descriptor);

					printf("%f %f\n", sum1, sum2);*/

					models_->emplace_back(homography);
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
						descriptor_.block<3, 3>(0, 5 + 3 * cameraIdx);
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
					const Eigen::Matrix3d &homography =
						descriptor_.block<3, 3>(0, 4 + 3 * cameraIdx);

					const double
						&x1 = s[0],
						&y1 = s[1],
						&x2 = s[9],
						&y2 = s[10];

					const double t1 = homography(0, 0) * x1 + homography(0, 1) * y1 + homography(0, 2);
					const double t2 = homography(1, 0) * x1 + homography(1, 1) * y1 + homography(1, 2);
					const double t3 = homography(2, 0) * x1 + homography(2, 1) * y1 + homography(2, 2);

					const double d1 = x2 - (t1 / t3);
					const double d2 = y2 - (t2 / t3);

					return d1 * d1 + d2 * d2;
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