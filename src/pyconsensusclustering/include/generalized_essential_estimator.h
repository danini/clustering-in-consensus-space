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

namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model_ estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model_ from a minimal sample
			class _NonMinimalSolverEngine, // The solver used for estimating the model from a non-minimal sample
			size_t _EstimateFocalLength = 0> 
			class GeneralizedEssentialMatrixEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model_ from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model_ from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

			const std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses;
			const size_t cameraNumber;
			
		public:
			GeneralizedEssentialMatrixEstimator(
				const std::vector<Eigen::Matrix<double, 3, 4>> &generalizedCameraPoses_) :
				generalizedCameraPoses(generalizedCameraPoses_),
				cameraNumber(generalizedCameraPoses_.size()),
				// Minimal solver engine used for estimating a model from a minimal sample
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>(generalizedCameraPoses_, generalizedCameraPoses_.size()))
			{}
			~GeneralizedEssentialMatrixEstimator() {}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			_MinimalSolverEngine &getMutableMinimalSolver()
			{
				return *minimal_solver;
			}

			_NonMinimalSolverEngine &getMutableNonMinimalSolver()
			{
				return *non_minimal_solver;
			}
			
			static constexpr bool isGeneralizedPose()
			{
				return true;
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool needInitialModel() {
				return _NonMinimalSolverEngine::needInitialModel();
			}

			// The size of a sample_ when doing inner RANSAC on a non-minimal sample
			OLGA_INLINE size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			// Estimating the essential matrix from a minimal sample
			OLGA_INLINE bool estimateModel(const cv::Mat& data, // The data_ points
				const size_t *sample, // The selected sample_ which will be used for estimation
				std::vector<Model>* models) const // The estimated model_ parameters
			{
				constexpr size_t sample_size = sampleSize(); // The size of a minimal sample

				// Estimating the model_ parameters by the solver engine
				if (!minimal_solver->estimateModel(data, // The data points
					sample, // The selected sample which will be used for estimation
					sample_size, // The size of a minimal sample required for the estimation
					*models)) // The estimated model_ parameters
					return false;
							 
				for (auto &model : *models)
				{
					if constexpr (_EstimateFocalLength)
					{
						Eigen::MatrixXd desc(3, 5 + cameraNumber * 3);
						desc.block<3, 5>(0, 0) << model.descriptor.block<3, 5>(0, 0);

						for (size_t ci = 0; ci < cameraNumber; ++ci)
						{
							const Eigen::Matrix3d &cameraRotation =
								generalizedCameraPoses[ci].block<3, 3>(0, 0);
							Eigen::Vector3d translation =
								cameraRotation * generalizedCameraPoses[ci].col(3) + model.descriptor.col(3);

							// The cross product matrix of the translation vector
							Eigen::Matrix3d cross_prod_t_dst_src;
							cross_prod_t_dst_src << 0, -translation(2), translation(1),
								translation(2), 0, -translation(0),
								-translation(1), translation(0), 0;

							Eigen::Matrix3d E =
								cross_prod_t_dst_src *
								cameraRotation * model.descriptor.block<3, 3>(0, 0);
							desc.block<3, 3>(0, 5 + ci * 3) = E;
						}
						model.descriptor = desc;

					}
					else
					{
						Eigen::MatrixXd desc(3, 4 + cameraNumber * 3);
						desc.block<3, 4>(0, 0) << model.descriptor.block<3, 4>(0, 0);

						for (size_t ci = 0; ci < cameraNumber; ++ci)
						{
							const Eigen::Matrix3d &cameraRotation =
								generalizedCameraPoses[ci].block<3, 3>(0, 0);
							Eigen::Vector3d translation =
								cameraRotation * generalizedCameraPoses[ci].col(3) + model.descriptor.col(3);

							// The cross product matrix of the translation vector
							Eigen::Matrix3d cross_prod_t_dst_src;
							cross_prod_t_dst_src << 0, -translation(2), translation(1),
								translation(2), 0, -translation(0),
								-translation(1), translation(0), 0;

							Eigen::Matrix3d E =
								cross_prod_t_dst_src *
								cameraRotation * model.descriptor.block<3, 3>(0, 0);
							desc.block<3, 3>(0, 4 + ci * 3) = E;
						}
						model.descriptor = desc;

						/*double sum = 0;
						for (int i = 0; i < sampleSize(); ++i)
							sum += residual(data.row(sample[i]), model);
						std::cout << " - " << sum / sampleSize() << std::endl;*/
					}

				}

				// Return true, if at least one model_ is kept
				return models->size() > 0;
			}

			// The squared sampson distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double sampsonDistance(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double squared_distance = squaredSampsonDistance(point_, descriptor_);
				return sqrt(squared_distance);
			}

			// The sampson distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double squaredSampsonDistance(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double* s = reinterpret_cast<double *>(point_.data);

				const size_t cameraIdx =
					static_cast<size_t>(s[8]);
				
				if constexpr (_EstimateFocalLength)
				{
					const Eigen::Matrix3d &E =
						descriptor_.block<3, 3>(0, 5 + 3 * cameraIdx);
					const double &focalLength = descriptor_(0, 4);

					const double
						x1 = s[0] / focalLength,
						y1 = s[1] / focalLength,
						x2 = s[9],
						y2 = s[10];


					const double
						&e11 = E(0, 0),
						&e12 = E(0, 1),
						&e13 = E(0, 2),
						&e21 = E(1, 0),
						&e22 = E(1, 1),
						&e23 = E(1, 2),
						&e31 = E(2, 0),
						&e32 = E(2, 1),
						&e33 = E(2, 2);

					double rxc = e11 * x2 + e21 * y2 + e31;
					double ryc = e12 * x2 + e22 * y2 + e32;
					double rwc = e13 * x2 + e23 * y2 + e33;
					double r = (x1 * rxc + y1 * ryc + rwc);
					double rx = e11 * x1 + e12 * y1 + e13;
					double ry = e21 * x1 + e22 * y1 + e23;

					return r * r /
						(rxc * rxc + ryc * ryc + rx * rx + ry * ry);
				}
				else
				{
					const double
						&x1 = s[0],
						&y1 = s[1],
						&x2 = s[9],
						&y2 = s[10];

					const Eigen::Matrix3d &E =
						descriptor_.block<3, 3>(0, 4 + 3 * cameraIdx);

					const double
						&e11 = E(0, 0),
						&e12 = E(0, 1),
						&e13 = E(0, 2),
						&e21 = E(1, 0),
						&e22 = E(1, 1),
						&e23 = E(1, 2),
						&e31 = E(2, 0),
						&e32 = E(2, 1),
						&e33 = E(2, 2);

					double rxc = e11 * x2 + e21 * y2 + e31;
					double ryc = e12 * x2 + e22 * y2 + e32;
					double rwc = e13 * x2 + e23 * y2 + e33;
					double r = (x1 * rxc + y1 * ryc + rwc);
					double rx = e11 * x1 + e12 * y1 + e13;
					double ry = e21 * x1 + e22 * y1 + e23;

					return r * r /
						(rxc * rxc + ryc * ryc + rx * rx + ry * ry);
				}
			}

			// The symmetric epipolar distance between a point_ correspondence and an essential matrix
			OLGA_INLINE double squaredSymmetricEpipolarDistance(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				const double* s = reinterpret_cast<double *>(point_.data);
				const double 
					&x1 = *s,
					&y1 = *(s + 1),
					&x2 = *(s + 2),
					&y2 = *(s + 3);

				const double 
					&e11 = descriptor_(0, 0),
					&e12 = descriptor_(0, 1),
					&e13 = descriptor_(0, 2),
					&e21 = descriptor_(1, 0),
					&e22 = descriptor_(1, 1),
					&e23 = descriptor_(1, 2),
					&e31 = descriptor_(2, 0),
					&e32 = descriptor_(2, 1),
					&e33 = descriptor_(2, 2);

				const double rxc = e11 * x2 + e21 * y2 + e31;
				const double ryc = e12 * x2 + e22 * y2 + e32;
				const double rwc = e13 * x2 + e23 * y2 + e33;
				const double r = (x1 * rxc + y1 * ryc + rwc);
				const double rx = e11 * x1 + e12 * y1 + e13;
				const double ry = e21 * x1 + e22 * y1 + e23;
				const double a = rxc * rxc + ryc * ryc;
				const double b = rx * rx + ry * ry;

				return r * r * (a + b) / (a * b);
			}

			// The squared residual function used for deciding which points are inliers
			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			// The squared residual function used for deciding which points are inliers
			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return squaredSampsonDistance(point_, descriptor_);
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				return sampsonDistance(point_, descriptor_);
			}

			// Validate the model by checking the number of inlier with symmetric epipolar distance
			// instead of Sampson distance. In general, Sampson distance is more accurate but less
			// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
			// every so-far-the-best model is checked if it has enough inlier with symmetric
			// epipolar distance as well. 
			bool isValidModel(Model& model_,
				const cv::Mat& data_,
				const std::vector<size_t> &inliers_,
				const size_t *minimal_sample_,
				const double threshold_,
				bool &model_updated_) const
			{
				return true;
			}
			
			// Estimating the model from a non-minimal sample
			OLGA_INLINE bool estimateModelNonminimal(
				const cv::Mat& data_,
				const size_t *sample_,
				const size_t &sample_number_,
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const
			{
				// Return of there are not enough points for the estimation
				if (sample_number_ < nonMinimalSampleSize())
					return false;

				std::vector<Model> models;
				if constexpr (needInitialModel())
					models.emplace_back(models_->at(0));

				/*double sum1 = 0;
				for (int i = 0; i < sample_number_; ++i)
					sum1 += residual(data_.row(sample_[i]), models.back().descriptor);*/

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
				for (auto &model : models)
				{
					Eigen::MatrixXd desc(3, offset + cameraNumber * 3);
					if constexpr (_EstimateFocalLength)
						desc.block<3, 5>(0, 0) << model.descriptor;
					else
						desc.block<3, 4>(0, 0) << model.descriptor;

					for (size_t ci = 0; ci < cameraNumber; ++ci)
					{
						const Eigen::Matrix3d &cameraRotation =
							generalizedCameraPoses[ci].block<3, 3>(0, 0);
						Eigen::Vector3d translation =
							cameraRotation * generalizedCameraPoses[ci].col(3) + model.descriptor.col(3);

						// The cross product matrix of the translation vector
						Eigen::Matrix3d cross_prod_t_dst_src;
						cross_prod_t_dst_src << 0, -translation(2), translation(1),
							translation(2), 0, -translation(0),
							-translation(1), translation(0), 0;

						Eigen::Matrix3d E =
							cross_prod_t_dst_src *
							cameraRotation * model.descriptor.block<3, 3>(0, 0);
						desc.block<3, 3>(0, offset + ci * 3) = E;
					}

					model.descriptor = desc;
					models_->emplace_back(model);

					/*double sum2 = 0;
					for (int i = 0; i < sample_number_; ++i)
						sum2 += residual(data_.row(sample_[i]), model.descriptor);

					printf("%f %f\n", sum1, sum2);*/
				}

				return true;
			}
			
			/************** Oriented epipolar constraints ******************/
			OLGA_INLINE void getEpipole(
				Eigen::Vector3d &epipole_, // The epipole 
				const Eigen::Matrix3d &essential_matrix_) const
			{
				constexpr double epsilon = 1.9984e-15;
				epipole_ = essential_matrix_.row(0).cross(essential_matrix_.row(2));

				for (auto i = 0; i < 3; i++)
					if ((epipole_(i) > epsilon) ||
						(epipole_(i) < -epsilon))
						return;
				epipole_ = essential_matrix_.row(1).cross(essential_matrix_.row(2));
			}

			OLGA_INLINE double getOrientationSignum(
				const Eigen::Matrix3d &essential_matrix_,
				const Eigen::Vector3d &epipole_,
				const cv::Mat &point_) const
			{
				double signum1 = essential_matrix_(0, 0) * point_.at<double>(2) + essential_matrix_(1, 0) * point_.at<double>(3) + essential_matrix_(2, 0),
					signum2 = epipole_(1) - epipole_(2) * point_.at<double>(1);
				return signum1 * signum2;
			}

			OLGA_INLINE int isOrientationValid(
				const Eigen::Matrix3d &essential_matrix_, // The fundamental matrix
				const cv::Mat &data_, // The data points
				const size_t *sample_, // The sample used for the estimation
				const size_t &sample_size_) const // The size of the sample
			{
				Eigen::Vector3d epipole; // The epipole in the second image
				getEpipole(epipole, essential_matrix_);

				double signum1, signum2;

				// The sample is null pointer, the method is applied to normalized data_
				if (sample_ == nullptr)
				{
					// Get the sign of orientation of the first point_ in the sample
					signum2 = getOrientationSignum(essential_matrix_, epipole, data_.row(0));
					for (size_t i = 1; i < sample_size_; i++)
					{
						// Get the sign of orientation of the i-th point_ in the sample
						signum1 = getOrientationSignum(essential_matrix_, epipole, data_.row(i));
						// The signs should be equal, otherwise, the fundamental matrix is invalid
						if (signum2 * signum1 < 0)
							return false;
					}
				}
				else
				{
					// Get the sign of orientation of the first point_ in the sample
					signum2 = getOrientationSignum(essential_matrix_, epipole, data_.row(sample_[0]));
					for (size_t i = 1; i < sample_size_; i++)
					{
						// Get the sign of orientation of the i-th point_ in the sample
						signum1 = getOrientationSignum(essential_matrix_, epipole, data_.row(sample_[i]));
						// The signs should be equal, otherwise, the fundamental matrix is invalid
						if (signum2 * signum1 < 0)
							return false;
					}
				}
				return true;
			}
		};
	}
}