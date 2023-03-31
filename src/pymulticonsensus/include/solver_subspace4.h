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
			class LinearSubspaceSolver : public SolverEngine
			{
			protected:
				bool isInitialized;

				OLGA_INLINE void gramSmithOrthogonalizationCV(
					const cv::Mat& points,
					cv::Mat& result) const;

				OLGA_INLINE void fitPlaneCV(
					const cv::Mat& pts,
					cv::Mat& plane) const;

				OLGA_INLINE void gramSmithOrthogonalization(
					const Eigen::MatrixXd& coefficients_,
					Eigen::MatrixXd& result_) const;

				OLGA_INLINE void fitPlane(
					const Eigen::MatrixXd& coefficients_,
					Eigen::MatrixXd& plane_) const;

			public:
				LinearSubspaceSolver() : isInitialized(false)
				{
				}

				~LinearSubspaceSolver()
				{
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
					return 4;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool LinearSubspaceSolver::estimateModel(
				const cv::Mat& data_,
				const size_t* sample_,
				size_t sample_number_,
				std::vector<Model>& models_,
				const double* weights_) const
			{
				Eigen::MatrixXd coefficients(sample_number_, data_.cols);

				double weight = 1.0;

				for (size_t i = 0; i < sample_number_; ++i)
				{
					const size_t idx =
						sample_ == nullptr ? i : sample_[i];

					if (weights_ != nullptr)
						weight = weights_[idx];

					for (size_t col = 0; col < data_.cols; ++col)
						coefficients(i, col) = weight * data_.at<double>(idx, col);
				}

				Eigen::Matrix<double, 5, 1> plane;
				if (sample_number_ == sampleSize())
				{
					const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients.transpose() * coefficients);
					if (lu.dimensionOfKernel() != 1)
						return false;
					plane = lu.kernel();
				}
				else
				{
					const Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(
						coefficients.transpose() * coefficients);
					const Eigen::MatrixXd& Q = qr.matrixQ();
					plane = Q.rightCols<1>();
				}

				plane /= plane.head<4>().norm();

				/*Eigen::MatrixXd coefficients(data_.cols, sample_number_);

				double weight = 1.0;

				for (size_t i = 0; i < sample_number_; ++i)
				{
					const size_t idx =
						sample_ == nullptr ? i : sample_[i];

					if (weights_ != nullptr)
						weight = weights_[idx];

					for (size_t col = 0; col < data_.cols; ++col)
						coefficients(col, i) = weight * data_.at<double>(idx, col);
				}

				/*Eigen::MatrixXd plane;
				fitPlane(coefficients, plane);

				if (plane.rows() == 0 || plane.cols() == 0)
					return false;

				if (plane.hasNaN())
					return false;*/

				models_.resize(models_.size() + 1);
				models_.back().descriptor = plane;
				return true;
			}


			OLGA_INLINE void LinearSubspaceSolver::gramSmithOrthogonalizationCV(
				const cv::Mat& points,
				cv::Mat& result) const
			{
				const int& K = points.rows;
				const int& D = 4; // points.cols;

				cv::Mat I = cv::Mat::eye(K, K, CV_64F);
				result = cv::Mat::zeros(K, D, CV_64F);

				cv::Mat col = points.col(0) / norm(points.col(0));
				col.copyTo(result.col(0));

				for (int i = 1; i < D; ++i)
				{
					cv::Mat newcol = (I - result * result.t()) * points.col(i);
					newcol = newcol / norm(newcol);
					newcol.copyTo(result.col(i));
					newcol.release();
				}

				col.release();
				I.release();
			}

			OLGA_INLINE void LinearSubspaceSolver::gramSmithOrthogonalization(
				const Eigen::MatrixXd& coefficients_,
				Eigen::MatrixXd& result_) const
			{
				const int& K = coefficients_.rows();
				const int& D = coefficients_.cols();

				Eigen::MatrixXd I = Eigen::MatrixXd::Identity(K, K);
				result_ = Eigen::MatrixXd::Zero(K, D);

				result_.col(0) << coefficients_.col(0) / coefficients_.col(0).norm();

				for (int i = 1; i < D; ++i)
				{
					result_.col(i) = (I - result_ * result_.transpose()) * coefficients_.col(i);
					result_.col(i) = result_.col(i) / result_.col(i).norm();
				}
			}

			OLGA_INLINE void LinearSubspaceSolver::fitPlaneCV(
				const cv::Mat& pts,
				cv::Mat& plane) const
			{
				cv::Mat result;
				gramSmithOrthogonalizationCV(pts, result);

				plane = cv::Mat::eye(pts.rows, pts.rows, CV_64F) - result * result.t();
				result.release();
			}

			OLGA_INLINE void LinearSubspaceSolver::fitPlane(
				const Eigen::MatrixXd& coefficients_,
				Eigen::MatrixXd& plane_) const
			{
				const int& K = coefficients_.rows();
				const int& D = coefficients_.cols();

				if (D == sampleSize())
				{
					Eigen::MatrixXd result;
					gramSmithOrthogonalization(coefficients_, result);

					plane_ = Eigen::MatrixXd::Identity(K, K) - result * result.transpose();
				}
				else
				{
					/*Eigen::MatrixXd result;
					gramSmithOrthogonalization(coefficients_ * coefficients_.transpose(), result);

					plane_ = Eigen::MatrixXd::Identity(K, K) - result * result.transpose();

					Eigen::JacobiSVD<Eigen::MatrixXd> es(coefficients_ * coefficients_.transpose(),
						Eigen::ComputeFullU | Eigen::ComputeThinV);
					plane_ = es.matrixU().real();

					std::cout << plane_ << std::endl;
					std::cout << std::endl;*/
					
					/*
					Eigen::JacobiSVD<Eigen::MatrixXd> es(coefficients_ * coefficients_.transpose(),
						Eigen::ComputeFullU | Eigen::ComputeThinV);
					plane_ = es.matrixU().real();*/
				}
			}

			/*bool EstimateModelNonminimal(const Mat& data,
				const int* sample,
				int sample_number,
				vector<Model>* models) const
			{
				return false;

				Model model;

				if (sample_number < 4)
					return false;

				Mat pts(sample_number, data.cols, CV_32F);

				for (int i = 0; i < sample_number; ++i)
				{
					int idx = sample[i];
					data.row(idx).copyTo(pts.row(i));
				}
				pts = pts.t();

				SVD svd(pts);
				pts.release();

				Mat plane = svd.u.t();

				//B = U(:, 1 : d);
				//Borth = U(:, d + 1 : end);

				//cout << plane << endl;

				model.descriptor = plane;
				model.estimator = this;
				models->push_back(model);

				/*Mat A(sample_number, 5, CV_64F);
				int idx;
				Mat mass_point = Mat::zeros(1, 4, CV_32F);
				for (int i = 0; i < sample_number; ++i)
				{
					idx = sample[i];
					mass_point.at<float>(0) += data.at<float>(idx, 0);
					mass_point.at<float>(1) += data.at<float>(idx, 1);
					mass_point.at<float>(2) += data.at<float>(idx, 2);
					mass_point.at<float>(3) += data.at<float>(idx, 3);

					A.at<double>(i, 0) = (double)data.at<float>(idx, 0);
					A.at<double>(i, 1) = (double)data.at<float>(idx, 1);
					A.at<double>(i, 2) = (double)data.at<float>(idx, 2);
					A.at<double>(i, 3) = (double)data.at<float>(idx, 3);
					A.at<double>(i, 4) = 1;
				}
				mass_point = mass_point * (1.0 / sample_number);

				Mat AtA = A.t() * A;
				Mat eValues, eVectors;
				eigen(AtA, eValues, eVectors);

				Mat subspace = eVectors.row(4);
				subspace.convertTo(subspace, CV_32F);

				float length = sqrt(subspace.at<float>(0) * subspace.at<float>(0) +
					subspace.at<float>(1) * subspace.at<float>(1) +
					subspace.at<float>(2) * subspace.at<float>(2) +
					subspace.at<float>(3) * subspace.at<float>(3));
				subspace.at<float>(0) /= length;
				subspace.at<float>(1) /= length;
				subspace.at<float>(2) /= length;
				subspace.at<float>(3) /= length;

				subspace.at<float>(4) = -(subspace.at<float>(0) * mass_point.at<float>(0) +
					subspace.at<float>(1) * mass_point.at<float>(1) +
					subspace.at<float>(2) * mass_point.at<float>(2) +
					subspace.at<float>(3) * mass_point.at<float>(3));

				model.descriptor = subspace.t();
				model.estimator = this;
				models->push_back(model);

				if (model.descriptor.rows == 0 || model.descriptor.cols == 0)
					return false;
				return true;
			}*/

			/*bool EstimateModel(const Mat& data,
				const int* sample,
				vector<Model>* models) const
			{

			}*/

		}
	}
}