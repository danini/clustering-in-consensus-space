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
#include "sturm.h"
#include <Eigen/StdVector>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class GeneralizedHomographyThreeTwoPointSolver : public SolverEngine
			{
			public:
				GeneralizedHomographyThreeTwoPointSolver()
				{
				}

				GeneralizedHomographyThreeTwoPointSolver(
					const std::vector<Eigen::Matrix<double, 3, 4>> &generalizedCameraPoses_,
					const size_t cameraNumber_)
				{
				}

				~GeneralizedHomographyThreeTwoPointSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				static constexpr size_t maximumSolutions()
				{
					return 3;
				}
				
				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 5;
				}

				static constexpr bool needInitialModel()
				{
					return false;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

				OLGA_INLINE Eigen::Matrix<double, 8, 10> parameterize_HN(const Eigen::Matrix<double, 3, 5>& q,
					const Eigen::Matrix<double, 3, 5>& p,
					const Eigen::Matrix<double, 3, 5>& c) const;

				OLGA_INLINE void transform(Eigen::Matrix<double, 3, 5>& q_d, Eigen::Matrix<double, 3, 5>& p_d, Eigen::Matrix<double, 3, 5>& c_d, Eigen::Matrix3d& Rtransform1, Eigen::Matrix3d& Rtransform2, Eigen::Vector3d& shift) const;
				OLGA_INLINE Eigen::Matrix<double, 4, 1> get_coeffs(const double* data) const;

				template <int m, int max_n>
				using Md = Eigen::Matrix<double, m, Eigen::Dynamic, 0, m, max_n>;

				template <class T>
				OLGA_INLINE constexpr T sign(T x) const;

				OLGA_INLINE void extract_homographies(
					const Md<9, 6>& vecs,
					const Eigen::Matrix3d& Rtransform,
					const Eigen::Matrix3d& Rtransform2,
					const Eigen::Vector3d& shift,
					std::vector<Eigen::Matrix3d>* Hs,
					std::vector<Eigen::Vector3d>* Nss) const;
			};

			OLGA_INLINE Eigen::Matrix<double, 8, 10> GeneralizedHomographyThreeTwoPointSolver::parameterize_HN(const Eigen::Matrix<double, 3, 5>& q, 
				const Eigen::Matrix<double, 3, 5>& p, 
				const Eigen::Matrix<double, 3, 5>& c) const
			{
				Eigen::Matrix<double, 8, 10> M;
				M << 0, -p(2, 1) * q(0, 1), 0, -p(2, 1) * q(1, 1), p(1, 1)* q(1, 1), p(1, 1)* q(2, 1), c(0, 1)* q(0, 1), c(0, 1)* q(1, 1), c(0, 1) * q(2, 1), p(1, 1) * q(0, 1),
					p(2, 1)* q(0, 1), 0, p(2, 1)* q(1, 1), 0, -p(0, 1) * q(1, 1), -p(0, 1) * q(2, 1), c(1, 1)* q(0, 1), c(1, 1)* q(1, 1), c(1, 1) * q(2, 1), -p(0, 1)* q(0, 1),
					0, -p(2, 2) * q(0, 2), 0, -p(2, 2) * q(1, 2), p(1, 2)* q(1, 2), p(1, 2)* q(2, 2), c(0, 2)* q(0, 2), c(0, 2)* q(1, 2), c(0, 2) * q(2, 2), p(1, 2) * q(0, 2),
					p(2, 2)* q(0, 2), 0, p(2, 2)* q(1, 2), 0, -p(0, 2) * q(1, 2), -p(0, 2) * q(2, 2), c(1, 2)* q(0, 2), c(1, 2)* q(1, 2), c(1, 2) * q(2, 2), -p(0, 2)* q(0, 2),
					0, -p(2, 3) * q(0, 3), 0, -p(2, 3) * q(1, 3), p(1, 3)* q(1, 3), p(1, 3)* q(2, 3), c(0, 3)* q(0, 3), c(0, 3)* q(1, 3), c(0, 3) * q(2, 3), p(1, 3) * q(0, 3),
					p(2, 3)* q(0, 3), 0, p(2, 3)* q(1, 3), 0, -p(0, 3) * q(1, 3), -p(0, 3) * q(2, 3), c(1, 3)* q(0, 3), c(1, 3)* q(1, 3), c(1, 3) * q(2, 3), -p(0, 3)* q(0, 3),
					0, -p(2, 4) * q(0, 4), 0, -p(2, 4) * q(1, 4), p(1, 4)* q(1, 4), p(1, 4)* q(2, 4), c(0, 4)* q(0, 4), c(0, 4)* q(1, 4), c(0, 4) * q(2, 4), p(1, 4) * q(0, 4),
					p(2, 4)* q(0, 4), 0, p(2, 4)* q(1, 4), 0, -p(0, 4) * q(1, 4), -p(0, 4) * q(2, 4), c(1, 4)* q(0, 4), c(1, 4)* q(1, 4), c(1, 4) * q(2, 4), -p(0, 4)* q(0, 4);

				return M;
			}

			OLGA_INLINE Eigen::Matrix<double, 4, 1> GeneralizedHomographyThreeTwoPointSolver::get_coeffs(const double* data) const
			{
				Eigen::Matrix<double, 4, 1> C;

				C(0) = data[0] * pow(data[1], 2) * data[8] * data[9] - pow(data[1], 3) * data[9];
				C(1) = 2 * data[0] * data[1] * data[3] * data[8] * data[9] - data[0] * data[1] * pow(data[6], 2) - data[0] * data[1] * pow(data[7], 2) - data[0] * data[1] * pow(data[8], 2) + data[0] * data[1] * pow(data[9], 2) + pow(data[1], 2) * data[2] * data[8] * data[9] - 3 * pow(data[1], 2) * data[3] * data[9] + pow(data[1], 2) * data[4] * data[6] + pow(data[1], 2) * data[5] * data[7] + pow(data[1], 2) * data[8];
				C(2) = data[0] * pow(data[3], 2) * data[8] * data[9] - data[0] * data[3] * pow(data[6], 2) - data[0] * data[3] * pow(data[7], 2) - data[0] * data[3] * pow(data[8], 2) + data[0] * data[3] * pow(data[9], 2) - data[0] * data[8] * data[9] + 2 * data[1] * data[2] * data[3] * data[8] * data[9] - data[1] * data[2] * pow(data[6], 2) - data[1] * data[2] * pow(data[7], 2) - data[1] * data[2] * pow(data[8], 2) + data[1] * data[2] * pow(data[9], 2) - 3 * data[1] * pow(data[3], 2) * data[9] + 2 * data[1] * data[3] * data[4] * data[6] + 2 * data[1] * data[3] * data[5] * data[7] + 2 * data[1] * data[3] * data[8] - data[1] * data[9];
				C(3) = data[2] * pow(data[3], 2) * data[8] * data[9] - data[2] * data[3] * pow(data[6], 2) - data[2] * data[3] * pow(data[7], 2) - data[2] * data[3] * pow(data[8], 2) + data[2] * data[3] * pow(data[9], 2) - data[2] * data[8] * data[9] - pow(data[3], 3) * data[9] + pow(data[3], 2) * data[4] * data[6] + pow(data[3], 2) * data[5] * data[7] + pow(data[3], 2) * data[8] - data[3] * data[9] + data[4] * data[6] + data[5] * data[7] + data[8];

				return C;
			}

			OLGA_INLINE void GeneralizedHomographyThreeTwoPointSolver::transform(
				Eigen::Matrix<double, 3, 5>& q_d,
				Eigen::Matrix<double, 3, 5>& p_d, 
				Eigen::Matrix<double, 3, 5>& c_d, 
				Eigen::Matrix3d& Rtransform1,
				Eigen::Matrix3d& Rtransform2,
				Eigen::Vector3d& shift) const
			{
				Eigen::Vector3d b(0, 0, 1);
				Eigen::Matrix3d id3;
				id3.setIdentity();

				// Coordinate transform Q, pin hole camera
				Eigen::Vector3d a = q_d.block<3, 1>(0, 0).normalized();
				Eigen::Vector3d v = a.cross(b);
				double c = a.dot(b);

				Eigen::Matrix3d vx;
				vx << 0, -v(2), v(1),
					v(2), 0, -v(0),
					-v(1), v(0), 0;

				Rtransform1 = id3 + vx + vx * vx / (1 + c);

				q_d = (Rtransform1 * q_d).colwise().hnormalized().colwise().homogeneous();	// q_d = Rtransform1 * q_d; q_d = q_d ./ q_d(3,:);

				q_d.colwise().normalize();	// q_d = q_d./sqrt(sum(q_d.^2));

				// Coordinate transform P, gen. camera
				shift = c_d.block<3, 1>(0, 0);
				a = p_d.block<3, 1>(0, 0).normalized();
				v = a.cross(b);
				c = a.dot(b);

				vx << 0, -v(2), v(1),
					v(2), 0, -v(0),
					-v(1), v(0), 0;

				Rtransform2 = id3 + vx + vx * vx / (1 + c);

				c_d = Rtransform2 * (c_d.colwise() - shift);
				p_d = Rtransform2 * p_d;

				p_d.colwise().normalize();	// p_d = p_d./sqrt(sum(p_d.^2));   mozno spolu rovno s tym vyssie

				for (int i = 0; i < 5; ++i)
					c_d.col(i) = p_d.col(i).cross(c_d.col(i));	// might be faster to do it in place and *(-1)
			}
			
			template <class T>
			OLGA_INLINE constexpr T GeneralizedHomographyThreeTwoPointSolver::sign(T x) const
			{
				return x == T(0) ? T(0) : x / abs(x);
			}

			OLGA_INLINE void GeneralizedHomographyThreeTwoPointSolver::extract_homographies(
				const Md<9, 6>& vecs,
				const Eigen::Matrix3d& Rtransform,
				const Eigen::Matrix3d& Rtransform2,
				const Eigen::Vector3d& shift,
				std::vector<Eigen::Matrix3d>* Hs, 
				std::vector<Eigen::Vector3d>* Nss) const
			{
				for (int i = 0; i < vecs.cols(); ++i)
				{
					Eigen::Vector3d N(vecs(6, i), vecs(7, i), vecs(8, i));

					Eigen::Matrix3d H;
					H << vecs(0, i), vecs(2, i), 0,
						vecs(1, i), vecs(3, i), 0,
						1, vecs(4, i), vecs(5, i);


					// Using the elimination ideal(the simplest polynomial before we eliminated h31 from it) to extract h31 from it.
					double scale = sqrt((vecs(7, i) * vecs(7, i) + vecs(8, i) * vecs(8, i)) / (vecs(2, i) * vecs(2, i) * vecs(8, i) * vecs(8, i) + vecs(3, i) * vecs(3, i) * vecs(8, i) * vecs(8, i) + vecs(4, i) * vecs(4, i) * vecs(8, i) * vecs(8, i) - (double)2 * vecs(4, i) * vecs(5, i) * vecs(7, i) * vecs(8, i) + vecs(5, i) * vecs(5, i) * vecs(7, i) * vecs(7, i)));


					// We need to ensure that the plane vector has a sign such that the
					// depth of the point is + ve.
					// If the 3D point is X = alpha * q then,
					// alpha * N ^ T * q + 1 = 0
					// which means that N ^ T * q must be - ve so that alpha is + ve
					scale *= -sign(scale * vecs(8, i));
					N *= scale;
					H *= scale;


					//The reverse coordinate transform for
					// homographyand the plane vector
					Nss->push_back(Rtransform.transpose() * N);
					Hs->push_back(Rtransform2.transpose() * H);
					Hs->back() *= Rtransform;
					Hs->back() -= shift * Nss->back().transpose();
				}
			}

			OLGA_INLINE bool GeneralizedHomographyThreeTwoPointSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				const size_t kColumns = data_.cols;
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				Eigen::Matrix<double, 3, 5> q, p, c;

				for (size_t i = 0; i < sample_number_; ++i)
				{
					const size_t idx =
						sample_ == nullptr ? i : sample_[i];

					const double *point_ptr =
						data_ptr + idx * kColumns;

					q(0, i) = point_ptr[0];
					q(1, i) = point_ptr[1];
					q(2, i) = 1;
					p(0, i) = point_ptr[2];
					p(1, i) = point_ptr[3];
					p(2, i) = point_ptr[4];
					c(0, i) = point_ptr[5];
					c(1, i) = point_ptr[6];
					c(2, i) = point_ptr[7];
				}

				Eigen::Matrix3d Rtransform1; Eigen::Matrix3d Rtransform2; Eigen::Vector3d shift;

				transform(q, p, c, Rtransform1, Rtransform2, shift);

				// Computing the coefficients
				// Estimate the scaled version of the H matrix
				auto M = parameterize_HN(q, p, c);

				Eigen::Matrix<double, 6, 1> h = (-M.block<8, 8>(0, 0)).lu().solve(M.block<8, 1>(0, 9)).block<6, 1>(0, 0);

				// estimate the scaled version of the plane vector
				Eigen::Matrix<double, 2, 3> b;	//useless
				b << M.block<1, 3>(5, 6),
					M.block<1, 3>(7, 6);

				Eigen::Vector2d A;
				A << -M.block<1, 6>(5, 0) * h - M(5, 9),
					-M.block<1, 6>(7, 0) * h - M(7, 9);

				Eigen::Vector2d c1 = b.block<2, 2>(0, 0).lu().solve(A);

				Eigen::Vector2d c2 = (-b.block<2, 2>(0, 0)).lu().solve(b.col(2));

				double data[10];
				std::copy(c1.data(), c1.data() + 2, data);
				std::copy(c2.data(), c2.data() + 2, data + 2);
				std::copy(h.data(), h.data() + 6, data + 4);

				auto coef = get_coeffs(data);	//coef should be just array

				double roots[3] = { 0 };

				int n_roots = pose_lib::sturm::bisect_sturm<3>(coef.data(), roots);

				Eigen::Matrix<double, 9, Eigen::Dynamic, 0, 9, 3> vecs(9, n_roots);

				for (size_t i = 0; i < n_roots; ++i)
				{
					vecs.block<6, 1>(0, i) = h;
					vecs(6, i) = data[0] + data[2] * roots[i];
					vecs(7, i) = data[1] + data[3] * roots[i];
					vecs(8, i) = roots[i];
				}

				std::vector<Eigen::Matrix3d> Hs;
				std::vector<Eigen::Vector3d> Nss;

				extract_homographies(vecs, Rtransform1, Rtransform2, shift, &Hs, &Nss);

				for (size_t i = 0; i < Hs.size(); ++i)
				{
					Homography model;
					model.descriptor = Eigen::MatrixXd(3, 4);
					model.descriptor << Hs[i], Nss[i];
					models_.emplace_back(model);
				}
				return true;
			}
		}
	}
}