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

#include <vector>
#include <unordered_set>
#include "gamma_values.cpp"

namespace clustering
{
	namespace losses
	{
		template <typename _Type = double>
		class Loss
		{
		public:
			virtual inline _Type get(
				const _Type& residual_,
				const _Type& threshold_) const = 0;

			virtual inline void get(
				const std::vector<_Type>& residuals_,
				const _Type& threshold_,
				std::vector<_Type>& weights_) const = 0;
		};

		template <typename _Type = double>
		class RANSACLoss : public Loss<_Type>
		{
		public:
			inline _Type get(
				const _Type& residual_,
				const _Type& threshold_) const
			{
				return residual_ <= threshold_;
			}

			inline void get(
				const std::vector<_Type>& residuals_,
				const _Type& threshold_,
				std::vector<_Type>& weights_) const
			{
				weights_.reserve(residuals_.size());
				for (const auto& residual : residuals_)
					weights_.emplace_back(residual <= threshold_);
			}
		};

		template <typename _Type,
			typename _Estimator,
			size_t _DimensionNumber>
		class MAGSACLoss : public Loss<_Type>
		{
		public:
			inline _Type get(
				const _Type& residual_,
				const _Type& threshold_) const
			{
				if constexpr (_DimensionNumber != 2 && _DimensionNumber != 4)
				{
					fprintf(stderr, "Incorrect dimension number (%d) for the MAGSAC loss function. It supposed to be either 2 or 4", _DimensionNumber);
					return 0;
				}

				if (residual_ > threshold_)
					return 0;

				// The degrees of freedom of the data from which the model is estimated.
				// E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
				constexpr size_t degrees_of_freedom = _DimensionNumber;
				// A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
				constexpr double k = 
					_DimensionNumber == 2 ? 
					3.03 : 3.64;
				// A multiplier to convert residual values to sigmas
				constexpr double threshold_to_sigma_multiplier = 1.0 / k;
				// Calculating k^2 / 2 which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				constexpr double squared_k_per_2 = k * k / 2.0;
				// Calculating (DoF - 1) / 2 which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
				// TODO: check
				constexpr double C = 0.25;
				// The size of a minimal sample used for the estimation
				constexpr size_t sample_size = _Estimator::sampleSize();
				// Calculating 2^(DoF - 1) which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				static const double two_ad_dof = std::pow(2.0, dof_minus_one_per_two);
				// Calculating C * 2^(DoF - 1) which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				static const double C_times_two_ad_dof = C * two_ad_dof;
				// Calculating the gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				static const double gamma_value = tgamma(dof_minus_one_per_two);
				// Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
				constexpr double gamma_k = 0.0036572608340910764;
				// Calculating the lower incomplete gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				static const double gamma_difference = gamma_value - gamma_k;
				// Calculate 2 * \sigma_{max}^2 a priori
				const double squared_sigma_max_2 = threshold_ * threshold_ * 2.0;
				// Divide C * 2^(DoF - 1) by \sigma_{max} a priori
				const double one_over_sigma = C_times_two_ad_dof / threshold_;
				// Calculate the weight of a point with 0 residual (i.e., fitting perfectly) a priori
				const double weight_zero = one_over_sigma * gamma_difference;

				// The weight
				double weight = 0.0;
				// If the residual is ~0, the point fits perfectly and it is handled differently
				if (residual_ < std::numeric_limits<double>::epsilon())
					weight = weight_zero;
				else
				{
					// Calculate the squared residual
					const double squared_residual = residual_ * residual_;
					// Get the position of the gamma value in the lookup table
					size_t x = round(precision_of_stored_gammas * squared_residual / squared_sigma_max_2);

					// If the sought gamma value is not stored in the lookup, return the closest element
					if (stored_gamma_number < x)
						x = stored_gamma_number;

					// Calculate the weight of the point
					weight = one_over_sigma * (stored_gamma_values[x] - gamma_k);
				}
			
				return weight / weight_zero;
			}


			inline void get(
				const std::vector<_Type>& residuals_,
				const _Type& threshold_,
				std::vector<_Type>& weights_) const
			{
				// The degrees of freedom of the data from which the model is estimated.
				// E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
				constexpr size_t degrees_of_freedom = 4;
				// A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
				constexpr double k = 3.64;
				// A multiplier to convert residual values to sigmas
				constexpr double threshold_to_sigma_multiplier = 1.0 / k;
				// Calculating k^2 / 2 which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				constexpr double squared_k_per_2 = k * k / 2.0;
				// Calculating (DoF - 1) / 2 which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
				// TODO: check
				constexpr double C = 0.25;
				// The size of a minimal sample used for the estimation
				constexpr size_t sample_size = _Estimator::sampleSize();
				// Calculating 2^(DoF - 1) which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				static const double two_ad_dof = std::pow(2.0, dof_minus_one_per_two);
				// Calculating C * 2^(DoF - 1) which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				static const double C_times_two_ad_dof = C * two_ad_dof;
				// Calculating the gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				static const double gamma_value = tgamma(dof_minus_one_per_two);
				// Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
				constexpr double gamma_k = 0.0036572608340910764;
				// Calculating the lower incomplete gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
				// due to being constant, it is better to calculate it a priori.
				static const double gamma_difference = gamma_value - gamma_k;
				// Calculate 2 * \sigma_{max}^2 a priori
				const double squared_sigma_max_2 = threshold_ * threshold_ * 2.0;
				// Divide C * 2^(DoF - 1) by \sigma_{max} a priori
				const double one_over_sigma = C_times_two_ad_dof / threshold_;
				// Calculate the weight of a point with 0 residual (i.e., fitting perfectly) a priori
				const double weight_zero = one_over_sigma * gamma_difference;

				// The weight
				weights_.reserve(residuals_.size());
				for (const auto& residual : residuals_)
				{
					double weight = 0.0;

					if (residual <= threshold_)
					{
						// If the residual is ~0, the point fits perfectly and it is handled differently
						if (residual < std::numeric_limits<double>::epsilon())
							weight = weight_zero;
						else
						{
							// Calculate the squared residual
							const double squared_residual = residual * residual;
							// Get the position of the gamma value in the lookup table
							size_t x = round(precision_of_stored_gammas * squared_residual / squared_sigma_max_2);

							// If the sought gamma value is not stored in the lookup, return the closest element
							if (stored_gamma_number < x)
								x = stored_gamma_number;

							// Calculate the weight of the point
							weight = one_over_sigma * (stored_gamma_values[x] - gamma_k);
						}
					}

					weights_.emplace_back(weight);
				}
			}
		};

		template <typename _Type = double>
		class TukeyBisquareLoss : public Loss<_Type>
		{
		public:
			inline _Type get(
				const _Type& residual_,
				const _Type& threshold_) const
			{
				if (residual_ > threshold_)
					return 0.0;
				const _Type division = residual_ / threshold_;
				return 1.0 - division * division;
			}

			inline void get(
				const std::vector<_Type>& residuals_,
				const _Type& threshold_,
				std::vector<_Type>& weights_) const
			{
				weights_.reserve(residuals_.size());
				for (const auto& residual : residuals_)
					weights_.emplace_back(get(residual, threshold_));
			}
		};

		template <typename _Type = double>
		class HuberLoss : public Loss<_Type>
		{
		public:
			inline _Type get(
				const _Type& residual_,
				const _Type& threshold_) const
			{
				if (residual_ < threshold_)
					return 1.0;
				return residual_ / threshold_;
			}

			inline void get(
				const std::vector<_Type>& residuals_,
				const _Type& threshold_,
				std::vector<_Type>& weights_) const
			{
				weights_.reserve(residuals_.size());
				for (const auto& residual : residuals_)
					weights_.emplace_back(get(residual, threshold_));
			}
		};

		template <typename _Type = double>
		class RedescendingHuberLoss : public Loss<_Type>
		{
		public:
			inline _Type get(
				const _Type& residual_,
				const _Type& threshold_) const
			{
				if (residual_ < threshold_)
					return 1.0;
				if (residual_ > 3 * threshold_)
					return 0.0;
				return residual_ / threshold_;
			}

			inline void get(
				const std::vector<_Type>& residuals_,
				const _Type& threshold_,
				std::vector<_Type>& weights_) const
			{
				weights_.reserve(residuals_.size());
				for (const auto& residual : residuals_)
					weights_.emplace_back(get(residual, threshold_));
			}
		};

		template <typename _Type = double>
		class WelschLoss : public Loss<_Type>
		{
		public:
			inline _Type get(
				const _Type& residual_,
				const _Type& threshold_) const
			{
				const _Type division = residual_ / threshold_;
				return std::exp(-(division * division));
			}

			inline void get(
				const std::vector<_Type>& residuals_,
				const _Type& threshold_,
				std::vector<_Type>& weights_) const
			{
				weights_.reserve(residuals_.size());
				for (const auto& residual : residuals_)
					weights_.emplace_back(get(residual, threshold_));
			}
		};
	}
}