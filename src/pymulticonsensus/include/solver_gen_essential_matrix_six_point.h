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
#include "fundamental_estimator.h"
#include "ComputerDeterminanHelper.h"
#include "Eigen/Core"
#include "Eigen/LU"
#include "Eigen/Eigenvalues"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the lookup table
			int poly84_multi_1[] = { 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112 };
			int poly84_multi_v1[] = { -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 27, 28, -1, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 47, 48, 49, 51, 52, 54, 57, 58, 59, 60, 61, 63, 64, 65, 66, 68, 69, 70, 72, 73, 75, 78, 79, 80, 81, 83, 84, 85, 87, 88, 90, 93, 94, 95, 97, 98, 100, 103, 104, 106, 109 };
			int poly84_multi_v2[] = { -1, 0, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 28, -1, -1, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 55, 58, 59, 60, 61, 62, 64, 65, 66, 67, 69, 70, 71, 73, 74, 76, 79, 80, 81, 82, 84, 85, 86, 88, 89, 91, 94, 95, 96, 98, 99, 101, 104, 105, 107, 110 };
			int poly84_multi_v3[] = { 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, -1, -1, -1, -1, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 97, 98, 99, 100, 101, 102, 106, 107, 108, 111 };

			int poly35_multi_1[] = { 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112 };
			int poly35_multi_v1[] = { 57, 58, 59, 60, 61, 63, 64, 65, 66, 68, 69, 70, 72, 73, 75, 78, 79, 80, 81, 83, 84, 85, 87, 88, 90, 93, 94, 95, 97, 98, 100, 103, 104, 106, 109 };
			int poly35_multi_v2[] = { 58, 59, 60, 61, 62, 64, 65, 66, 67, 69, 70, 71, 73, 74, 76, 79, 80, 81, 82, 84, 85, 86, 88, 89, 91, 94, 95, 96, 98, 99, 101, 104, 105, 107, 110 };
			int poly35_multi_v3[] = { 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 97, 98, 99, 100, 101, 102, 106, 107, 108, 111 };
			int poly35_multi_v1Square[] = { 29, 30, 31, 32, 33, 36, 37, 38, 39, 42, 43, 44, 47, 48, 51, 57, 58, 59, 60, 63, 64, 65, 68, 69, 72, 78, 79, 80, 83, 84, 87, 93, 94, 97, 103 };
			int poly35_multi_v2Square[] = { 31, 32, 33, 34, 35, 38, 39, 40, 41, 44, 45, 46, 49, 50, 53, 59, 60, 61, 62, 65, 66, 67, 70, 71, 74, 80, 81, 82, 85, 86, 89, 95, 96, 99, 105 };
			int poly35_multi_v3Square[] = { 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 87, 88, 89, 90, 91, 92, 100, 101, 102, 108 };
			
			bool selectedPolynomials[] = { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class GeneralizedEssentialMatrixSixPointSolver : public SolverEngine
			{
			public:
				GeneralizedEssentialMatrixSixPointSolver()
				{
				}

				~GeneralizedEssentialMatrixSixPointSolver()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 6;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				static constexpr size_t maximumSolutions()
				{
					return 40;
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
				typedef Eigen::Matrix<double, 84, 14> Matrix_84x14;
				typedef Eigen::Matrix<double, 14, 14> Matrix_14x14;
				//---------------------------------------------------------------
				typedef Eigen::Matrix<double, 35, 4> Matrix_35x4;
				typedef Eigen::Matrix<double, 4, 4> Matrix_4x4;
				//---------------------------------------------------------------

				void ComputeFMatrix3x3(const double *P1, const double *P2, int reference_pts_id,
					double F_part0[5][10], double F_part1[5][10], double F_part2[5][10]) const;
				void GenerateCoefficient3x3FromF(const double F_part0[5][10], const double F_part1[5][10],
					const double F_part2[5][10], int rowId[3], const int &poly_coefficient_size, double poly_coefficient[84])  const;
				void GenerateCoefficient3x3(const double *P1, const double *P2, Matrix_84x14 &poly_coefficient_84x14) const;
				void ComputeFMatrix2x2(const double *P1, const double *P2, int reference_pts_id, double F_part0[3][10], double F_part1[3][10]) const;
				void GenerateCoefficient2x2FromF(const double F_part0[3][10], const double F_part1[3][10],
					const int rowId[2], const int &poly_coefficient_size, double polyCoefficient[35])  const;
				void GenerateCoefficient2x2(const double *P1, const double *P2, Matrix_35x4 &poly_coefficient_35x4) const;
				void ComputeOnePolynomial_113(double *input, int numOfMonomials, int *poly84_multi, double *output) const;
				void ConstructActionMatrix(const Eigen::MatrixXd &finalmatrix, Eigen::MatrixXd &action_matrix) const;
				bool IsEigenVectorASolution(const Eigen::Matrix<double, 40, 1> &eigen_vector) const;
				int FindAllSolutionsFromActionMatrix(const Eigen::Matrix<double, 3, 6> &q1_matrix,
					const Eigen::Matrix<double, 3, 6> &q1_prime_matrix,
					const Eigen::Matrix<double, 3, 6> &q2_matrix,
					const Eigen::MatrixXd &action_matrix, double quaternion[3 * 40], double t[3 * 40], double R[9 * 40]) const;
				void ConstructMatrix113x73(double *poly_coefficient_84x14, double *poly_coefficient_35x4, Eigen::MatrixXd &finalmatrix) const;
				void ComputePluckerCoordinate(const double ray_directions1[3 * 6], const double centers1[3 * 6], const double ray_directions2[3 * 6], Eigen::Matrix<double, 3, 6> &q1_matrix, Eigen::Matrix<double, 3, 6> &q1_prime_matrix, Eigen::Matrix<double, 3, 6> &q2_matrix)  const;
				int RelativePoseGeneralizedCalibratedPinhole(
					const double ray_directions1[3 * 6],
					const double centers1[3 * 6],
					const double ray_directions2[3 * 6],
					double quaternion[3 * 40],
					double t[3 * 40],
					double R[9 * 40]) const;
			};


			void GeneralizedEssentialMatrixSixPointSolver::ComputeFMatrix3x3(const double *P1, const double *P2, int reference_pts_id,
				double F_part0[5][10], double F_part1[5][10], double F_part2[5][10]) const
			{
				// F_part0 saves the coefficients of {v1,v2,v3} of the first column. F_part1 and F_part2 are for the second and third columns.
				int &j = reference_pts_id;
				int Fmatrix_row_index = 0;
				for (int i = 0; i < 6; ++i) {	// compute each row of the 5x3 matrix F
					if (j != i) {
						double in1[] = { P1[i * 6],P1[i * 6 + 1],P1[i * 6 + 2],P1[i * 6 + 3],P1[i * 6 + 4],P1[i * 6 + 5], P1[j * 6],P1[j * 6 + 1],P1[j * 6 + 2],P1[j * 6 + 3],P1[j * 6 + 4],P1[j * 6 + 5] };
						double in2[] = { P2[i * 3],P2[i * 3 + 1],P2[i * 3 + 2], P2[j * 3],P2[j * 3 + 1],P2[j * 3 + 2] };
						Generate_F3x3_matrix(in1, in2, F_part0[Fmatrix_row_index], F_part1[Fmatrix_row_index], F_part2[Fmatrix_row_index]);
						++Fmatrix_row_index;
					}
				}
			}

			void GeneralizedEssentialMatrixSixPointSolver::GenerateCoefficient3x3FromF(const double F_part0[5][10], const double F_part1[5][10],
				const double F_part2[5][10], int rowId[3], const int &poly_coefficient_size, double poly_coefficient[84])  const
			{
				// compute the determinant of a 3x3 matrix. The 3x3 matrix in the 5x3 matrix is determined using the variable 'rowId'
				int determinantIndex[6][3] = { {rowId[0], rowId[1], rowId[2]}, {rowId[0], rowId[2], rowId[1]},
					{rowId[1], rowId[2], rowId[0]}, {rowId[1], rowId[0], rowId[2]}, {rowId[2], rowId[0], rowId[1]}, {rowId[2], rowId[1], rowId[0]} };
				double sign = 1.0;

				Eigen::Matrix<double, 84, 1> poly_coefficient_temp;
				for (int i = 0; i < 6; ++i) {
					int index0 = determinantIndex[i][0];
					int index1 = determinantIndex[i][1];
					int index2 = determinantIndex[i][2];
					det3x3_helper(F_part0[index0], F_part1[index1], F_part2[index2], poly_coefficient_temp.data());
					for (int j = 0; j < poly_coefficient_size; j++) {
						poly_coefficient[j] += sign * poly_coefficient_temp(j);
					}
					sign = -sign;
				}
			}

			void GeneralizedEssentialMatrixSixPointSolver::GenerateCoefficient3x3(const double *P1, const double *P2, Matrix_84x14 &poly_coefficient_84x14) const
			{
				//compute the coefficients of the 84x14 matrix. There are 14 independent equations, and each polynomial contains 84 monomials.
				double F_part0[5][10] = { 0 };
				double F_part1[5][10] = { 0 };
				double F_part2[5][10] = { 0 };

				int row_ids[10][3] = { {0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}, {0,1,4}, {0,2,4},{1,2,4},{0,3,4},{1,3,4},{2,3,4} };
				int poly_row_id = 0;

				// setting the 3D points corresponding to the 5th (the last one) ray as the origin of world coordinate.
				int reference_pts_id = 5;
				ComputeFMatrix3x3(P1, P2, reference_pts_id, F_part0, F_part1, F_part2);
				for (int i = 0; i < 10; ++i) {
					GenerateCoefficient3x3FromF(F_part0, F_part1, F_part2, row_ids[i], poly_coefficient_84x14.rows(), poly_coefficient_84x14.data() + poly_row_id * 84);
					poly_row_id++;
				}
				// setting the 3D points corresponding to the 4th  ray as the origin of world coordinate.
				reference_pts_id = 4;
				ComputeFMatrix3x3(P1, P2, reference_pts_id, F_part0, F_part1, F_part2);
				for (int i = 0; i < 4; ++i) {
					GenerateCoefficient3x3FromF(F_part0, F_part1, F_part2, row_ids[i], poly_coefficient_84x14.rows(), poly_coefficient_84x14.data() + poly_row_id * 84);
					poly_row_id++;
				}
			}

			void GeneralizedEssentialMatrixSixPointSolver::ComputeFMatrix2x2(const double *P1, const double *P2, int reference_pts_id, double F_part0[3][10], double F_part1[3][10]) const
			{
				// similar to ComputeFMatrix3x3	
				int &j = reference_pts_id;		// this id should be less than or equal to 3
				int FmatrixRowIndex = 0;
				for (int i = 0; i < 4; ++i) { // compute each row of the 4x2 matrix embedded in F
					if (j != i) {
						double in1[] = { P1[i * 6],P1[i * 6 + 1],P1[i * 6 + 2],P1[i * 6 + 3],P1[i * 6 + 4],P1[i * 6 + 5], P1[j * 6],P1[j * 6 + 1],P1[j * 6 + 2],P1[j * 6 + 3],P1[j * 6 + 4],P1[j * 6 + 5] };
						double in2[] = { P2[i * 3],P2[i * 3 + 1],P2[i * 3 + 2], P2[j * 3],P2[j * 3 + 1],P2[j * 3 + 2] };
						Generate_F2x2_matrix(in1, in2, F_part0[FmatrixRowIndex], F_part1[FmatrixRowIndex]);
						++FmatrixRowIndex;
					}
				}
			}

			void GeneralizedEssentialMatrixSixPointSolver::GenerateCoefficient2x2FromF(const double F_part0[3][10], const double F_part1[3][10],
				const int rowId[2], const int &poly_coefficient_size, double polyCoefficient[35])  const
			{
				//
				int determinantIndex[2][2] = { {rowId[0], rowId[1]}, {rowId[1], rowId[0]} };
				double sign = 1.0;
				Eigen::Matrix<double, 35, 1> poly_coefficient_temp;
				for (int i = 0; i < 2; ++i) {
					int index0 = determinantIndex[i][0];
					int index1 = determinantIndex[i][1];
					det2x2_helper(F_part0[index0], F_part1[index1], poly_coefficient_temp.data());
					for (int j = 0; j < poly_coefficient_size; j++) {
						polyCoefficient[j] += sign * poly_coefficient_temp(j);
					}
					sign = -sign;
				}
			}

			void GeneralizedEssentialMatrixSixPointSolver::GenerateCoefficient2x2(const double *P1, const double *P2, Matrix_35x4 &poly_coefficient_35x4) const
			{
				////compute the coefficients of the 35x4 matrix. There are 4 independent equations, and each polynomial contains 35 monomials.
				double F_part0[3][10] = { 0 };
				double F_part1[3][10] = { 0 };
				int row_ids[3][2] = { {0,1},{0,2},{1,2} };

				int poly_row_id = 0;

				int reference_pts_id = 3;
				ComputeFMatrix2x2(P1, P2, reference_pts_id, F_part0, F_part1);
				for (int i = 0; i < 3; ++i) {
					GenerateCoefficient2x2FromF(F_part0, F_part1, row_ids[i], poly_coefficient_35x4.rows(), poly_coefficient_35x4.data() + poly_row_id * 35);
					++poly_row_id;
				}

				reference_pts_id = 2;
				ComputeFMatrix2x2(P1, P2, reference_pts_id, F_part0, F_part1);
				for (int i = 0; i < 1; ++i) {
					GenerateCoefficient2x2FromF(F_part0, F_part1, row_ids[i], poly_coefficient_35x4.rows(), poly_coefficient_35x4.data() + poly_row_id * 35);
					++poly_row_id;
				}
			}

			void GeneralizedEssentialMatrixSixPointSolver::ComputeOnePolynomial_113(double *input, int numOfMonomials, int *poly84_multi, double *output) const
			{
				for (int i = 0; i < numOfMonomials; ++i)
				{
					if (poly84_multi[i] != -1)	// -1 means the monomials are removed.
					{
						output[poly84_multi[i]] = input[i];	// multiplication of a monomial. It is equivalent to coefficient shifting
					}
				}
			}

			//void ConstructActionMatrix(const Matrix_113x73 &finalmatrix,  Matrix_40x40 &action_matrix)
			void GeneralizedEssentialMatrixSixPointSolver::ConstructActionMatrix(const Eigen::MatrixXd &finalmatrix, Eigen::MatrixXd &action_matrix) const
			{
				int row_ids_finalmatrix_for_action_matrix[] = { 42, 43, 44, 45, 47, 48, 49, 51, 52, 54, 63, 64, 65, 66, 68, 69, 70,71 };
				int row_ids_in_action_matrix[] = { 0 ,    1 ,    2  ,   3,     4,     5,     6 ,    7 ,    8,     9,    10 ,   11 ,   12 ,   13 ,   20 ,   21 ,   22,    23 };
				int action_matrix_available_row_id[] = { 14 ,   15 ,   16 ,   17 ,   18,    19 ,   24,    25,    26,    27,    28,    29,    30,    31 ,   32  ,  33,    34 ,   35  ,  36 ,   37,    38,    39 };
				int action_matrix_available_col_Id[] = { 0,     1 ,    2 ,    4 ,    5,     7,    10,    11,    12,    14,    15,    17,    20,    21,    22,    24,    25,    27  ,  30 ,   31 ,   33,    36 };
				for (int i = 0; i < sizeof(action_matrix_available_row_id) / sizeof(action_matrix_available_row_id[0]); ++i)	// these are already in the bases.
					action_matrix(action_matrix_available_row_id[i], action_matrix_available_col_Id[i]) = 1.0;
				for (int i = 0; i < sizeof(row_ids_in_action_matrix) / sizeof(row_ids_in_action_matrix[0]); ++i)	// there are from the coefficient matrix after Gaussian eliminated
					action_matrix.block(row_ids_in_action_matrix[i], 0, 1, 40) = finalmatrix.transpose().block(row_ids_finalmatrix_for_action_matrix[i], 113 - 40, 1, 40)*(-1.0);	// the value need to multiply by -1
			}

			bool GeneralizedEssentialMatrixSixPointSolver::IsEigenVectorASolution(const Eigen::Matrix<double, 40, 1> &eigen_vector) const
			{
				//[ v21^3*v23^2, v21^2*v22*v23^2, v21*v22^2*v23^2, v22^3*v23^2, v21^2*v23^3, v21*v22*v23^3, v22^2*v23^3, v21*v23^4, v22*v23^4, v23^5,
				// v21^3*v23, v21^2*v22*v23, v21*v22^2*v23, v22^3*v23, v21^2*v23^2, v21*v22*v23^2, v22^2*v23^2, v21*v23^3, v22*v23^3, v23^4,
				// v21^3, v21^2*v22, v21*v22^2, v22^3, v21^2*v23, v21*v22*v23, v22^2*v23, v21*v23^2, v22*v23^2, v23^3, 
				// v21^2, v21*v22, v22^2, v21*v23, v22*v23, v23^2, v21, v22, v23, 1]
				// check v21^3*v23^2 * v23 == v21^2*v23^3 * v21 
				// check v21^2*v22*v23^2 * v22 ==  v21*v22^2*v23^2 * v21
				const double kEpsilon = 1e-4;
				return ((fabs(eigen_vector(0) * eigen_vector(38) - eigen_vector(4) * eigen_vector(36)) < kEpsilon)
					&& (fabs(eigen_vector(1) * eigen_vector(37) - eigen_vector(2) * eigen_vector(36)) < kEpsilon));
			}

			int GeneralizedEssentialMatrixSixPointSolver::FindAllSolutionsFromActionMatrix(const Eigen::Matrix<double, 3, 6> &q1_matrix,
				const Eigen::Matrix<double, 3, 6> &q1_prime_matrix,
				const Eigen::Matrix<double, 3, 6> &q2_matrix,
				const Eigen::MatrixXd &action_matrix, double quaternion[3 * 40], double t[3 * 40], double R[9 * 40]) const
			{
				// compute the solution from the action matrix
				// Each value of the eigenvector corresponds to 	
				//[ v21^3*v23^2, v21^2*v22*v23^2, v21*v22^2*v23^2, v22^3*v23^2, v21^2*v23^3, v21*v22*v23^3, v22^2*v23^3, v21*v23^4, v22*v23^4, v23^5,
				// v21^3*v23, v21^2*v22*v23, v21*v22^2*v23, v22^3*v23, v21^2*v23^2, v21*v22*v23^2, v22^2*v23^2, v21*v23^3, v22*v23^3, v23^4,
				// v21^3, v21^2*v22, v21*v22^2, v22^3, v21^2*v23, v21*v22*v23, v22^2*v23, v21*v23^2, v22*v23^2, v23^3, 
				// v21^2, v21*v22, v22^2, v21*v23, v22*v23, v23^2, v21, v22, v23, 1]
				//const EigenSolver<Matrix_40x40> eigen_solver(action_matrix);
				const Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(action_matrix);
				//const auto & eigen_vectors = eigen_solver.eigenvectors();
				const Eigen::Matrix<std::complex<double>, -1, -1> & eigen_vectors = eigen_solver.eigenvectors();
				//const auto & eigen_values = eigen_solver.eigenvalues();
				const Eigen::Matrix<std::complex<double>, -1, 1> & eigen_values = eigen_solver.eigenvalues();


				Eigen::Map<Eigen::Matrix<double, 3, 40> > quaternion_matrix(quaternion);
				Eigen::Map<Eigen::Matrix<double, 3, 40> > t_matrix(t);
				Eigen::Map<Eigen::Matrix<double, 9, 40> > R_matrix(R);
				//Map<const Matrix<double, 3, 6> > q1_matrix(q1);
				//Map<const Matrix<double, 3, 6> > q1_prime_matrix(q1_prime);
				//Map<const Matrix<double, 3, 6> > q2_matrix(q2);

				int solutionIndex = 0;
				for (int i = 0; i < eigen_vectors.cols(); ++i) {
					if (eigen_values[i].imag() == 0) {	// real solutions
						Eigen::Matrix<double, 3, 1> one_quaternion;
						Eigen::Matrix<double, 40, 1> eigen_vector = eigen_vectors.col(i).real();
						if (!IsEigenVectorASolution(eigen_vector)) continue;
						for (int j = 0; j < 3; ++j) {
							//quaternion_matrix(j, solutionIndex) = (eigen_vectors(36 + j, i)/eigen_vectors(39, i)).real();
							one_quaternion(j) = (eigen_vectors(36 + j, i) / eigen_vectors(39, i)).real();
						}
						// compute R
						Eigen::Matrix3d skew_matrix;
						skew_matrix << 0, -one_quaternion(2), one_quaternion(1), one_quaternion(2), 0, -one_quaternion(0), -one_quaternion(1), one_quaternion(0), 0;
						Eigen::Matrix3d one_R_matrix = 2 * (one_quaternion*one_quaternion.transpose() - skew_matrix)
							+ (1 - one_quaternion.transpose() * one_quaternion) * Eigen::Matrix3d::Identity();
						Eigen::Matrix3d one_R_matrix_temp = one_R_matrix * one_R_matrix.transpose();
						one_R_matrix = one_R_matrix / sqrt(one_R_matrix_temp(0, 0));

						//	std::cout << one_R_matrix << std::endl << std::endl;;

							// compute T
						Eigen::Matrix<double, 6, 1> b;
						Eigen::Matrix<double, 6, 3> A;
						Eigen::Matrix<double, 3, 6> Rq1 = one_R_matrix * q1_matrix;
						for (int i = 0; i < 6; i++) {
							A(i, 0) = Rq1(1, i) * q2_matrix(2, i) - Rq1(2, i) * q2_matrix(1, i);
							A(i, 1) = Rq1(2, i) * q2_matrix(0, i) - Rq1(0, i) * q2_matrix(2, i);
							A(i, 2) = Rq1(0, i) * q2_matrix(1, i) - Rq1(1, i) * q2_matrix(0, i);
							b(i) = q2_matrix.col(i).transpose() * one_R_matrix * q1_prime_matrix.col(i);
						}
						//Matrix<double, 3, 1> one_t_matrix = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b); // least square. Not sure why crashed
						Eigen::Matrix<double, 3, 1> one_t_matrix = A.colPivHouseholderQr().solve(b);
						// assign the value 
						quaternion_matrix.col(solutionIndex) = one_quaternion;
						R_matrix.col(solutionIndex) = Eigen::Map<Eigen::Matrix<double, 9, 1> >(one_R_matrix.data());
						t_matrix.col(solutionIndex) = one_t_matrix;

						/*if( fabs(one_quaternion(0) - 0.3981) < 0.01)
						{
							std::cout << A << std::endl << std::endl;
							std::cout << b << std::endl << std::endl;
							std::cout << -one_R_matrix.transpose() * t_matrix.col(solutionIndex) << std::endl;
						}*/
						++solutionIndex;
					}
				}
				return solutionIndex;
			}


			void GeneralizedEssentialMatrixSixPointSolver::ConstructMatrix113x73(double *poly_coefficient_84x14, double *poly_coefficient_35x4, Eigen::MatrixXd &finalmatrix) const
			{
				// !!! finalmatrix need to be initialized to 0
				// Matrix_113x73 finalmatrix = Matrix_113x73::Zero(113,73);

				Eigen::Map<Matrix_84x14> poly_coefficient_matrix_84x14(poly_coefficient_84x14);
				Eigen::PartialPivLU<Matrix_14x14> lu_14_14(poly_coefficient_matrix_84x14.topRows(14).transpose());
				Matrix_84x14 u_84_14 = (lu_14_14.solve(poly_coefficient_matrix_84x14.transpose())).transpose();
				//Matrix_84x14 u_84_14 = poly_coefficient_matrix_84x14;

				Eigen::Map<Matrix_35x4> poly_coefficient_matrix_35x4(poly_coefficient_35x4);
				Eigen::PartialPivLU<Matrix_4x4> lu_4_4(poly_coefficient_matrix_35x4.topRows(4).transpose());
				Matrix_35x4 u_35_4 = (lu_4_4.solve(poly_coefficient_matrix_35x4.transpose())).transpose();

				// degree 6 polynomials multiply 1,v1,v2,v3
				int colIdx = 0;
				int assignedColIdx = 0;

				//poly84_multi indicates the coefficient shift after multiplying 1,v1,v2,v3
				int *poly84_multi[] = { poly84_multi_1, poly84_multi_v1, poly84_multi_v2, poly84_multi_v3 };
				for (int i = 0; i < sizeof(poly84_multi) / sizeof(poly84_multi[0]); ++i) {	// the size of poly84_multi
					for (int j = 0; j < u_84_14.cols(); ++j) {	// the number of equations with polynomial degree equals 6
						if (selectedPolynomials[colIdx]) {
							ComputeOnePolynomial_113(u_84_14.col(j).data(), u_84_14.rows(), poly84_multi[i], finalmatrix.col(assignedColIdx).data());
							++assignedColIdx;
						}
						++colIdx;
					}
				}
				// degree 4 multiply 1, v1, v2, v3, v1^2, v2^2, v3^2
				int *poly35_multi[] = { poly35_multi_1, poly35_multi_v1, poly35_multi_v2, poly35_multi_v3, poly35_multi_v1Square, poly35_multi_v2Square, poly35_multi_v3Square };
				for (int i = 0; i < sizeof(poly35_multi) / sizeof(poly35_multi[0]); ++i) {	// the size of poly35_multi
					for (int j = 0; j < u_35_4.cols(); ++j) {	// the number of equations with polynomial degree equals 4
						if (selectedPolynomials[colIdx]) {
							ComputeOnePolynomial_113(u_35_4.col(j).data(), u_35_4.rows(), poly35_multi[i], finalmatrix.col(assignedColIdx).data());
							++assignedColIdx;
						}
						++colIdx;
					}
				}
				// permute finalmatrix so that the left 73x73 matrix is full rank, and do Gaussian elimination
				Eigen::PermutationMatrix<113> perm_finalmatrix;
				perm_finalmatrix.setIdentity();
				for (int i = 68; i < 78; i++)
					perm_finalmatrix.indices()[i + 5] = i;
				for (int i = 78; i < 83; i++)
					perm_finalmatrix.indices()[i - 10] = i;
				finalmatrix = (finalmatrix.transpose() * perm_finalmatrix).transpose();

				//Eigen::PartialPivLU<Matrix_73x73> lu_73_73(finalmatrix.topRows(73).transpose());	
				Eigen::PartialPivLU<Eigen::MatrixXd> lu_73_73(finalmatrix.topRows(73).transpose());
				//finalmatrix = (lu_73_73.solve(finalmatrix.transpose()) ).transpose();
				finalmatrix.bottomRows(40) = (lu_73_73.solve(finalmatrix.bottomRows(40).transpose())).transpose();

			}


			void GeneralizedEssentialMatrixSixPointSolver::ComputePluckerCoordinate(const double ray_directions1[3 * 6], const double centers1[3 * 6], const double ray_directions2[3 * 6], Eigen::Matrix<double, 3, 6> &q1_matrix, Eigen::Matrix<double, 3, 6> &q1_prime_matrix, Eigen::Matrix<double, 3, 6> &q2_matrix)  const
			{
				Eigen::Matrix<double, 3, 6> ray_directions1_matrix(ray_directions1);
				Eigen::Matrix<double, 3, 6> center1_matrix(centers1);
				Eigen::Matrix<double, 3, 6> ray_directions2_matrix(ray_directions2);
				for (int i = 0; i < 6; ++i) {
					q1_matrix.col(i) = ray_directions1_matrix.col(i) / ray_directions1_matrix.col(i).norm();
					q2_matrix.col(i) = ray_directions2_matrix.col(i) / ray_directions2_matrix.col(i).norm();
				}
				for (int i = 0; i < 6; ++i) {
					q1_prime_matrix.col(i) = (center1_matrix.col(i) + q1_matrix.col(i)).cross(center1_matrix.col(i));
				}
			}

			int GeneralizedEssentialMatrixSixPointSolver::RelativePoseGeneralizedCalibratedPinhole(
				const double ray_directions1[3 * 6],
				const double centers1[3 * 6],
				const double ray_directions2[3 * 6],
				double quaternion[3 * 40],
				double t[3 * 40],
				double R[9 * 40]) const
			{
				// The 1st 4 pts in q1 should come from one pinhole camera. If not, return 0.
				for (int i = 0; i < 3 * 4; ++i) {
					if (centers1[i] != 0) return 0;
				}

				// compute plucker coordinate
				Eigen::Matrix<double, 3, 6> q1_matrix;
				Eigen::Matrix<double, 3, 6> q1_prime_matrix;
				Eigen::Matrix<double, 3, 6> q2_matrix;
				ComputePluckerCoordinate(ray_directions1, centers1, ray_directions2, q1_matrix, q1_prime_matrix, q2_matrix);
				// combine q1_matrix and q1_prime_matrix	
				Eigen::Matrix<double, 6, 6> Q1_matrix;
				Q1_matrix.topRows(3) = q1_matrix;
				Q1_matrix.bottomRows(3) = q1_prime_matrix;

				// compute the coefficients. For polynomials with maximum degree 6, there are 14 independent equations.
				Matrix_84x14 poly_coefficient_84x14 = Matrix_84x14::Zero(84, 14);
				GenerateCoefficient3x3(Q1_matrix.data(), q2_matrix.data(), poly_coefficient_84x14);
				// 4 independent equations
				Matrix_35x4 poly_coefficient_35x4 = Matrix_35x4::Zero(35, 4);
				GenerateCoefficient2x2(Q1_matrix.data(), q2_matrix.data(), poly_coefficient_35x4);

				//  finalmatrix is the matrix after multiply variables and doing Gaussian elimination (need to be initialized to 0).
				Eigen::MatrixXd finalmatrix = Eigen::MatrixXd::Zero(113, 73);
				ConstructMatrix113x73(poly_coefficient_84x14.data(), poly_coefficient_35x4.data(), finalmatrix);
				// construct action matrix from finalmatrix
				//Matrix_40x40 action_matrix =  Matrix_40x40::Zero(40,40); // need to be initialized to 0
				Eigen::MatrixXd action_matrix = Eigen::MatrixXd::Zero(40, 40); // need to be initialized to 0
				ConstructActionMatrix(finalmatrix, action_matrix);
				// compute all solutions (quaternion, translation, and rotation) from action matrix
				Eigen::Matrix<double, 3, 40> allsolutions = Eigen::Matrix<double, 3, 40>::Zero(3, 40);
				return FindAllSolutionsFromActionMatrix(q1_matrix, q1_prime_matrix, q2_matrix, action_matrix, quaternion, t, R);

				//	Map<Matrix<double, 3, 40> > quaternion_matrix(quaternion);
				//	std::cout << quaternion_matrix << std::endl << std::endl;;
				//	Map<Matrix<double, 9, 40> > R_matrix(R);
				//	std::cout << R_matrix << std::endl << std::endl;;
				//	Map<Matrix<double, 3, 40> > t_matrix(t);
				//	std::cout << t_matrix << std::endl;
			}

			// Estimate the model parameters from the given point sample
			// using weighted fitting if possible.
			OLGA_INLINE bool GeneralizedEssentialMatrixSixPointSolver::estimateModel(
				const cv::Mat& data_, // The set of data points
				const size_t *sample_, // The sample used for the estimation
				size_t sample_number_, // The size of the sample
				std::vector<Model> &models_, // The estimated model parameters
				const double *weights_) const // The weight for each point
			{
				const size_t kColumns = data_.cols;
				const double *data_ptr = reinterpret_cast<double *>(data_.data);

				double ray_directions1[3 * 6];
				double centers1[3 * 6];
				double ray_directions2[3 * 6];

				for (size_t i = 0; i < sample_number_; ++i)
				{
					const size_t idx =
						sample_ == nullptr ? i : sample_[i];

					const double *point_ptr =
						data_ptr + idx * kColumns;

					ray_directions1[i * 3] = point_ptr[0];
					ray_directions1[i * 3 + 1] = point_ptr[1];
					ray_directions1[i * 3 + 2] = 1;
					ray_directions2[i * 3] = point_ptr[2];
					ray_directions2[i * 3 + 1] = point_ptr[3];
					ray_directions2[i * 3 + 2] = point_ptr[4];
					centers1[i * 3] = point_ptr[5];
					centers1[i * 3 + 1] = point_ptr[6];
					centers1[i * 3 + 2] = point_ptr[7];
				}

				double quaternion[3 * 40];
				double ts[3 * 40];
				double Rs[9 * 40];

				int numSols = RelativePoseGeneralizedCalibratedPinhole(
					ray_directions1,
					centers1,
					ray_directions2,
					quaternion,
					ts,
					Rs);
			
				for (int i = 0; i < numSols; ++i)
				{
					Eigen::Vector3d t;
					Eigen::Matrix3d R;
					/*t << ts[3 * i], ts[3 * i + 1], ts[3 * i + 2];
					R = Eigen::Map<Eigen::Matrix3d>(&(Rs[9 * i]));*/
					
					R = Eigen::Map<Eigen::Matrix3d>(&(Rs[9 * i]));
					t = Eigen::Map<Eigen::Vector3d>(&(ts[3 * i]));

					/*R <<
						Rs[9 * i], Rs[9 * i + 1], Rs[9 * i + 2],
						Rs[9 * i + 3], Rs[9 * i + 4], Rs[9 * i + 5],
						Rs[9 * i + 6], Rs[9 * i + 7], Rs[9 * i + 8];*/

					Model model;
					model.descriptor = Eigen::MatrixXd(3, 4);
					model.descriptor << R, -t;
					models_.emplace_back(model);
				}

				return numSols > 0;
			}
		}
	}
}