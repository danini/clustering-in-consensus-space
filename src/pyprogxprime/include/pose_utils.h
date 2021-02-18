#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Eigen>

namespace pose
{
	/*********************************************************************
	***************************** DECLARATION ****************************
	*********************************************************************/
	inline bool linearTriangulation(
		const Eigen::Matrix<double, 3, 4>& projection_1_,
		const Eigen::Matrix<double, 3, 4>& projection_2_,
		const cv::Mat& point_,
		Eigen::Vector4d& triangulated_point_);

	inline void decomposeHomographyMatrix(
		const Eigen::Matrix3d& homography_,
		const Eigen::Matrix3d& intrinsics_src_,
		const Eigen::Matrix3d& intrinsics_dst_,
		std::vector<Eigen::Matrix3d>& Rs_dst_src_,
		std::vector<Eigen::Vector3d>& ts_dst_src_,
		std::vector<Eigen::Vector3d>& normals_);

	inline void poseFromHomographyMatrix(
		const Eigen::Matrix3d& homography_,
		const Eigen::Matrix3d& intrinsics_src_,
		const Eigen::Matrix3d& intrinsics_dst_,
		const cv::Mat& correspondences_,
		const std::vector<size_t>& inliers_,
		Eigen::Matrix3d& R_dst_src_,
		Eigen::Vector3d& t_dst_src_,
		Eigen::Vector3d& normal_,
		std::vector<Eigen::Vector3d>& points3D_);

	inline double calculateDepth(
		const Eigen::Matrix<double, 3, 4>& proj_matrix_,
		const Eigen::Vector3d& point3D_);

	inline bool checkCheirality(
		const Eigen::Matrix3d& R_dst_src_,
		const Eigen::Vector3d& t_dst_src_,
		const cv::Mat& correspondences_,
		const std::vector<size_t>& inliers_,
		std::vector<Eigen::Vector3d>& points3D_);

	inline double computeOppositeOfMinor(
		const Eigen::Matrix3d& matrix_,
		const size_t row_,
		const size_t col_);

	template <typename T>
	inline int signOfNumber(const T val);

	inline Eigen::Matrix3d computeHomographyRotation(
		const Eigen::Matrix3d& H_normalized,
		const Eigen::Vector3d& tstar,
		const Eigen::Vector3d& n,
		const double v);

	/*********************************************************************
	*************************** IMPLEMENTATION ***************************
	*********************************************************************/
	inline void convertToRotationMatrix(Eigen::Matrix3d& R_dst_src_, 
		double* scale_) 
	{
		const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(R_dst_src_);
		const Eigen::PermutationMatrix<3, 3> permutation_matrix(qr.colsPermutation());
		R_dst_src_ = qr.householderQ();
		const Eigen::VectorXd diag = qr.matrixQR().diagonal();
		for (int i = 0; i < 3; ++i)
			if (diag(i) < 0)
				R_dst_src_.col(i) = -R_dst_src_.col(i);
		R_dst_src_ = R_dst_src_ * permutation_matrix.inverse();

		// Recovering the scale of the input matrix
		if (scale_ != nullptr)
			*scale_ = diag.cwiseAbs().sum() / 3.0;
	}

	// Deciding if a particular point is in from of the camera represented by its rotation and
	// translation
	inline bool isTriangulatedPointInFrontOfCameras(
		const cv::Mat& correspondence_,
		const Eigen::Matrix3d& rotation_,
		const Eigen::Vector3d& position_) {

		Eigen::Vector3d dir1, dir2;
		dir1 << correspondence_.at<double>(0), correspondence_.at<double>(1), 1;
		dir2 << correspondence_.at<double>(2), correspondence_.at<double>(3), 1;

		const double dir1_sq = dir1.squaredNorm();
		const double dir2_sq = dir2.squaredNorm();
		const double dir1_dir2 = dir1.dot(dir2);
		const double dir1_pos = dir1.dot(position_);
		const double dir2_pos = dir2.dot(position_);

		return (
			dir2_sq * dir1_pos - dir1_dir2 * dir2_pos > 0 &&
			dir1_dir2 * dir1_pos - dir1_sq * dir2_pos > 0);
	}

	// Decomposes the essential matrix into the rotation R and translation t such
	// that E can be any of the four candidate solutions: [rotation1 | translation],
	// [rotation1 | -translation], [rotation2 | translation], [rotation2 |
	// -translation].
	inline void decomposeEssentialMatrix(
		const Eigen::Matrix3d& essentialMatrix_,
		Eigen::Matrix3d& rotation_1_,
		Eigen::Matrix3d& rotation_2_,
		Eigen::Vector3d& translation_)
	{
		Eigen::Matrix3d d;
		d << 0, 1, 0, -1, 0, 0, 0, 0, 1;

		const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
			essentialMatrix_, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d V = svd.matrixV();
		if (U.determinant() < 0) {
			U.col(2) *= -1.0;
		}

		if (V.determinant() < 0) {
			V.col(2) *= -1.0;
		}

		// Possible configurations.
		rotation_1_ = U * d * V.transpose();
		rotation_2_ = U * d.transpose() * V.transpose();
		translation_ = U.col(2).normalized();
	}

	// Recovering the relative pose from the essential matrix
	inline int getPoseFromEssentialMatrix(
		const Eigen::Matrix3d& essentialMatrix_,
		const cv::Mat& normalizedCorrespondences_,
		Eigen::Matrix3d& rotation_,
		Eigen::Vector3d& translation_)
	{
		// Decompose the essential matrix.
		Eigen::Matrix3d rotation1, rotation2;
		Eigen::Vector3d translation;
		decomposeEssentialMatrix(essentialMatrix_, rotation1, rotation2, translation);
		const std::vector<Eigen::Matrix3d> rotations = { rotation1, rotation1, rotation2, rotation2 };

		// From the 4 candidate poses, find the one with the most triangulated points
		// in front of the camera.
		std::vector<size_t> points_in_front_of_cameras(4, 0);
		const Eigen::Matrix<double, 3, 4> projectionMatrixSource = Eigen::Matrix<double, 3, 4>::Identity();

		std::vector<double> bestDistances(normalizedCorrespondences_.rows, std::numeric_limits<double>::max());
		std::vector<int> bestPoses(normalizedCorrespondences_.rows, 5);
		for (auto i = 0; i < 4; i++)
		{
			Eigen::Matrix<double, 3, 4> projectionMatrixDestination;
			projectionMatrixDestination << rotations[i], ((i % 2 ? -1 : 1) * translation);

			for (size_t pointIdx = 0; pointIdx < normalizedCorrespondences_.rows; ++pointIdx) {

				Eigen::Matrix<double, 4, 1> triangulatedPoint;
				linearTriangulation(projectionMatrixSource, 
					projectionMatrixDestination, 
					normalizedCorrespondences_.row(pointIdx), 
					triangulatedPoint);

				Eigen::Vector3d projectedPointSource = 
					projectionMatrixSource * triangulatedPoint;

				if (projectedPointSource(2) < 0)
					continue;

				Eigen::Vector3d projectedPointDestination = 
					projectionMatrixDestination * triangulatedPoint;

				if (projectedPointDestination(2) < 0)
					continue;

				const double& kX1 = normalizedCorrespondences_.at<double>(pointIdx, 0),
					& kY1 = normalizedCorrespondences_.at<double>(pointIdx, 1),
					& kX2 = normalizedCorrespondences_.at<double>(pointIdx, 2),
					& kY2 = normalizedCorrespondences_.at<double>(pointIdx, 3);

				projectedPointSource /= projectedPointSource(2);
				projectedPointDestination /= projectedPointDestination(2);

				const double dX1 = projectedPointSource(0) - kX1,
					dY1 = projectedPointSource(1) - kY1,
					dX2 = projectedPointDestination(0) - kX2,
					dY2 = projectedPointDestination(1) - kY2;

				const double squaredResidualSource =
					dX1 * dX1 + dY1 * dY1;
				const double squaredResidualDestination =
					dX2 * dX2 + dY2 * dY2;
				const double totalSquaredResidual =
					squaredResidualSource + squaredResidualDestination;

				if (squaredResidualSource < bestDistances[pointIdx])
				{
					bestDistances[pointIdx] = squaredResidualSource;
					bestPoses[pointIdx] = i;
				}
			}
		}

		for (const int& vote : bestPoses)
			if (vote < 5)
				++points_in_front_of_cameras[vote];

		// Find the pose with the most points in front of the camera.
		const auto& max_element = 
			std::max_element(points_in_front_of_cameras.begin(),
				points_in_front_of_cameras.end());
		const int max_index =
			std::distance(points_in_front_of_cameras.begin(), max_element);

		// Set the pose.
		rotation_ = rotations[max_index];
		translation_ = (max_index % 2 ? -1 : 1) * translation; 
		return *max_element;
	}

	inline void poseFromHomographyMatrix(
		const Eigen::Matrix3d& homography_,
		const Eigen::Matrix3d& intrinsicsSource_,
		const Eigen::Matrix3d& intrinsicsDestination_,
		const cv::Mat& correspondences_,
		const std::vector<size_t>& inliers_,
		Eigen::Matrix3d& RDstSrc_,
		Eigen::Vector3d& tDstSrc_,
		Eigen::Vector3d& normal_,
		std::vector<Eigen::Vector3d>& points3D_) 
	{
		std::vector<Eigen::Matrix3d> RCmbs;
		std::vector<Eigen::Vector3d> tCmbs;
		std::vector<Eigen::Vector3d> nCmbs;

		decomposeHomographyMatrix(
			homography_,
			intrinsicsSource_,
			intrinsicsDestination_,
			RCmbs,
			tCmbs,
			nCmbs);

		points3D_.clear();
		for (size_t i = 0; i < RCmbs.size(); ++i) {
			std::vector<Eigen::Vector3d> points3DCmb;
			checkCheirality(RCmbs[i], 
				tCmbs[i], 
				correspondences_, 
				inliers_, 
				points3DCmb);

			if (points3DCmb.size() >= points3D_.size()) 
			{
				RDstSrc_ = RCmbs[i];
				tDstSrc_ = tCmbs[i];
				normal_ = nCmbs[i];
				points3D_ = points3DCmb;
			}
		}
	}

	// Decomposing a given homography matrix into the possible pose parameters
	inline void decomposeHomographyMatrix(
		const Eigen::Matrix3d& homography_, // The homography matrix to be decomposed
		const Eigen::Matrix3d& intrinsicsSource_, // The intrinsic parameters of the first camera
		const Eigen::Matrix3d& intrinsicsDestination_, // The intrinsic parameters of the second camera
		std::vector<Eigen::Matrix3d>& RsDstSrc_, // The possible rotation matrices
		std::vector<Eigen::Vector3d>& tsDstSrc_, // The possible translation vectors
		std::vector<Eigen::Vector3d>& normals_) // The possible plane normals
	{
		// Remove calibration from homography.
		Eigen::Matrix3d normalizedHomography = intrinsicsDestination_.inverse() * homography_ * intrinsicsSource_;

		// Remove scale from normalized homography.
		Eigen::JacobiSVD<Eigen::Matrix3d> normalizedHomographySVD(normalizedHomography);
		normalizedHomography.array() /= normalizedHomographySVD.singularValues()[1];

		const Eigen::Matrix3d S = 
			normalizedHomography.transpose() * normalizedHomography - Eigen::Matrix3d::Identity();

		// Check if H is rotation matrix.
		const double kMinInfinityNorm = 1e-3;
		if (S.lpNorm<Eigen::Infinity>() < kMinInfinityNorm) {
			RsDstSrc_.emplace_back(normalizedHomography);
			tsDstSrc_.emplace_back(Eigen::Vector3d::Zero());
			normals_.emplace_back(Eigen::Vector3d::Zero());
			return;
		}

		const double M00 = computeOppositeOfMinor(S, 0, 0);
		const double M11 = computeOppositeOfMinor(S, 1, 1);
		const double M22 = computeOppositeOfMinor(S, 2, 2);

		const double rtM00 = std::sqrt(M00);
		const double rtM11 = std::sqrt(M11);
		const double rtM22 = std::sqrt(M22);

		const double M01 = computeOppositeOfMinor(S, 0, 1);
		const double M12 = computeOppositeOfMinor(S, 1, 2);
		const double M02 = computeOppositeOfMinor(S, 0, 2);

		const int e12 = signOfNumber(M12);
		const int e02 = signOfNumber(M02);
		const int e01 = signOfNumber(M01);

		const double nS00 = std::abs(S(0, 0));
		const double nS11 = std::abs(S(1, 1));
		const double nS22 = std::abs(S(2, 2));

		const std::array<double, 3> nS{ {nS00, nS11, nS22} };
		const size_t idx = std::distance(nS.begin(), std::max_element(nS.begin(), nS.end()));

		Eigen::Vector3d np1;
		Eigen::Vector3d np2;
		if (idx == 0) {
			np1[0] = S(0, 0);
			np2[0] = S(0, 0);
			np1[1] = S(0, 1) + rtM22;
			np2[1] = S(0, 1) - rtM22;
			np1[2] = S(0, 2) + e12 * rtM11;
			np2[2] = S(0, 2) - e12 * rtM11;
		}
		else if (idx == 1) {
			np1[0] = S(0, 1) + rtM22;
			np2[0] = S(0, 1) - rtM22;
			np1[1] = S(1, 1);
			np2[1] = S(1, 1);
			np1[2] = S(1, 2) - e02 * rtM00;
			np2[2] = S(1, 2) + e02 * rtM00;
		}
		else if (idx == 2) {
			np1[0] = S(0, 2) + e01 * rtM11;
			np2[0] = S(0, 2) - e01 * rtM11;
			np1[1] = S(1, 2) + rtM00;
			np2[1] = S(1, 2) - rtM00;
			np1[2] = S(2, 2);
			np2[2] = S(2, 2);
		}

		const double traceS = S.trace();
		const double v = 2.0 * std::sqrt(1.0 + traceS - M00 - M11 - M22);

		const double ESii = signOfNumber(S(idx, idx));
		const double r_2 = 2 + traceS + v;
		const double nt_2 = 2 + traceS - v;

		const double r = std::sqrt(r_2);
		const double n_t = std::sqrt(nt_2);

		const Eigen::Vector3d n1 = np1.normalized();
		const Eigen::Vector3d n2 = np2.normalized();

		const double half_nt = 0.5 * n_t;
		const double esii_t_r = ESii * r;

		const Eigen::Vector3d t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
		const Eigen::Vector3d t2_star = half_nt * (esii_t_r * n1 - n_t * n2);

		const Eigen::Matrix3d R1 = computeHomographyRotation(normalizedHomography, t1_star, n1, v);
		const Eigen::Vector3d t1 = R1 * t1_star;

		const Eigen::Matrix3d R2 = computeHomographyRotation(normalizedHomography, t2_star, n2, v);
		const Eigen::Vector3d t2 = R2 * t2_star;

		RsDstSrc_.emplace_back(-R1);
		RsDstSrc_.emplace_back(R1);
		RsDstSrc_.emplace_back(-R2);
		RsDstSrc_.emplace_back(R2);

		tsDstSrc_.emplace_back(t1);
		tsDstSrc_.emplace_back(-t1);
		tsDstSrc_.emplace_back(t2);
		tsDstSrc_.emplace_back(-t2);

		normals_.emplace_back(-n1);
		normals_.emplace_back(n1);
		normals_.emplace_back(-n2);
		normals_.emplace_back(n2);
	}

	// Calculate the rotation matrix from a given homography, translation,
	// plane normal, and plane distance.
	inline Eigen::Matrix3d computeHomographyRotation(
		const Eigen::Matrix3d& Hnormalized_,
		const Eigen::Vector3d& tstar_,
		const Eigen::Vector3d& normal_,
		const double v_) {
		return Hnormalized_ * (Eigen::Matrix3d::Identity() - (2.0 / v_) * tstar_ * normal_.transpose());
	}

	inline double computeOppositeOfMinor(
		const Eigen::Matrix3d& matrix_,
		const size_t row_,
		const size_t col_) {
		const size_t col1 = col_ == 0 ? 1 : 0;
		const size_t col2 = col_ == 2 ? 1 : 2;
		const size_t row1 = row_ == 0 ? 1 : 0;
		const size_t row2 = row_ == 2 ? 1 : 2;
		return (matrix_(row1, col2) * matrix_(row2, col1) - matrix_(row1, col1) * matrix_(row2, col2));
	}

	// The method collection the 3D points which are in front of both cameras
	// after triangulation.
	inline bool checkCheirality(
		const Eigen::Matrix3d& RDstSrc_,
		const Eigen::Vector3d& tDstSrc_,
		const cv::Mat& correspondences_,
		const std::vector<size_t>& inliers_,
		std::vector<Eigen::Vector3d>& points3D_)
	{
		// Initialize the first camera's projection matrix to be in the origin
		static const Eigen::Matrix<double, 3, 4> projectionMatrixSource = Eigen::Matrix<double, 3, 4>::Identity();

		// Initialize the second camera's projection matrix from the given rotation and translation
		Eigen::Matrix<double, 3, 4> projectionMatrixDestination;
		projectionMatrixDestination.leftCols<3>() = RDstSrc_;
		projectionMatrixDestination.rightCols<1>() = tDstSrc_;

		// Iterating through all correspondences, estimating their 3D coordinates and
		// collecting the ones which end up in front of both cameras.
		constexpr double kMinDepth = std::numeric_limits<double>::epsilon();
		const double maxDepth = 1000.0f * (RDstSrc_.transpose() * tDstSrc_).norm();
		points3D_.clear();
		for (size_t i = 0; i < inliers_.size(); ++i) {

			const size_t& pointIdx = inliers_[i];

			Eigen::Vector4d homogeneousPoint3D;
			linearTriangulation(
				projectionMatrixSource, // The first cameras's projection matrix
				projectionMatrixDestination, // The second cameras's projection matrix
				correspondences_.row(pointIdx), // The point correspondence
				homogeneousPoint3D); // The estimated 3D coordinate in their homogeneous form
			const Eigen::Vector3d point3D = homogeneousPoint3D.head<3>();

			const double depthSource = calculateDepth(projectionMatrixSource, point3D); // Get the depth in the first image
			if (depthSource > kMinDepth && depthSource < maxDepth) {
				const double depth2 =
					calculateDepth(projectionMatrixDestination, point3D); // Get the depth in the second image
				if (depth2 > kMinDepth && depth2 < maxDepth) {
					points3D_.push_back(point3D);
				}
			}
		}
		// The procedure was successful if there is at least a single 3D point in front of both cameras
		return !points3D_.empty();
	}

	// Returning the depth of a 3D point given a projection matrix
	inline double calculateDepth(
		const Eigen::Matrix<double, 3, 4>& projectionMatrix_,
		const Eigen::Vector3d& point3D_) 
	{
		const double proj_z = projectionMatrix_.row(2).dot(point3D_.homogeneous());
		return proj_z * projectionMatrix_.col(2).norm();
	}

	// Returns the sign of a number
	template <typename T>
	inline int signOfNumber(const T value_) {
		return (T(0) < value_) - (value_ < T(0));
	}

	inline bool linearTriangulation(
		const Eigen::Matrix<double, 3, 4>& projectionSource_,
		const Eigen::Matrix<double, 3, 4>& projectionDestination_,
		const cv::Mat& point_,
		Eigen::Vector4d& triangulatedPoint_)
	{
		Eigen::Matrix4d designMatrix;
		designMatrix.row(0) = point_.at<double>(0) * projectionSource_.row(2) - projectionSource_.row(0);
		designMatrix.row(1) = point_.at<double>(1) * projectionSource_.row(2) - projectionSource_.row(1);
		designMatrix.row(2) = point_.at<double>(2) * projectionDestination_.row(2) - projectionDestination_.row(0);
		designMatrix.row(3) = point_.at<double>(3) * projectionDestination_.row(2) - projectionDestination_.row(1);

		// Extract nullspace.
		triangulatedPoint_ = designMatrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
		return true;
	}
}