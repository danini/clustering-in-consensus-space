#pragma once

#include <vector>
#include <sophus/se3.hpp>

namespace pose_averaging {

	enum class eTranslationAveragingMethod {
		Naive,
		OpenMVG_L2
	};

	bool poseAveraging_openMVG(
		const std::vector<Sophus::SE3d>& poses_,
		Sophus::SE3d& estimatedPose_,
		eTranslationAveragingMethod translationAvergaingMethod = eTranslationAveragingMethod::OpenMVG_L2);

	// weighted rotation averaging
	bool rotationAveraging_openMVG(
		const std::vector<Sophus::SE3d>& poses_,
		const std::vector<double>& weights,
		Eigen::Matrix3d& rotationMatrix,
		std::vector<bool>& vec_inliers);

	bool rotationAveraging_openMVG(
		const std::vector<Sophus::SE3d>& poses_,
		Eigen::Matrix3d& rotationMatrix, 
		std::vector<bool>& vec_inliers);

	// Random hacking
	// NOTE: this is just a really naive "translation averaging": just takes the mean of the translational part of the inlier poses
	bool translationAveraging_naive(
		const std::vector<Sophus::SE3d>& poses_,
		const std::vector<bool>& vec_inliers,
		Eigen::Vector3d& t);

	// Random hacking
	// By default, computes L2 chordal minimization of the translational part of the inlier poses
	bool translationAveraging_openMVG(
		const std::vector<Sophus::SE3d>& poses_,
		const std::vector<bool>& vec_inliers,
		Eigen::Vector3d& t);

	// TODO / ideas
	// - a kind of translation averaging, that re-estimates the translation vector robustly, given the global pose and feature correspondences
	// - a kind of translation averaging, that re-estimates the translation vector robustly, given the global pose and homographies

}