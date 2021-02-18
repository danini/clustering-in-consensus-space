#include <vector>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"
#include "progx_utils.h"
#include "losses.h"
#include "progressive_x_prime.h"
#include "median_shift_clustering.h"
#include "mean_shift_clustering.h"
#include "dbscan_clustering.h"
#include "distances.h"
#include "utils.h"
#include "pose_utils.h"
#include "modified_types.h"
#include "GCoptimization.h"
#include "grid_neighborhood_graph.h"
#include "flann_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "modified_fundamental_estimator.h"
#include "solver_homography_three_point.h"
#include "modified_homography_estimator.h"
#include "subspace4_estimator.h"
#include "essential_estimator.h"
#include "preemption_sprt.h"

#include <ctime>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <gflags/gflags.h>

#include <mutex>

DEFINE_string(data_path, "",
	"The path where all the data are found.");
DEFINE_string(source_image_path, "",
	"The path to the source image.");
DEFINE_string(destination_image_path, "",
	"The path to the destination image.");
DEFINE_string(source_intrinsics_path, "",
	"The path to the intrinsic parameters of the source image.");
DEFINE_string(destination_intrinsics_path, "",
	"The path to the intrinsic parameters of the destination image.");
DEFINE_string(source_pose_path, "",
	"The path to the pose parameters of the source image.");
DEFINE_string(destination_pose_path, "",
	"The path to the pose parameters of the destination image.");
DEFINE_string(workspace_path, "",
	"This is where the algorithm saves the results.");

DEFINE_double(confidence, 0.9999,
	"The confidence of the multi-model fitting.");
DEFINE_double(threshold, 2.0,
	"The inlier-outlier threshold for homography fitting.");
DEFINE_double(epipolar_geometry_threshold, 0.75,
	"The inlier-outlier threshold for essential matrix fitting.");
DEFINE_int32(maximum_iterations, 100,
	"The maximum round of multi-model fitting.");
DEFINE_int32(starting_hypothesis_number, 10,
	"The number of hypotheses proposed before the optimization starts.");
DEFINE_int32(added_hypothesis_number, 10,
	"The number of hypotheses proposed in each round.");
DEFINE_double(model_to_model_distance, 0.8,
	"The maximum accepted similarity in the clustering. A values in-between [0, 1].");
DEFINE_int32(minimum_point_number, 20,
	"The minimum number of points required to keep a model.");
DEFINE_bool(visualize_results, true,
	"A flag to determine if the results should be visualized.");
DEFINE_bool(estimate_essential_matrix, true,
	"A flag to determine if the essential matrix should also be estimated.");

// Typedef for the clustering in the high-dimensional consensus space
typedef clustering::density::DBScanClustering<
	progx::ModelData,
	clustering::distances::TanimotoDistance<progx::ModelData>> ClusteringMethod;

// The exact type of the multi-model fitting algorithm
typedef progx::ProgressiveXPrime<
	// The currently used clustering technique
	ClusteringMethod, 
	// Using Tanimoto distance as the model-to-model distances
	clustering::distances::TanimotoDistance<progx::ModelData>, 
	// Using MAGSAC++-based weights both in the iteratively re-weighted LSQ fitting and for the model representation in the consensus space.
	clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultHomographyEstimator, 4>, 
	// We are looking for homographies, thus the homography estimator as specified here.
	progx::utils::DefaultHomographyEstimator,
	// The sampler used for finding minimal samples
	gcransac::sampler::ProgressiveNapsacSampler<4>> ProgXPrime;

void estimateHomographies(
	const cv::Mat& correspondences_,
	const cv::Mat& sourceImage_,
	const cv::Mat& destinationImage_,
	std::vector<gcransac::Model>& models_,
	std::vector<progx::ModelData>& modelData_);

void estimateEssentialMatrix(
	const cv::Mat& correspondences_,
	const cv::Mat& normalizedCorrespondences_,
	const cv::Mat& sourceImage_,
	const cv::Mat& destinationImage_,
	const Eigen::Matrix3d& sourceIntrinsics_,
	const Eigen::Matrix3d& destinationIntrinsics_,
	Eigen::Matrix3d& essentialMatrix_);

double translationError(
	const Eigen::Vector3d& reference_translation_,
	const Eigen::Vector3d& estimated_translation_);

double rotationError(
	const Eigen::Matrix3d& reference_rotation_,
	const Eigen::Matrix3d& estimated_rotation_);

void poseAveraging(
	const std::vector<Eigen::Matrix<double, 3, 4>>& poses_,
	Eigen::Matrix<double, 3, 4>& estimatedPose_);

std::mutex writing_mutex;

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Read the images
	cv::Mat sourceImage = cv::imread(FLAGS_data_path + FLAGS_source_image_path), // The source image
		destinationImage = cv::imread(FLAGS_data_path + FLAGS_destination_image_path); // The destination image

	if (sourceImage.empty()) // Check if the source image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image \"" << FLAGS_data_path + FLAGS_source_image_path << "\"" << std::endl;
		return -1;
	}

	if (destinationImage.empty()) // Check if the destination image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image \"" << FLAGS_data_path + FLAGS_destination_image_path << "\"" << std::endl;
		return -1;
	}

	// Reading the intrinsic camera parameters
	Eigen::Matrix3d sourceIntrinsics,
		destinationIntrinsics;

	if (!gcransac::utils::loadMatrix<double, 3, 3>(
		FLAGS_data_path + FLAGS_source_intrinsics_path,
		sourceIntrinsics))
	{
		fprintf(stderr, "File '%s' containing the intrinsic camera parameters of the source image is not found.\n",
			FLAGS_data_path + FLAGS_source_intrinsics_path.c_str());
		return -1;
	}

	if (!gcransac::utils::loadMatrix<double, 3, 3>(
		FLAGS_data_path + FLAGS_destination_intrinsics_path,
		destinationIntrinsics))
	{
		fprintf(stderr, "File '%s' containing the intrinsic camera parameters of the destination image is not found.\n",
			FLAGS_data_path + FLAGS_destination_intrinsics_path.c_str());
		return -1;
	}

	// Reading the intrinsic camera parameters
	Eigen::Matrix<double, 4, 3> sourcePose,
		destinationPose;

	if (!gcransac::utils::loadMatrix<double, 4, 3>(
		FLAGS_data_path + FLAGS_source_pose_path,
		sourcePose))
	{
		fprintf(stderr, "File '%s' containing the pose parameters of the source image is not found.\n",
			FLAGS_data_path + FLAGS_source_pose_path.c_str());
		return -1;
	}

	if (!gcransac::utils::loadMatrix<double, 4, 3>(
		FLAGS_data_path + FLAGS_destination_pose_path,
		destinationPose))
	{
		fprintf(stderr, "File '%s' containing the pose parameters of the destination image is not found.\n",
			FLAGS_data_path + FLAGS_destination_pose_path.c_str());
		return -1;
	}

	// TODO: check this
	Eigen::Matrix3d relativeRotation =
		destinationPose.block<3, 3>(0, 0) * sourcePose.block<3, 3>(0, 0).transpose();
	Eigen::Vector3d relativeTranslation =
		destinationPose.bottomRows<1>().transpose() - sourcePose.block<3, 3>(0, 0).transpose() * sourcePose.bottomRows<1>().transpose();

	// Detecting point correspondences
	cv::Mat correspondences;

	gcransac::utils::detectFeatures(
		FLAGS_workspace_path + FLAGS_source_image_path + "_" + FLAGS_destination_image_path + ".txt", // The path where the correspondences are read from or saved to.
		sourceImage, // The source image
		destinationImage, // The destination image
		correspondences); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	if (correspondences.rows == 0)
	{
		fprintf(stderr, "No correspondences are found. There must be some problem.\n");
		return -1;
	}

	printf("%d correspondences are found.\n", correspondences.rows);

	// The found model parameters
	std::vector<gcransac::Model> homographies;
	// The statistics of the models, e.g., inliers.
	std::vector<progx::ModelData> homographyData;

	estimateHomographies(
		correspondences,
		sourceImage,
		destinationImage,
		homographies,
		homographyData);

	const size_t kHomographyNumber = homographies.size();

	if (kHomographyNumber == 0 &&
		!FLAGS_estimate_essential_matrix)
	{
		fprintf(stderr, "No homographies have been found and the essential matrix estimation is turned off. Therefore, the program terminates.");
		return -1;
	}

	// Initializing the vector consisting of the poses decomposed from the homographies and, possibly, the essential matrix
	std::vector<Eigen::Matrix<double, 3, 4>> poses;
	std::vector<std::vector<Eigen::Vector3d>> points3D;
	if (FLAGS_estimate_essential_matrix)
		poses.reserve(kHomographyNumber + 1);
	else
		poses.reserve(kHomographyNumber);

	// Decompose homographies
	for (size_t homographyIdx = 0; homographyIdx < kHomographyNumber; ++homographyIdx)
	{
		std::vector<Eigen::Vector3d> points3d;
		const Eigen::Matrix3d& homography = homographies[homographyIdx].descriptor;
		const auto& inliers = homographyData[homographyIdx].inliers;
		Eigen::Matrix3d rotation;
		Eigen::Vector3d translation,
			normal;

		pose::poseFromHomographyMatrix(
			homography,
			sourceIntrinsics,
			destinationIntrinsics,
			correspondences,
			inliers,
			rotation,
			translation,
			normal,
			points3d);

		poses.resize(poses.size() + 1);
		poses.back() << rotation, translation;
	}

	// Estimate the essential matrix if needed
	if (FLAGS_estimate_essential_matrix)
	{
		// Normalize the point coordinate by the intrinsic matrices
		cv::Mat normalizedCorrespondences(correspondences.size(), CV_64F);
		gcransac::utils::normalizeCorrespondences(correspondences,
			sourceIntrinsics,
			destinationIntrinsics,
			normalizedCorrespondences);

		Eigen::Matrix3d essentialMatrix;

		estimateEssentialMatrix(
			correspondences,
			normalizedCorrespondences,
			sourceImage,
			destinationImage,
			sourceIntrinsics,
			destinationIntrinsics,
			essentialMatrix);

		Eigen::Matrix3d rotation;
		Eigen::Vector3d translation;

		pose::getPoseFromEssentialMatrix(
			essentialMatrix,
			normalizedCorrespondences,
			rotation,
			translation);

		poses.resize(poses.size() + 1);
		poses.back() << rotation, translation;
	}

	// Do the pose averaging
	Eigen::Matrix<double, 3, 4> finalPose;
	poseAveraging(
		poses,
		finalPose);

	// Calculate the pose errors
	poses.emplace_back(finalPose);

	for (size_t poseIdx = 0; poseIdx < poses.size(); ++poseIdx)
	{
		const auto& pose = poses[poseIdx];

		double rErr = rotationError(
			relativeRotation,
			pose.block<3, 3>(0, 0));
		double tErr = translationError(
			relativeTranslation,
			pose.rightCols<1>());

		if (poseIdx < poses.size() - 1)
			printf("%d. pose | Rotation error = %f degrees | Translation error = %f degrees.\n", 
				poseIdx, rErr, tErr);
		else
			printf("Averaged pose | Rotation error = %f degrees | Translation error = %f degrees.\n",
				poseIdx, rErr, tErr);
	}

	return 0;
}

void poseAveraging(
	const std::vector<Eigen::Matrix<double, 3, 4>> &poses_,
	Eigen::Matrix<double, 3, 4> &estimatedPose_)
{
	// TODO
}

void estimateEssentialMatrix(
	const cv::Mat& correspondences_,
	const cv::Mat& normalizedCorrespondences_,
	const cv::Mat& sourceImage_,
	const cv::Mat& destinationImage_,
	const Eigen::Matrix3d &sourceIntrinsics_,
	const Eigen::Matrix3d& destinationIntrinsics_,
	Eigen::Matrix3d& essentialMatrix_)
{
	// The main sampler used in GC-RANSAC
	gcransac::sampler::UniformSampler main_sampler(&correspondences_);
	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&correspondences_);

		   // Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	gcransac::neighborhood::GridNeighborhoodGraph<4> neighborhood(&correspondences_,
		{ sourceImage_.cols / static_cast<double>(8),
			sourceImage_.rows / static_cast<double>(8),
			destinationImage_.cols / static_cast<double>(8),
			destinationImage_.rows / static_cast<double>(8) },
		8);

	// Normalize the threshold by the average of the focal lengths
	const double kNormalizedThreshold =
		FLAGS_epipolar_geometry_threshold / ((sourceIntrinsics_(0, 0) + sourceIntrinsics_(1, 1) +
			destinationIntrinsics_(0, 0) + destinationIntrinsics_(1, 1)) / 4.0);

	// Apply Graph-cut RANSAC
	gcransac::utils::DefaultEssentialMatrixEstimator estimator(
		sourceIntrinsics_,
		destinationIntrinsics_);
	std::vector<int> inliers;
	gcransac::EssentialMatrix model;

	// Initializing SPRT test
	gcransac::preemption::SPRTPreemptiveVerfication<gcransac::utils::DefaultEssentialMatrixEstimator> preemptive_verification(
		correspondences_,
		estimator,
		0.1);

	gcransac::GCRANSAC<gcransac::utils::DefaultEssentialMatrixEstimator,
		gcransac::neighborhood::GridNeighborhoodGraph<4>,
		gcransac::MSACScoringFunction<gcransac::utils::DefaultEssentialMatrixEstimator>,
		gcransac::preemption::SPRTPreemptiveVerfication<gcransac::utils::DefaultEssentialMatrixEstimator>> gcransac;
	gcransac.settings.threshold = kNormalizedThreshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
	gcransac.settings.confidence = FLAGS_confidence; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

	// Start GC-RANSAC
	gcransac.run(normalizedCorrespondences_,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model,
		preemptive_verification);

	// Get the statistics of the results
	const gcransac::utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	essentialMatrix_ = model.descriptor;

	// Print the statistics
	printf("Essential matrix estimation statistics:\n");
	printf("\tElapsed time = %f secs\n", statistics.processing_time);
	printf("\tInlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("\tApplied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("\tApplied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("\tNumber of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));
}

void estimateHomographies(
	const cv::Mat& correspondences_,
	const cv::Mat& sourceImage_,
	const cv::Mat& destinationImage_,
	std::vector<gcransac::Model>& models_,
	std::vector<progx::ModelData>& modelData_)
{
	// The main sampler is used inside the local optimization
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::sampler::ProgressiveNapsacSampler<4> sampler(&correspondences_, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		progx::utils::DefaultHomographyEstimator::sampleSize(), // The size of a minimal sample
		{ static_cast<double>(sourceImage_.cols), // The width of the source image
			static_cast<double>(sourceImage_.rows), // The height of the source image
			static_cast<double>(destinationImage_.cols), // The width of the destination image
			static_cast<double>(destinationImage_.rows) }); // The height of the destination image
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsedSeconds = end - start; // The elapsed time in seconds
	printf("P-NAPSAC initialization time = %f secs.\n", elapsedSeconds.count());

	// Setting the parameters for the multi-model fitting algorithm
	ProgXPrime progressiveXPrime;

	auto& settings = progressiveXPrime.getMutableSettings();
	settings.inlierOutlierThreshold = FLAGS_threshold;
	settings.modelDistanceThreshold = FLAGS_model_to_model_distance;
	settings.maximumIterations = FLAGS_maximum_iterations;
	settings.minimumInlierNumber = FLAGS_minimum_point_number;
	settings.startingHypothesisNumber = FLAGS_starting_hypothesis_number;
	settings.addedHypothesisNumber = FLAGS_added_hypothesis_number;
	settings.confidence = FLAGS_confidence;
	settings.print();

	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	progressiveXPrime.run(
		correspondences_,
		sampler,
		models_,
		modelData_);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsedSeconds = end - start; // The elapsed time in seconds

	const double kTime = elapsedSeconds.count();
	const size_t kModelNumber = models_.size();

	printf("Multi-homography estimation statistics:\n");
	printf("\tProcessing time = %f seconds\n", kTime);
	printf("\tNumber of models found = %d\n", kModelNumber);

	// Visualize the results if needed
	if (FLAGS_visualize_results)
	{
		printf("Visualizing the results. Note that the points can be assigned to multiple models, therefore, the color coding is not necessarily consistent.\n");
		printf("Press a key to continue.\n");

		progressiveXPrime.image1 =
			sourceImage_;
		progressiveXPrime.image2 =
			destinationImage_;

		std::vector<std::vector<size_t>> clusterIndices(kModelNumber);
		for (size_t modelIdx = 0; modelIdx < kModelNumber; ++modelIdx)
			clusterIndices[modelIdx].emplace_back(modelIdx);

		progressiveXPrime.displayClustering(
			correspondences_,
			modelData_,
			clusterIndices);
		cv::waitKey(0);
	}
}


void poseFromMultiHomographyTest()
{

	// Decomposing the homographies
	/*Eigen::Vector3d normal;
	for (size_t homographyIdx = 0; homographyIdx < kModelNumber; ++homographyIdx)
	{
		const auto& homography = models[homographyIdx];
		const auto& homographyData = modelData[homographyIdx];
		poses.resize(poses.size() + 1);
		auto& pose = poses.back();
		points3D.resize(poses.size());

		Eigen::Matrix3d& rotation = pose.block<3, 3>(0, 0);
		Eigen::Vector3d& translation = pose.rightCols<1>();

		pose::poseFromHomographyMatrix(
			homography.descriptor,
			sourceIntrinsics,
			destinationIntrinsics,
			correspondences,
			homographyData.inliers,
			rotation,
			translation,
			normal,
			points3D.back());
	}

	// Estimate the essential matrix if needed
	/*if constexpr (kEstimateEssentialMatrix)
	{
		gcransac::sampler::UniformSampler local_optimization_sampler(&correspondences); // The local optimization sampler is used inside the local optimization

		// Checking if the samplers are initialized successfully.
		if (!sampler.isInitialized() ||
			!local_optimization_sampler.isInitialized())
		{
			fprintf(stderr, "One of the samplers is not initialized successfully.\n");
			return;
		}

		gcransac::neighborhood::GridNeighborhoodGraph<4> neighborhood(&correspondences,
			{ sourceImage.cols / static_cast<double>(8),
				sourceImage.rows / static_cast<double>(8),
				destinationImage.cols / static_cast<double>(8),
				destinationImage.rows / static_cast<double>(8) },
			8);

		// Normalize the point coordinate by the intrinsic matrices
		cv::Mat normalizedCorrespondences(correspondences.size(), CV_64F);
		gcransac::utils::normalizeCorrespondences(correspondences,
			sourceIntrinsics,
			destinationIntrinsics,
			normalizedCorrespondences);

		// Normalize the threshold by the average of the focal lengths
		const double kNormalizedThreshold =
			kEssentialMatrixThreshold / ((sourceIntrinsics(0, 0) + sourceIntrinsics(1, 1) +
				destinationIntrinsics(0, 0) + destinationIntrinsics(1, 1)) / 4.0);

		// Apply Graph-cut RANSAC
		gcransac::utils::DefaultEssentialMatrixEstimator estimator(
			sourceIntrinsics,
			destinationIntrinsics);
		std::vector<int> inliers;
		gcransac::EssentialMatrix model;

		// Initializing SPRT test
		gcransac::preemption::SPRTPreemptiveVerfication<gcransac::utils::DefaultEssentialMatrixEstimator> preemptive_verification(
			correspondences,
			estimator,
			0.1);

		gcransac::GCRANSAC<gcransac::utils::DefaultEssentialMatrixEstimator,
			gcransac::neighborhood::GridNeighborhoodGraph<4>,
			gcransac::MSACScoringFunction<gcransac::utils::DefaultEssentialMatrixEstimator>,
			gcransac::preemption::SPRTPreemptiveVerfication<gcransac::utils::DefaultEssentialMatrixEstimator>> gcransac;
		gcransac.settings.threshold = kNormalizedThreshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
		gcransac.settings.confidence = kConfidence; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(normalizedCorrespondences,
			estimator,
			&sampler,
			&local_optimization_sampler,
			&neighborhood,
			model,
			preemptive_verification);

		// Get the statistics of the results
		const gcransac::utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

		// Print the statistics
		printf("Essential matrix estimation statistics:\n");
		printf("\tElapsed time = %f secs\n", statistics.processing_time);
		printf("\tInlier number = %d\n", static_cast<int>(statistics.inliers.size()));
		printf("\tApplied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
		printf("\tApplied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
		printf("\tNumber of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

		// Decompose the essential matrix
	}*/
}

double rotationError(const Eigen::Matrix3d& reference_rotation_,
	const Eigen::Matrix3d& estimated_rotation_)
{
	constexpr double radian_to_degree_multiplier = 180.0 / M_PI;

	const double trace_R_est_times_R_ref =
		(estimated_rotation_ * reference_rotation_.transpose()).trace();

	double error_cos = 0.5 * (trace_R_est_times_R_ref - 1.0);

	// Avoid invalid values due to numerical errors.
	error_cos = std::clamp(error_cos, -1.0, 1.0);

	return radian_to_degree_multiplier * std::acos(error_cos);
}

double translationError(
	const Eigen::Vector3d& reference_translation_,
	const Eigen::Vector3d& estimated_translation_)
{
	return std::acos(std::clamp(reference_translation_.normalized().dot(estimated_translation_.normalized()), -1.0, 1.0));
}