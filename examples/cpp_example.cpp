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
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>

struct stat info;

enum Problem { Homography, TwoViewMotion, RigidMotion, Pose6D };

typedef clustering::density::DBScanClustering<
	progx::ModelData,
	clustering::distances::TanimotoDistance<progx::ModelData>> ClusteringMethod;

void testMultiHomographyFitting(
	const std::string& scene_name_, // The name of the current scene 
	const std::string& source_path_, // The path of the source image
	const std::string& destination_path_, // The path of the destination image
	const std::string& input_correspondence_path_, // The path of the detected correspondences
	const std::string& output_correspondence_path_,  // The path of the correspondences saved with their labels
	const std::string& output_match_image_path_, // The path where the images with the labelings are saved
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_,
	const double minimum_point_number_,
	const bool visualize_results_,
	const bool visualize_inner_steps_,
	const bool has_detected_correspondences_ = false);

void testMultiMotionFitting(
	const std::string& scene_name_, // The name of the current scene 
	const std::string& video_path_, // The path of the source image
	const std::string& input_correspondence_path_, // The path of the detected correspondences
	const std::string& output_correspondence_path_,  // The path of the correspondences saved with their labels
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_, // A flag to determine if the results should be visualized
	const bool visualize_inner_steps_, // A flag to determine if the inner steps should be visualized.
	const bool has_detected_correspondences_);

void testMultiTwoViewMotionFitting(
	const std::string& scene_name_, // The name of the current scene 
	const std::string& source_path_, // The path of the source image
	const std::string& destination_path_, // The path of the destination image
	const std::string& input_correspondence_path_, // The path of the detected correspondences
	const std::string& output_correspondence_path_,  // The path of the correspondences saved with their labels
	const std::string& output_match_image_path_, // The path where the images with the labelings are saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_, // A flag to determine if the results should be visualized
	const bool visualize_inner_steps_, // A flag to determine if the inner steps should be visualized.
	const bool has_detected_correspondences_);

void testMulti2DLineFitting(
	const std::string& data_path_,
	const std::string& ground_truth_path_,
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_); // A flag to determine if the results should be visualized

bool initializeScene(const std::string& scene_name_,
	std::string& src_image_path_,
	std::string& dst_image_path_,
	std::string& input_correspondence_path_,
	std::string& output_correspondence_path_,
	std::string& output_matched_image_path_,
	const std::string& root_directory = "", // The root directory where the "results" and "data" folder are
	const bool has_detected_correspondences_ = false); // Determine if the correspondences are stored under the data folder or should be detected later

double rotationError(const Eigen::Matrix3d& reference_rotation_,
	const Eigen::Matrix3d& estimated_rotation_);

void poseFromMultiHomographyTest();

void drawMatches(
	const cv::Mat& points_,
	const std::vector<size_t>& inliers_,
	const cv::Mat& image_src_,
	const cv::Mat& image_dst_,
	cv::Mat& out_image_,
	int circle_radius_,
	const cv::Scalar& color_);

std::mutex writing_mutex;
int settings_number = 0;
constexpr bool runTestHomography = true;
constexpr bool runTestTwoViewMotion = false;
constexpr bool runTestMotion = false;

std::vector<std::string> getAvailableTestScenes(const Problem& problem_);

int main(int argc, const char* argv[])
{
	const std::string root_directory = ""; // The directory where the 'data' folder is found

	const double confidence = 0.99,
		spatial_coherence_weight = 0.1,
		neighborhood_ball_radius = 20,
		maximum_tanimoto_similarity = 0.8;

	const bool visualize_results = true, // A flag to tell if the resulting labeling should be visualized
		visualize_inner_steps = false; // A flag to tell if the steps of the algorithm should be visualized

	if constexpr (runTestMotion)
	{
		const std::vector<double> thresholds = { 0.0004, 0.0006, 0.0008, 0.001, 0.0025, 0.075, 0.01, 0.0125, 0.025, 0.05, 0.1, 0.5, 1.0 };
		const std::vector<double> tanimotoDistances = { 0.75, 0.8, 0.85, 0.90 };
		const std::vector<int> minimumPoints = { 5, 10, 15, 20 };
		const std::vector<double> confidences = { /*0.9, 0.95, 0.99, 0.999,*/ 0.9999/*, 0.99999*/ };
		const std::vector<int> maximumIterations = { 10, 20, 50, 100, 150, 200 };
		const std::vector<int> startingHypothesisNumber = { 10 /*, 5, 1, 25 /*, 50, 100, 150, 200, 250, 300, 500*/ };
		const std::vector<int> addedHypothesisNumber = { 1, 2, 5, 10 /*, 5, 1, 25 /*, 50, 100, 150, 200, 250, 300, 500*/ };
		int repetitions = 3;

		std::vector<progx::MultiModelSettings> settings;

		for (int rep = 0; rep < repetitions; ++rep)
			for (const auto& minimum_point_number : minimumPoints)
				for (const auto& maxIters : maximumIterations)
					for (const auto& threshold : thresholds)
						for (const auto& tanimoto : tanimotoDistances)
							for (const auto& confidence : confidences)
								for (const auto& startNumber : startingHypothesisNumber)
									for (const auto& addedNumber : addedHypothesisNumber)
									{
										settings.resize(settings.size() + 1);
										auto& item = settings.back();
										item.minimumInlierNumber = minimum_point_number;
										item.maximumIterations = maxIters;
										item.inlierOutlierThreshold = threshold;
										item.modelDistanceThreshold = tanimoto;
										item.confidence = confidence;
										item.startingHypothesisNumber = startNumber;
										item.addedHypothesisNumber = addedNumber;
									}

		size_t settingIdx = 0;

#pragma omp parallel for num_threads(24)
		for (int settingIdx = 0; settingIdx < settings.size(); ++settingIdx)
		{
			for (const std::string& scene : getAvailableTestScenes(Problem::RigidMotion))
			{
				printf("Processed scene = %s.\n", scene.c_str());

				std::string video_path = root_directory + "data/" + scene + "/" + scene + ".avi", // Path of the source image
					input_correspondence_path = root_directory + "data/" + scene + "/" + scene + ".txt", // Path where the detected correspondences are saved
					output_correspondence_path = "results/" + scene + "/result_" + scene + ".txt"; // Path where the inlier correspondences are saved

				const auto& currentSettings = settings[settingIdx];

				testMultiMotionFitting(
					scene, // The name of the current scene
					video_path, // The source image's path
					input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
					output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
					currentSettings.confidence, // The RANSAC confidence value
					currentSettings.inlierOutlierThreshold, // The used inlier-outlier threshold in GC-RANSAC.
					currentSettings.maximumIterations,
					currentSettings.startingHypothesisNumber,
					currentSettings.addedHypothesisNumber,
					currentSettings.modelDistanceThreshold,
					currentSettings.minimumInlierNumber, // The minimum number of inlier for a model to be kept.
					visualize_results, // A flag to determine if the results should be visualized
					visualize_inner_steps, // A flag to determine if the inner steps should be visualized.
					true);  // In this dataset, the correspondences and a reference labeling are provided				
			}

		}
	}

	if constexpr (runTestTwoViewMotion)
	{
		const std::vector<double> thresholds = { 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
		const std::vector<double> tanimotoDistances = { 0.75, 0.8, 0.85, 0.90 };
		const std::vector<int> minimumPoints = { 20, 30, 40 };
		const std::vector<double> confidences = { /*0.9, 0.95, 0.99, 0.999,*/ 0.9999/*, 0.99999*/ };
		const std::vector<int> maximumIterations = { 50, 75, 100 };
		const std::vector<int> startingHypothesisNumber = { 10 /*, 5, 1, 25 /*, 50, 100, 150, 200, 250, 300, 500*/ };
		const std::vector<int> addedHypothesisNumber = { 10 /*, 5, 1, 25 /*, 50, 100, 150, 200, 250, 300, 500*/ };
		int repetitions = 3;

		std::vector<progx::MultiModelSettings> settings;

		for (int rep = 0; rep < repetitions; ++rep)
		for (const auto& minimum_point_number : minimumPoints)
			for (const auto& maxIters : maximumIterations)
				for (const auto& threshold : thresholds)
					for (const auto& tanimoto : tanimotoDistances)
						for (const auto& confidence : confidences)
							for (const auto& startNumber : startingHypothesisNumber)
								for (const auto& addedNumber : addedHypothesisNumber)
									{
										settings.resize(settings.size() + 1);
										auto& item = settings.back();
										item.minimumInlierNumber = minimum_point_number;
										item.maximumIterations = maxIters;
										item.inlierOutlierThreshold = threshold;
										item.modelDistanceThreshold = tanimoto;
										item.confidence = confidence;
										item.startingHypothesisNumber = startNumber;
										item.addedHypothesisNumber = addedNumber;
									}

#pragma omp parallel for num_threads(30)
	for (int settingIdx = 0; settingIdx < settings.size(); ++settingIdx)
	{
		for (const std::string& scene : getAvailableTestScenes(Problem::TwoViewMotion))
		{
			const size_t minimum_point_number = 15; // The minimum number of inliers needed to accept a model instance, i.e., two times the sample size
			const double inlier_outlier_threshold = 5.0; // The inlier-outlier threshold used to assign points to models

			printf("Processed scene = %s.\n", scene.c_str());

			std::string src_image_path, // Path of the source image
				dst_image_path, // Path of the destination image
				input_correspondence_path, // Path where the detected correspondences are saved
				output_correspondence_path, // Path where the inlier correspondences are saved
				output_matched_image_path; // Path where the matched image is saved

			// Initializing the paths 
			initializeScene(scene, // The scene's name
				src_image_path, // The path of the source image
				dst_image_path, // The path of the destination image
				input_correspondence_path, // The path of the detected correspondences
				output_correspondence_path, // The path of the correspondences saved with their labels
				output_matched_image_path, // The path where the images with the labelings are saved
				root_directory, // The root directory where the "results" and "data" folder are
				true); // In this dataset, the correspondences and a reference labeling are provided

				const auto& currentSettings = settings[settingIdx];

				testMultiTwoViewMotionFitting(
					scene, // The name of the current scene
					src_image_path, // The source image's path
					dst_image_path, // The destination image's path
					input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
					output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
					output_matched_image_path, // The path where the matched image pair will be saved
					currentSettings.confidence, // The RANSAC confidence value
					currentSettings.inlierOutlierThreshold, // The used inlier-outlier threshold in GC-RANSAC.
					currentSettings.maximumIterations,
					currentSettings.startingHypothesisNumber,
					currentSettings.addedHypothesisNumber,
					currentSettings.modelDistanceThreshold,
					currentSettings.minimumInlierNumber, // The minimum number of inlier for a model to be kept.
					visualize_results, // A flag to determine if the results should be visualized
					visualize_inner_steps, // A flag to determine if the inner steps should be visualized.
					true);  // In this dataset, the correspondences and a reference labeling are provided
			}
		}
	}

	if constexpr (runTestHomography)
	{
		std::vector<progx::MultiModelSettings> settings;
		const std::vector<double> thresholds = { 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0 };
		const std::vector<double> tanimotoDistances = { 0.75, 0.8, 0.85, 0.90 };
		const std::vector<int> minimumPoints = { 20, 10, 15 };
		const std::vector<double> confidences = { /*0.9, 0.95, 0.99, 0.9, 0.95, 0.99, 0.999,*/ 0.9999/*, 0.99999*/ };
		const std::vector<int> maximumIterations = { 50, 75, 150, 100, 200 };
		const std::vector<int> startingHypothesisNumber = { 10 /*, 5, 1, 25 /*, 50, 100, 150, 200, 250, 300, 500*/ };
		const std::vector<int> addedHypothesisNumber = { 1, 5, 10 /*, 5, 1, 25 /*, 50, 100, 150, 200, 250, 300, 500*/ };
		int repetitions = 3;

		for (int rep = 0; rep < repetitions; ++rep)
			for (const auto& minimum_point_number : minimumPoints)
				for (const auto& maxIters : maximumIterations)
					for (const auto& threshold : thresholds)
						for (const auto& tanimoto : tanimotoDistances)
							for (const auto& confidence : confidences)
								for (const auto& startNumber : startingHypothesisNumber)
									for (const auto& addedNumber : addedHypothesisNumber)
									{
										settings.resize(settings.size() + 1);
										auto& item = settings.back();
										item.minimumInlierNumber = minimum_point_number;
										item.maximumIterations = maxIters;
										item.inlierOutlierThreshold = threshold;
										item.modelDistanceThreshold = tanimoto;
										item.confidence = confidence;
										item.startingHypothesisNumber = startNumber;
										item.addedHypothesisNumber = addedNumber;
									}

#pragma omp parallel for num_threads(30)
		for (int settingIdx = 0; settingIdx < settings.size(); ++settingIdx)
		{
			for (const std::string& scene : getAvailableTestScenes(Problem::Homography))
			{
				printf("Processed scene = %s.\n", scene.c_str());

				std::string src_image_path, // Path of the source image
					dst_image_path, // Path of the destination image
					input_correspondence_path, // Path where the detected correspondences are saved
					output_correspondence_path, // Path where the inlier correspondences are saved
					output_matched_image_path; // Path where the matched image is saved

				// Initializing the paths 
				initializeScene(scene, // The scene's name
					src_image_path, // The path of the source image
					dst_image_path, // The path of the destination image
					input_correspondence_path, // The path of the detected correspondences
					output_correspondence_path, // The path of the correspondences saved with their labels
					output_matched_image_path, // The path where the images with the labelings are saved
					root_directory, // The root directory where the "results" and "data" folder are
					true); // In this dataset, the correspondences and a reference labeling are provided

				const auto& currentSettings = settings[settingIdx];

				testMultiHomographyFitting(
					scene, // The name of the current scene
					src_image_path, // The source image's path
					dst_image_path, // The destination image's path
					input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
					output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
					output_matched_image_path, // The path where the matched image pair will be saved
					currentSettings.confidence, // The RANSAC confidence value
					currentSettings.inlierOutlierThreshold, // The used inlier-outlier threshold in GC-RANSAC.
					currentSettings.maximumIterations,
					currentSettings.startingHypothesisNumber,
					currentSettings.addedHypothesisNumber,
					currentSettings.modelDistanceThreshold,
					currentSettings.minimumInlierNumber, // The minimum number of inlier for a model to be kept.
					visualize_results, // A flag to determine if the results should be visualized
					visualize_inner_steps, // A flag to determine if the inner steps should be visualized.
					true);  // In this dataset, the correspondences and a reference labeling are provided
			}
		}
	}
	
	return 0;
}

std::vector<std::string> getAvailableTestScenes(const Problem& problem_)
{
	switch (problem_)
	{
	case Problem::Homography:
		/*, "boxesandbooks"*/ /*"glasscasea", "glasscaseb",*/  /*, "stairs"*/
		// Scenes from the AdelaideRMF H dataset. Correspondences are obtained by the EPOS method.
		return { "oldclassicswing", "unihouse", "unionhouse", "barrsmith", 
		 "bonhall", "bonython", "elderhalla",
		 "elderhallb", "johnssona",
		 "ladysymon", "library", "napiera",
		 "napierb", "neem", "nese", "oldclassicswing",
		 "physics", "sene", "johnssonb" };
	case Problem::TwoViewMotion:
		// Scenes from the AdelaideRMF F dataset. Correspondences are obtained by the EPOS method.
		return { "biscuitbookbox", "breadcartoychips", "breadcubechips", "breadtoycar",
			"carchipscube", "cubebreadtoychips", "dinobooks", "toycubecar",
			"biscuit", "book", "breadcube", "breadtoy",
			"cube", "cubetoy", "game", "gamebiscuit",
			"cubechips", "boardgame" };
	case Problem::Pose6D:
		// A scene from the T-LESS dataset. Correspondences are obtained by the EPOS method.
		return { "tless" };
	case Problem::RigidMotion:
		return { "cars1", "cars10_g12", "cars10_g13", "cars10_g23",
			"cars2", "cars2B_g12", "cars2B_g13", "cars2B_g23",
			"cars2_06_g12", "cars2_06_g13", "cars2_06_g23", "cars2_07_g12",
			"cars2_07_g13", "cars2_07_g23", "cars3_g12", "cars3_g13",
			"cars3_g23", "cars4", "cars5_g12", "cars5_g13",
			"cars5_g23", "cars6", "cars7", "cars8",
			"cars9_g12", "cars9_g13", "cars9_g23", "truck1",
			"truck2", "kanatani1", "kanatani2" };
	default:
		return {};
	}
}

// Initializing the paths regarding the current scene
bool initializeScene(const std::string& scene_name_, // The scene's name
	std::string& src_image_path_, // The path of the source image
	std::string& dst_image_path_, // The path of the destination image
	std::string& input_correspondence_path_, // The path of the detected correspondences
	std::string& output_correspondence_path_, // The path of the correspondences saved with their labels
	std::string& output_matched_image_path_, // The path where the images with the labelings are saved 
	const std::string& root_directory_, // The root directory where the "results" and "data" folder are
	const bool has_detected_correspondences_) // Determine if the correspondences are stored under the data folder or should be detected later
{
	// The directory to which the results will be saved
	std::string dir = "results/" + scene_name_;

	// The source image's path
	src_image_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "1.jpg";
	if (cv::imread(src_image_path_).empty())
		src_image_path_ = root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "1.png";
	if (cv::imread(src_image_path_).empty())
	{
		std::cerr << "Error while loading source image \"" <<
			src_image_path_ << "\"";
		return false;
	}

	// The destination image's path
	dst_image_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "2.jpg";
	if (cv::imread(dst_image_path_).empty())
		dst_image_path_ = root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + "2.png";
	if (cv::imread(dst_image_path_).empty())
	{
		std::cerr << "Error while loading destination image \"" <<
			dst_image_path_ << "\"";
		return false;
	}

	// The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	if (has_detected_correspondences_) // If the correspondences are provided with the dataset, load them.
		input_correspondence_path_ =
		root_directory_ + "data/" + scene_name_ + "/" + scene_name_ + ".txt";
	else // Otherwise, they will be saved under folder "results".
		input_correspondence_path_ =
		"results/" + scene_name_ + "/" + scene_name_ + "_points_with_no_annotation.txt";
	// The path where the inliers of the estimated fundamental matrices will be saved
	output_correspondence_path_ =
		"results/" + scene_name_ + "/result_" + scene_name_ + ".txt";
	// The path where the matched image pair will be saved
	output_matched_image_path_ =
		"results/" + scene_name_ + "/matches_" + scene_name_ + ".png";

	return true;
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

void poseFromMultiHomographyTest()
{
	const std::string kSourceImagePath = "",
		kDestinationImagePath = "",
		kWorkspacePath = "";

	constexpr bool kVisualizeResult = true,
		kEstimateEssentialMatrix = true;
	constexpr double kInlierOutlierThreshold = 2.0,
		kEssentialMatrixThreshold = 0.75,
		kModelDistanceThreshold = 0.80,
		kConfidence = 0.99;
	constexpr size_t kMaximumIterations = 100,
		kMinimumInlierNumber = 20,
		kStartingHypothesisNumber = 10,
		kAddedHypothesisNumber = 10;

	// Read the images
	cv::Mat sourceImage = cv::imread(kSourceImagePath), // The source image
		destinationImage = cv::imread(kDestinationImagePath); // The destination image

	if (sourceImage.empty()) // Check if the source image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image \"" << kSourceImagePath << "\"" << std::endl;
		return;
	}

	if (destinationImage.empty()) // Check if the destination image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image \"" << kDestinationImagePath << "\"" << std::endl;
		return;
	}

	// Reading the intrinsic camera parameters
	Eigen::Matrix3d sourceIntrinsics,
		destinationIntrinsics;

	// Detecting point correspondences
	cv::Mat correspondences;

	gcransac::utils::detectFeatures(
		kWorkspacePath, // The path where the correspondences are read from or saved to.
		sourceImage, // The source image
		destinationImage, // The destination image
		correspondences); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"
	printf("%d correspondences are found.\n", correspondences.rows);

	// The main sampler is used inside the local optimization
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::sampler::ProgressiveNapsacSampler<4> sampler(&correspondences, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		progx::utils::DefaultHomographyEstimator::sampleSize(), // The size of a minimal sample
		{ static_cast<double>(sourceImage.cols), // The width of the source image
			static_cast<double>(sourceImage.rows), // The height of the source image
			static_cast<double>(destinationImage.cols), // The width of the destination image
			static_cast<double>(destinationImage.rows) }); // The height of the destination image
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsedSeconds = end - start; // The elapsed time in seconds
	printf("P-NAPSAC initialization time = %f secs.\n", elapsedSeconds.count());

	// Applying Progressive-X*
	typedef progx::ProgressiveXPrime<
		ClusteringMethod,
		clustering::distances::TanimotoDistance<progx::ModelData>,
		clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultHomographyEstimator, 4>,
		progx::utils::DefaultHomographyEstimator,
		gcransac::sampler::ProgressiveNapsacSampler<4>> ProgXPrime;

	ProgXPrime progressiveXPrime;

	auto& settings = progressiveXPrime.getMutableSettings();
	settings.inlierOutlierThreshold = kInlierOutlierThreshold;
	settings.modelDistanceThreshold = kModelDistanceThreshold;
	settings.maximumIterations = kMaximumIterations;
	settings.minimumInlierNumber = kMinimumInlierNumber;
	settings.startingHypothesisNumber = kStartingHypothesisNumber;
	settings.addedHypothesisNumber = kAddedHypothesisNumber;
	settings.confidence = kConfidence;

	std::vector<gcransac::Model> models;
	std::vector<progx::ModelData> modelData;

	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	progressiveXPrime.run(
		correspondences,
		sampler,
		models,
		modelData);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsedSeconds = end - start; // The elapsed time in seconds
	
	const double kTime = elapsedSeconds.count();
	const size_t kModelNumber = models.size();

	printf("Multi-homography estimation statistics:\n");
	printf("\tNumber of models found = %d.\n", kModelNumber);
	printf("\tProcessing time = %f secs.\n", kTime);

	// Visualize the results if needed
	if constexpr (kVisualizeResult)
	{
		printf("Visualizing the results. Note that the points can be assigned to multiple models, therefore, the color coding is not necessarily consistent.\n");
		printf("Press a key to continue.\n");

		progressiveXPrime.image1 =
			sourceImage;
		progressiveXPrime.image2 =
			destinationImage;

		std::vector<std::vector<size_t>> clusterIndices(kModelNumber);
		for (size_t modelIdx = 0; modelIdx < kModelNumber; ++modelIdx)
			clusterIndices[modelIdx].emplace_back(modelIdx);
 
		progressiveXPrime.displayClustering(
			correspondences,
			modelData,
			clusterIndices);
	}

	// Initializing the vector consisting of the poses decomposed from the homographies and, possibly, the essential matrix
	std::vector<Eigen::Matrix<double, 3, 4>> poses;
	std::vector<std::vector<Eigen::Vector3d>> points3D;
	if constexpr (kEstimateEssentialMatrix)
		poses.reserve(kModelNumber + 1);
	else
		poses.reserve(kModelNumber);

	// Decomposing the homographies
	Eigen::Vector3d normal;
	for (size_t homographyIdx = 0; homographyIdx < kModelNumber; ++homographyIdx)
	{
		/*const auto& homography = models[homographyIdx];
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
			points3D.back());*/
	}

	// Estimate the essential matrix if needed
	if constexpr (kEstimateEssentialMatrix)
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
	}
}

void testMulti2DLineFitting(
	const std::string &data_path_,
	const std::string &ground_truth_path_,
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_) // A flag to determine if the results should be visualized
{
	static std::mutex savingMutex;

	// Load the point coordinates
	cv::Mat points;
	gcransac::utils::loadPointsFromFile<2, 1, false>(
		points,
		data_path_.c_str());

	// Loading the ground thruth polygon's coordinates
	cv::Mat groundTruthPolygon;
	gcransac::utils::loadPointsFromFile<2, 1, false>(
		groundTruthPolygon,
		ground_truth_path_.c_str());

	// The main sampler is used inside the local optimization
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::sampler::ProgressiveNapsacSampler<2> sampler(&points, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		gcransac::utils::Default2DLineEstimator::sampleSize(), // The size of a minimal sample
		{ static_cast<double>(256), // The width of the source image
			static_cast<double>(256) }); // The height of the source image
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("P-NAPSAC initialization time = %f secs.\n", elapsed_seconds.count());

	gcransac::sampler::UniformSampler uniform_sampler(&points);

	// Applying Progressive-X
	typedef progx::ProgressiveXPrime<
		clustering::density::MeanShiftClustering<
			progx::ModelData,
			clustering::distances::TanimotoDistance<progx::ModelData>>,
		clustering::distances::TanimotoDistance<progx::ModelData>,
		clustering::losses::MAGSACLoss<double, gcransac::utils::Default2DLineEstimator, 2>,
		gcransac::utils::Default2DLineEstimator,
		gcransac::sampler::UniformSampler> ProgXPrime;

	ProgXPrime progressiveXPrime;

	auto& settings = progressiveXPrime.getMutableSettings();
	settings.inlierOutlierThreshold = inlier_outlier_threshold_;
	settings.modelDistanceThreshold = maximum_tanimoto_similarity_;
	settings.maximumIterations = maximum_iterations;
	settings.minimumInlierNumber = minimum_point_number_;
	settings.startingHypothesisNumber = starting_hypothesis_number_;
	settings.addedHypothesisNumber = added_hypothesis_number_;
	settings.confidence = confidence_;

	std::vector<gcransac::Model> models;
	std::vector<progx::ModelData> modelData;

	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	progressiveXPrime.run(
		points,
		uniform_sampler,
		models,
		modelData);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsed_seconds = end - start; // The elapsed time in seconds
	double time = elapsed_seconds.count();

	// Calculate the end points of the line segments
	std::vector<Eigen::Vector4d> lineSegments(models.size());
	for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
	{
		const double
			& a = models[modelIdx].descriptor(0),
			& b = models[modelIdx].descriptor(1),
			& c = models[modelIdx].descriptor(2);

		Eigen::Vector2d lineTangent;
		lineTangent << b, -a;

		Eigen::Vector2d pointOnLine;
		pointOnLine << 0, -c / b;

		const auto& inliers = modelData[modelIdx].inliers;

		if (inliers.size() < minimum_point_number_)
			continue;

		auto& lineSegment = lineSegments[modelIdx];
		double minParameter = std::numeric_limits<double>::max(),
			maxParameter = std::numeric_limits<double>::lowest(),
			parameter;
		Eigen::Vector2d point;

		// Projecting the point to the line
		Eigen::Matrix3d coefficients;
		coefficients << 2, 0, a,
			0, 2, b,
			a, b, 0;
		const Eigen::Matrix3d coefficientsTransposed =
			coefficients.transpose();
		const Eigen::Matrix3d covariance =
			coefficients.transpose() * coefficients;
		Eigen::Vector3d inhomogeneousPart;
		inhomogeneousPart(2) = -c;

		for (const auto& inlierIdx : inliers)
		{
			inhomogeneousPart(0) = 2 * points.at<double>(inlierIdx, 0);
			inhomogeneousPart(1) = 2 * points.at<double>(inlierIdx, 1);

			point = covariance.llt().solve(coefficientsTransposed * inhomogeneousPart).head<2>();
			parameter = (point(0) - pointOnLine(0)) / lineTangent(0);

			if (parameter < minParameter)
			{
				minParameter = parameter;
				lineSegment(0) = point(1);
				lineSegment(1) = point(0);
			}

			if (parameter > maxParameter)
			{
				maxParameter = parameter;
				lineSegment(2) = point(1);
				lineSegment(3) = point(0);
			}
		}
	}

	if (models.size() > 0)
	{
		/*models.resize(2);
		lineSegments.resize(2);

		lineSegments[0] <<
			58.000359395945750,  38.998678677808194,
			93.799802642640543,  48.736027253919126;
		lineSegments[1] <<
			100 * 0.912948081320132, 100 * 0.468980685570082,
			100 * 1.048219253051902, 100 * 0.424575343525450;
		groundTruthPolygon = (cv::Mat_<double>(8, 2) <<
			102,   43,
			96,    45,
			90,    47,
			85,    46,
			79,    44,
			73,    42,
			68,    41,
			62,    39);*/

		// Calculate the error from the ground truth
		Eigen::VectorXd distances(groundTruthPolygon.rows);
		for (size_t pointIdx = 0; pointIdx < groundTruthPolygon.rows; ++pointIdx)
			distances(pointIdx) = std::numeric_limits<double>::max();

		for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
		{
			const auto& lineSegment = lineSegments[modelIdx];

			Eigen::Vector3d v1, v2;
			v1 << lineSegment.head<2>(), 1;
			v2 << lineSegment.tail<2>(), 1;

			double d = 0;

			for (size_t segmentIdx = 0; segmentIdx < groundTruthPolygon.rows; ++segmentIdx)
			{
				Eigen::Vector3d gtSegment;
				gtSegment(0) = groundTruthPolygon.at<double>(segmentIdx, 0);
				gtSegment(1) = groundTruthPolygon.at<double>(segmentIdx, 1);
				gtSegment(2) = 1;

				Eigen::Vector3d
					a = v1 - v2,
					b = gtSegment - v2,
					c = gtSegment - v1;

				double D1 = abs(a.cross(b)(2)) / sqrt(a(0) * a(0) + a(1) * a(1));
				double D2 = sqrt(c(0) * c(0) + c(1) * c(1));
				double D3 = sqrt(b(0) * b(0) + b(1) * b(1));

				bool insegment = a.dot(b) * (-a).dot(c) >= 0;
				if (insegment)
					d = D1;
				else
					d = MIN(D2, D3);

				distances(segmentIdx) = 
					MIN(distances(segmentIdx), d);
			}
		}

		const double meanError = distances.mean();
		printf("The mean error is %f px.\n", meanError);
		printf("The processing time is %f secs.\n", time);

		savingMutex.lock();
		std::ofstream file("lineSegmentTuningMeanShiftMAGSACUniform.csv", std::fstream::app);
		file <<
			inlier_outlier_threshold_ << ";" <<
			maximum_tanimoto_similarity_ << ";" <<
			maximum_iterations << ";" <<
			minimum_point_number_ << ";" <<
			starting_hypothesis_number_ << ";" <<
			added_hypothesis_number_ << ";" <<
			confidence_ << ";" <<
			meanError << ";" <<
			time << ";" <<
			models.size() << "\n";
		savingMutex.unlock();
	}
	else
	{
		savingMutex.lock();
		std::ofstream file("lineSegmentTuningMeanShiftMAGSACUniform.csv", std::fstream::app);
		file <<
			inlier_outlier_threshold_ << ";" <<
			maximum_tanimoto_similarity_ << ";" <<
			maximum_iterations << ";" <<
			minimum_point_number_ << ";" <<
			starting_hypothesis_number_ << ";" <<
			added_hypothesis_number_ << ";" <<
			confidence_ << ";" <<
			std::numeric_limits<double>::max() << ";" <<
			time << ";" <<
			models.size() << "\n";
		savingMutex.unlock();
	}

	if (visualize_results_)
	{
		cv::Mat image = cv::Mat::zeros(256, 256, CV_8UC3);

		for (size_t pointIdx = 0; pointIdx < points.rows; ++pointIdx)
			cv::circle(image,
				(cv::Point2d)points.row(pointIdx),
				1,
				cv::Scalar(255, 255, 255),
				-1);

		for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
		{
			printf("Model index = %d\n", modelIdx);
			cv::Mat tmpImage = image.clone();

			cv::Scalar color(rand() / (double)RAND_MAX * 255.0,
				rand() / (double)RAND_MAX * 255.0,
				rand() / (double)RAND_MAX * 255.0);

			for (const auto& inlierIdx : modelData[modelIdx].inliers)
				cv::circle(tmpImage,
					(cv::Point2d)points.row(inlierIdx),
					1,
					color,
					-1);

			// Draw the line segments
			cv::line(
				tmpImage,
				cv::Point(lineSegments[modelIdx](0), lineSegments[modelIdx](1)),
				cv::Point(lineSegments[modelIdx](2), lineSegments[modelIdx](3)),
				cv::Scalar(0, 255, 0),
				1);

			const double
				& a = models[modelIdx].descriptor(0),
				& b = models[modelIdx].descriptor(1),
				& c = models[modelIdx].descriptor(2);

			cv::line(
				tmpImage,
				cv::Point(0, -c / b),
				cv::Point(255, (-255 * a - c) / b),
				cv::Scalar(255, 0, 0),
				1);

			cv::resize(tmpImage, tmpImage, cv::Size(tmpImage.cols * 3, tmpImage.rows * 3));
			cv::imshow("Image", tmpImage);
			cv::waitKey(0);
		}

	}
}

void testMultiMotionFitting(
	const std::string& scene_name_, // The name of the current scene 
	const std::string& video_path_, // The path of the source image
	const std::string& input_correspondence_path_, // The path of the detected correspondences
	const std::string& output_correspondence_path_,  // The path of the correspondences saved with their labels
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_, // A flag to determine if the results should be visualized
	const bool visualize_inner_steps_, // A flag to determine if the inner steps should be visualized.
	const bool has_detected_correspondences_)
{
	static std::mutex saving_mutex;

	// Read the images
	//cv::Mat source_image = cv::imread(source_path_); // The source image
	//cv::Mat destination_image = cv::imread(destination_path_); // The destination image

	// Read point tracks
	cv::Mat originalPoints, points;
	std::vector<int> reference_labeling;
	int frames = 0;
	int gt_model_number = 0;
	readAnnotatedPointSequence(input_correspondence_path_, frames, originalPoints, reference_labeling);
	projectDataToRDimensionalSpace(originalPoints, points, 5);
	points = points.t();

	size_t reference_model_number = 0;
	for (const auto& label : reference_labeling)
		reference_model_number = MAX(reference_model_number, label);


	if (true) {
		// The main sampler is used inside the local optimization
		std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
		start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
		gcransac::sampler::UniformSampler sampler(&points); 
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
		printf("Uniform sampler initialization time = %f secs.\n", elapsed_seconds.count());

		// Applying Progressive-X
		typedef progx::ProgressiveXPrime<
			clustering::density::DBScanClustering<
			progx::ModelData,
			clustering::distances::TanimotoDistance<progx::ModelData>>,
			clustering::distances::TanimotoDistance<progx::ModelData>,
			clustering::losses::MAGSACLoss<double, progx::utils::DefaultLinearSubspaceEstimator, 4>,
			progx::utils::DefaultLinearSubspaceEstimator,
			gcransac::sampler::UniformSampler> ProgXPrime;

		ProgXPrime progressiveXPrime;

		auto& settings = progressiveXPrime.getMutableSettings();
		settings.inlierOutlierThreshold = inlier_outlier_threshold_;
		settings.modelDistanceThreshold = maximum_tanimoto_similarity_;
		settings.maximumIterations = maximum_iterations;
		settings.minimumInlierNumber = minimum_point_number_;
		settings.startingHypothesisNumber = starting_hypothesis_number_;
		settings.addedHypothesisNumber = added_hypothesis_number_;
		settings.confidence = confidence_;

		std::vector<gcransac::Model> models;
		std::vector<progx::ModelData> modelData;

		start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
		progressiveXPrime.run(
			points,
			sampler,
			models,
			modelData);
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		elapsed_seconds = end - start; // The elapsed time in seconds
		double time = elapsed_seconds.count();

		progx::utils::DefaultLinearSubspaceEstimator estimator;
		for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
		{
			size_t inlierNums = 0;
			for (size_t pointIdx = 0; pointIdx < points.rows; ++pointIdx)
			{
				const double distance =
					estimator.squaredResidual(
						points.row(pointIdx),
						models[modelIdx]);

				if (distance < inlier_outlier_threshold_ * inlier_outlier_threshold_)
				{
					++inlierNums;
				}
			}

			std::cout << "Inliers = " << inlierNums << std::endl;
		}

		// Calculate the misclassification error if a reference labeling is known
		double misclassification_error1 = getMisclassificationError<progx::utils::DefaultLinearSubspaceEstimator>(
			modelData,
			reference_labeling,
			modelData.size(),
			reference_model_number);

		// Get a labeling
		for (double spatialWeight = 0.0; spatialWeight <= 1.0; spatialWeight += 0.1)
			for (double labelCost = 0.0; labelCost <= 30.0; labelCost += 5)
			{
				std::vector<size_t> labels;
				std::vector<int> intLabels;
				size_t max_label;
				getLabeling<progx::utils::DefaultLinearSubspaceEstimator>(
					points,
					models,
					0.01,
					inlier_outlier_threshold_,
					spatialWeight,
					labelCost,
					labels,
					max_label);
				intLabels.resize(labels.size());

				std::vector<cv::Scalar> colors(max_label + 1);
				for (auto& color : colors)
					color = cv::Scalar((double)rand() / RAND_MAX * 255, (double)rand() / RAND_MAX * 255, (double)rand() / RAND_MAX * 255);

				for (size_t pointIdx = 0; pointIdx < labels.size(); ++pointIdx)
				{
					intLabels[pointIdx] = labels[pointIdx];
					if (intLabels[pointIdx] == max_label)
						intLabels[pointIdx] = -1;

					/*cv::circle(source_image,
						cv::Point(points.at<double>(pointIdx, 0), points.at<double>(pointIdx, 1)),
						3,
						colors[labels[pointIdx]],
						-1);

					cv::circle(destination_image,
						cv::Point(points.at<double>(pointIdx, 2), points.at<double>(pointIdx, 3)),
						3,
						colors[labels[pointIdx]],
						-1);*/
				}

				double misclassification_error2 = getMisclassificationError(
					intLabels,
					reference_labeling,
					max_label,
					reference_model_number);

				double ME = (misclassification_error1 == -1 || misclassification_error2 == -1 ?
					MAX(misclassification_error1, misclassification_error2) :
					MIN(misclassification_error1, misclassification_error2));

				printf("Processing time = %f secs.\n", time);
				printf("Misclassification error <= (%f, %f)\%.\n", misclassification_error1, misclassification_error2);
				printf("Number of found model instances = %d (there are %d instances in the reference labeling).\n", modelData.size(), reference_model_number);

				saving_mutex.lock();
				std::ofstream file("tuningM.csv", std::fstream::app);
				file << scene_name_ << ";"
					<< "tanimoto" << ";"
					<< "magsac" << ";"
					<< "dbscan" << ";"
					<< spatialWeight << ";"
					<< labelCost << ";"
					<< confidence_ << ";"
					<< starting_hypothesis_number_ << ";"
					<< added_hypothesis_number_ << ";"
					<< minimum_point_number_ << ";"
					<< maximum_iterations << ";"
					<< inlier_outlier_threshold_ << ";"
					<< maximum_tanimoto_similarity_ << ";"
					<< minimum_point_number_ << ";"
					<< time << ";"
					<< ME << ";"
					<< (int)modelData.size() - (int)reference_model_number << "\n";
				file.close();
				saving_mutex.unlock();
			}

		/*cv::namedWindow("Image 1", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::namedWindow("Image 2", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::resizeWindow("Image 1", cv::Size(1024, 1024.0 / source_image.cols * source_image.rows));
		cv::resizeWindow("Image 2", cv::Size(1024, 1024.0 / destination_image.cols * destination_image.rows));
		cv::imshow("Image 1", source_image);
		cv::imshow("Image 2", destination_image);
		cv::waitKey(0);*/
	}
}

void testMultiTwoViewMotionFitting(
	const std::string& scene_name_, // The name of the current scene 
	const std::string& source_path_, // The path of the source image
	const std::string& destination_path_, // The path of the destination image
	const std::string& input_correspondence_path_, // The path of the detected correspondences
	const std::string& output_correspondence_path_,  // The path of the correspondences saved with their labels
	const std::string& output_match_image_path_, // The path where the images with the labelings are saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_, // A flag to determine if the results should be visualized
	const bool visualize_inner_steps_, // A flag to determine if the inner steps should be visualized.
	const bool has_detected_correspondences_)
{
	static std::mutex saving_mutex;

	// Read the images
	cv::Mat source_image = cv::imread(source_path_); // The source image
	cv::Mat destination_image = cv::imread(destination_path_); // The destination image

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image \"" << source_path_ << "\"" << std::endl;
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image \"" << destination_path_ << "\"" << std::endl;
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	std::vector<int> reference_labeling;
	size_t reference_model_number = 0;
	if (has_detected_correspondences_)
		loadPointsWithLabels(points,
			reference_labeling,
			reference_model_number,
			input_correspondence_path_.c_str());
	else
		gcransac::utils::detectFeatures(
			input_correspondence_path_, // The path where the correspondences are read from or saved to.
			source_image, // The source image
			destination_image, // The destination image
			points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	if (true) {
		// The main sampler is used inside the local optimization
		std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
		start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
		gcransac::sampler::ProgressiveNapsacSampler<4> sampler(&points, // All data points
			{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
			progx::utils::DefaultFundamentalMatrixEstimator::sampleSize(), // The size of a minimal sample
			{ static_cast<double>(source_image.cols), // The width of the source image
				static_cast<double>(source_image.rows), // The height of the source image
				static_cast<double>(destination_image.cols), // The width of the destination image
				static_cast<double>(destination_image.rows) }); // The height of the destination image
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
		printf("P-NAPSAC initialization time = %f secs.\n", elapsed_seconds.count());

		progx::utils::DefaultFundamentalMatrixEstimator estimator;
		gcransac::sampler::UniformSampler uniformSampler(&points);
		gcransac::sampler::ConnectedComponentSampler connectedComponentSampler(
			&points,
			estimator.sampleSize(),
			20,
			200,
			5,
			false);

		// Applying Progressive-X
		typedef progx::ProgressiveXPrime<
			clustering::density::MeanShiftClustering<
			progx::ModelData,
			clustering::distances::TanimotoDistance<progx::ModelData>>,
			clustering::distances::TanimotoDistance<progx::ModelData>,
			clustering::losses::MAGSACLoss<double, progx::utils::DefaultFundamentalMatrixEstimator, 4>,
			progx::utils::DefaultFundamentalMatrixEstimator,
			gcransac::sampler::ConnectedComponentSampler> ProgXPrime;

		ProgXPrime progressiveXPrime;

		auto& settings = progressiveXPrime.getMutableSettings();
		settings.inlierOutlierThreshold = inlier_outlier_threshold_;
		settings.modelDistanceThreshold = maximum_tanimoto_similarity_;
		settings.maximumIterations = maximum_iterations;
		settings.minimumInlierNumber = minimum_point_number_;
		settings.startingHypothesisNumber = starting_hypothesis_number_;
		settings.addedHypothesisNumber = added_hypothesis_number_;
		settings.confidence = confidence_;

		/*progressiveXPrime.image1 =
			source_image;
		progressiveXPrime.image2 =
			destination_image;*/

		std::vector<gcransac::Model> models;
		std::vector<progx::ModelData> modelData;

		start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
		progressiveXPrime.run(
			points,
			connectedComponentSampler,
			models,
			modelData);
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		elapsed_seconds = end - start; // The elapsed time in seconds
		double time = elapsed_seconds.count();

		for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
		{
			size_t inlierNums = 0;
			for (size_t pointIdx = 0; pointIdx < points.rows; ++pointIdx)
			{
				const double distance =
					estimator.squaredResidual(
						points.row(pointIdx),
						models[modelIdx]);

				if (distance < inlier_outlier_threshold_ * inlier_outlier_threshold_)
				{
					++inlierNums;
				}
			}

			std::cout << "Inliers = " << inlierNums << std::endl;
		}

		// Calculate the misclassification error if a reference labeling is known
		double misclassification_error1 = getMisclassificationError<gcransac::utils::DefaultFundamentalMatrixEstimator>(
			modelData,
			reference_labeling,
			modelData.size(),
			reference_model_number);

		// Get a labeling
		for (double spatialWeight = 0.0; spatialWeight <= 1.0; spatialWeight += 0.1)
			for (double labelCost = 0.0; labelCost <= 30.0; labelCost += 5)
			{
				std::vector<size_t> labels;
				std::vector<int> intLabels;
				size_t max_label;
				getLabeling<gcransac::utils::DefaultFundamentalMatrixEstimator>(
					points,
					models,
					20.0,
					inlier_outlier_threshold_,
					spatialWeight,
					labelCost,
					labels,
					max_label);
				intLabels.resize(labels.size());

				std::vector<cv::Scalar> colors(max_label + 1);
				for (auto& color : colors)
					color = cv::Scalar((double)rand() / RAND_MAX * 255, (double)rand() / RAND_MAX * 255, (double)rand() / RAND_MAX * 255);

				for (size_t pointIdx = 0; pointIdx < labels.size(); ++pointIdx)
				{
					intLabels[pointIdx] = labels[pointIdx];
					if (intLabels[pointIdx] == max_label)
						intLabels[pointIdx] = -1;

					/*cv::circle(source_image,
						cv::Point(points.at<double>(pointIdx, 0), points.at<double>(pointIdx, 1)),
						3,
						colors[labels[pointIdx]],
						-1);

					cv::circle(destination_image,
						cv::Point(points.at<double>(pointIdx, 2), points.at<double>(pointIdx, 3)),
						3,
						colors[labels[pointIdx]],
						-1);*/
				}

				double misclassification_error2 = getMisclassificationError(
					intLabels,
					reference_labeling,
					max_label,
					reference_model_number);

				double ME = (misclassification_error1 == -1 || misclassification_error2 == -1 ?
					MAX(misclassification_error1, misclassification_error2) :
					MIN(misclassification_error1, misclassification_error2));

				printf("Processing time = %f secs.\n", time);
				printf("Misclassification error <= (%f, %f)\%.\n", misclassification_error1, misclassification_error2);
				printf("Number of found model instances = %d (there are %d instances in the reference labeling).\n", modelData.size(), reference_model_number);

				saving_mutex.lock();
				std::ofstream file("tuningFCC.csv", std::fstream::app);
				file << scene_name_ << ";"
					<< "tanimoto" << ";"
					<< "magsac cc" << ";"
					<< "mean-shift" << ";"
					<< spatialWeight << ";"
					<< labelCost << ";"
					<< confidence_ << ";"
					<< starting_hypothesis_number_ << ";"
					<< added_hypothesis_number_ << ";"
					<< minimum_point_number_ << ";"
					<< maximum_iterations << ";"
					<< inlier_outlier_threshold_ << ";"
					<< maximum_tanimoto_similarity_ << ";"
					<< minimum_point_number_ << ";"
					<< time << ";"
					<< ME << ";"
					<< (int)modelData.size() - (int)reference_model_number << "\n";
				file.close();
				saving_mutex.unlock();
			}

		/*cv::namedWindow("Image 1", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::namedWindow("Image 2", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::resizeWindow("Image 1", cv::Size(1024, 1024.0 / source_image.cols * source_image.rows));
		cv::resizeWindow("Image 2", cv::Size(1024, 1024.0 / destination_image.cols * destination_image.rows));
		cv::imshow("Image 1", source_image);
		cv::imshow("Image 2", destination_image);
		cv::waitKey(0);*/
	}


	if (true) {
		// The main sampler is used inside the local optimization
		std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
		start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
		gcransac::sampler::ProgressiveNapsacSampler<4> sampler(&points, // All data points
			{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
			progx::utils::DefaultFundamentalMatrixEstimator::sampleSize(), // The size of a minimal sample
			{ static_cast<double>(source_image.cols), // The width of the source image
				static_cast<double>(source_image.rows), // The height of the source image
				static_cast<double>(destination_image.cols), // The width of the destination image
				static_cast<double>(destination_image.rows) }); // The height of the destination image
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
		printf("P-NAPSAC initialization time = %f secs.\n", elapsed_seconds.count());

		progx::utils::DefaultFundamentalMatrixEstimator estimator;
		gcransac::sampler::UniformSampler uniformSampler(&points);
		gcransac::sampler::ConnectedComponentSampler connectedComponentSampler(
			&points,
			estimator.sampleSize(),
			20,
			200,
			5,
			false);

		// Applying Progressive-X
		typedef progx::ProgressiveXPrime<
			clustering::density::DBScanClustering<
			progx::ModelData,
			clustering::distances::TanimotoDistance<progx::ModelData>>,
			clustering::distances::TanimotoDistance<progx::ModelData>,
			clustering::losses::MAGSACLoss<double, progx::utils::DefaultFundamentalMatrixEstimator, 4>,
			progx::utils::DefaultFundamentalMatrixEstimator,
			gcransac::sampler::ConnectedComponentSampler> ProgXPrime;

		ProgXPrime progressiveXPrime;

		auto& settings = progressiveXPrime.getMutableSettings();
		settings.inlierOutlierThreshold = inlier_outlier_threshold_;
		settings.modelDistanceThreshold = maximum_tanimoto_similarity_;
		settings.maximumIterations = maximum_iterations;
		settings.minimumInlierNumber = minimum_point_number_;
		settings.startingHypothesisNumber = starting_hypothesis_number_;
		settings.addedHypothesisNumber = added_hypothesis_number_;
		settings.confidence = confidence_;

		/*progressiveXPrime.image1 =
			source_image;
		progressiveXPrime.image2 =
			destination_image;*/

		std::vector<gcransac::Model> models;
		std::vector<progx::ModelData> modelData;

		start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
		progressiveXPrime.run(
			points,
			connectedComponentSampler,
			models,
			modelData);
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		elapsed_seconds = end - start; // The elapsed time in seconds
		double time = elapsed_seconds.count();

		for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
		{
			size_t inlierNums = 0;
			for (size_t pointIdx = 0; pointIdx < points.rows; ++pointIdx)
			{
				const double distance =
					estimator.squaredResidual(
						points.row(pointIdx),
						models[modelIdx]);

				if (distance < inlier_outlier_threshold_ * inlier_outlier_threshold_)
				{
					++inlierNums;
				}
			}

			std::cout << "Inliers = " << inlierNums << std::endl;
		}

		// Calculate the misclassification error if a reference labeling is known
		double misclassification_error1 = getMisclassificationError<gcransac::utils::DefaultFundamentalMatrixEstimator>(
			modelData,
			reference_labeling,
			modelData.size(),
			reference_model_number);

		// Get a labeling
		for (double spatialWeight = 0.0; spatialWeight <= 1.0; spatialWeight += 0.1)
			for (double labelCost = 0.0; labelCost <= 30.0; labelCost += 5)
			{
				std::vector<size_t> labels;
				std::vector<int> intLabels;
				size_t max_label;
				getLabeling<gcransac::utils::DefaultFundamentalMatrixEstimator>(
					points,
					models,
					20.0,
					inlier_outlier_threshold_,
					spatialWeight,
					labelCost,
					labels,
					max_label);
				intLabels.resize(labels.size());

				std::vector<cv::Scalar> colors(max_label + 1);
				for (auto& color : colors)
					color = cv::Scalar((double)rand() / RAND_MAX * 255, (double)rand() / RAND_MAX * 255, (double)rand() / RAND_MAX * 255);

				for (size_t pointIdx = 0; pointIdx < labels.size(); ++pointIdx)
				{
					intLabels[pointIdx] = labels[pointIdx];
					if (intLabels[pointIdx] == max_label)
						intLabels[pointIdx] = -1;

					/*cv::circle(source_image,
						cv::Point(points.at<double>(pointIdx, 0), points.at<double>(pointIdx, 1)),
						3,
						colors[labels[pointIdx]],
						-1);

					cv::circle(destination_image,
						cv::Point(points.at<double>(pointIdx, 2), points.at<double>(pointIdx, 3)),
						3,
						colors[labels[pointIdx]],
						-1);*/
				}

				double misclassification_error2 = getMisclassificationError(
					intLabels,
					reference_labeling,
					max_label,
					reference_model_number);

				double ME = (misclassification_error1 == -1 || misclassification_error2 == -1 ?
					MAX(misclassification_error1, misclassification_error2) :
					MIN(misclassification_error1, misclassification_error2));

				printf("Processing time = %f secs.\n", time);
				printf("Misclassification error <= (%f, %f)\%.\n", misclassification_error1, misclassification_error2);
				printf("Number of found model instances = %d (there are %d instances in the reference labeling).\n", modelData.size(), reference_model_number);

				saving_mutex.lock();
				std::ofstream file("tuningFCC.csv", std::fstream::app);
				file << scene_name_ << ";"
					<< "tanimoto" << ";"
					<< "magsac cc" << ";"
					<< "dbscan" << ";"
					<< spatialWeight << ";"
					<< labelCost << ";"
					<< confidence_ << ";"
					<< starting_hypothesis_number_ << ";"
					<< added_hypothesis_number_ << ";"
					<< minimum_point_number_ << ";"
					<< maximum_iterations << ";"
					<< inlier_outlier_threshold_ << ";"
					<< maximum_tanimoto_similarity_ << ";"
					<< minimum_point_number_ << ";"
					<< time << ";"
					<< ME << ";"
					<< (int)modelData.size() - (int)reference_model_number << "\n";
				file.close();
				saving_mutex.unlock();
			}

		/*cv::namedWindow("Image 1", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::namedWindow("Image 2", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::resizeWindow("Image 1", cv::Size(1024, 1024.0 / source_image.cols * source_image.rows));
		cv::resizeWindow("Image 2", cv::Size(1024, 1024.0 / destination_image.cols * destination_image.rows));
		cv::imshow("Image 1", source_image);
		cv::imshow("Image 2", destination_image);
		cv::waitKey(0);*/
	}

	source_image.release();
	destination_image.release();
}


void testMultiHomographyFitting(
	const std::string& scene_name_, // The name of the current scene 
	const std::string& source_path_, // The path of the source image
	const std::string& destination_path_, // The path of the destination image
	const std::string& input_correspondence_path_, // The path of the detected correspondences
	const std::string& output_correspondence_path_,  // The path of the correspondences saved with their labels
	const std::string& output_match_image_path_, // The path where the images with the labelings are saved
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_, // A flag to determine if the results should be visualized
	const bool visualize_inner_steps_, // A flag to determine if the inner steps should be visualized.
	const bool has_detected_correspondences_)
{
	static std::mutex saving_mutex;

	// Read the images
	cv::Mat source_image = cv::imread(source_path_); // The source image
	cv::Mat destination_image = cv::imread(destination_path_); // The destination image

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image \"" << source_path_ << "\"" << std::endl;
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image \"" << destination_path_ << "\"" << std::endl;
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	std::vector<int> reference_labeling;
	size_t reference_model_number = 0;
	if (has_detected_correspondences_)
		loadPointsWithLabels(points,
			reference_labeling,
			reference_model_number,
			input_correspondence_path_.c_str());
	else
		gcransac::utils::detectFeatures(
			input_correspondence_path_, // The path where the correspondences are read from or saved to.
			source_image, // The source image
			destination_image, // The destination image
			points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	// The main sampler is used inside the local optimization
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::sampler::ProgressiveNapsacSampler<4> sampler(&points, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		progx::utils::DefaultHomographyEstimator::sampleSize(), // The size of a minimal sample
		{ static_cast<double>(source_image.cols), // The width of the source image
			static_cast<double>(source_image.rows), // The height of the source image
			static_cast<double>(destination_image.cols), // The width of the destination image
			static_cast<double>(destination_image.rows) }); // The height of the destination image
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("P-NAPSAC initialization time = %f secs.\n", elapsed_seconds.count());

	progx::utils::DefaultHomographyEstimator estimator;
	gcransac::sampler::UniformSampler uniformSampler(&points);
	/*gcransac::sampler::ConnectedComponentSampler connectedComponentSampler(
		&points,
		estimator.sampleSize(),
		20,
		200,
		5,
		false);*/

	// Applying Progressive-X
	typedef progx::ProgressiveXPrime<
		ClusteringMethod,
		clustering::distances::TanimotoDistance<progx::ModelData>,
		clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultHomographyEstimator, 4>,
		progx::utils::DefaultHomographyEstimator,
		gcransac::sampler::UniformSampler> ProgXPrime;

	ProgXPrime progressiveXPrime;

	auto& settings = progressiveXPrime.getMutableSettings();
	settings.inlierOutlierThreshold = inlier_outlier_threshold_;
	settings.modelDistanceThreshold = maximum_tanimoto_similarity_;
	settings.maximumIterations = maximum_iterations;
	settings.minimumInlierNumber = minimum_point_number_;
	settings.startingHypothesisNumber = starting_hypothesis_number_;
	settings.addedHypothesisNumber = added_hypothesis_number_;
	settings.confidence = confidence_;

	progressiveXPrime.image1 =
		source_image;
	progressiveXPrime.image2 =
		destination_image;

	std::vector<gcransac::Model> models;
	std::vector<progx::ModelData> modelData;

	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	progressiveXPrime.run(
		points,
		uniformSampler,
		models,
		modelData);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsed_seconds = end - start; // The elapsed time in seconds
	double time = elapsed_seconds.count();

	for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
	{
		size_t inlierNums = 0;
		for (size_t pointIdx = 0; pointIdx < points.rows; ++pointIdx)
		{
			const double distance =
				estimator.squaredResidual(
					points.row(pointIdx),
					models[modelIdx]);

			if (distance < inlier_outlier_threshold_ * inlier_outlier_threshold_)
			{
				++inlierNums;
			}
		}

		std::cout << "Inliers = " << inlierNums << std::endl;
	}

	// Calculate the misclassification error if a reference labeling is known
	double misclassification_error1 = getMisclassificationError<progx::utils::DefaultHomographyEstimator>(
		modelData,
		reference_labeling,
		modelData.size(),
		reference_model_number);

	for (double spatialWeight = 0.0; spatialWeight <= 0.5; spatialWeight += 0.1)
	{
		for (double labelCost = 0.0; labelCost <= 100; labelCost += 5)
		{
			// Get a labeling
			std::vector<size_t> labels;
			std::vector<int> intLabels;
			size_t max_label;
			getLabeling<progx::utils::DefaultHomographyEstimator>(
				points,
				models,
				20.0,
				inlier_outlier_threshold_,
				spatialWeight,
				labelCost,
				labels,
				max_label);
			intLabels.resize(labels.size());

			std::vector<cv::Scalar> colors(max_label + 1);
			for (auto& color : colors)
				color = cv::Scalar((double)rand() / RAND_MAX * 255, (double)rand() / RAND_MAX * 255, (double)rand() / RAND_MAX * 255);

			for (size_t pointIdx = 0; pointIdx < labels.size(); ++pointIdx)
			{
				intLabels[pointIdx] = labels[pointIdx];
				if (intLabels[pointIdx] == max_label)
					intLabels[pointIdx] = -1;
			}

			double misclassification_error2 = getMisclassificationError(
				intLabels,
				reference_labeling,
				max_label,
				reference_model_number);

			double ME = (misclassification_error1 == -1 || misclassification_error2 == -1 ?
				MAX(misclassification_error1, misclassification_error2) :
				MIN(misclassification_error1, misclassification_error2));

			printf("Processing time = %f secs.\n", time);
			printf("Misclassification error <= (%f, %f)\%.\n", misclassification_error1, misclassification_error2);
			printf("Number of found model instances = %d (there are %d instances in the reference labeling).\n", modelData.size(), reference_model_number);

			saving_mutex.lock();
			std::ofstream file("tuningHUniform.csv", std::fstream::app);
			file << scene_name_ << ";"
				<< "tanimoto" << ";"
				<< "magsac + uni" << ";"
				<< ClusteringMethod::getName() << ";"
				<< spatialWeight << ";"
				<< labelCost << ";"
				<< confidence_ << ";"
				<< starting_hypothesis_number_ << ";"
				<< added_hypothesis_number_ << ";"
				<< minimum_point_number_ << ";"
				<< maximum_iterations << ";"
				<< inlier_outlier_threshold_ << ";"
				<< maximum_tanimoto_similarity_ << ";"
				<< minimum_point_number_ << ";"
				<< time << ";"
				<< ME << ";"
				<< (int)modelData.size() - (int)reference_model_number << "\n";
			file.close();
			saving_mutex.unlock();
		}
	}

	/*cv::namedWindow("Image 1", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	cv::namedWindow("Image 2", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	cv::resizeWindow("Image 1", cv::Size(1024, 1024.0 / source_image.cols * source_image.rows));
	cv::resizeWindow("Image 2", cv::Size(1024, 1024.0 / destination_image.cols * destination_image.rows));
	cv::imshow("Image 1", source_image);
	cv::imshow("Image 2", destination_image);
	cv::waitKey(0);*/
}

void drawMatches(
	const cv::Mat& points_,
	const std::vector<size_t>& inliers_,
	const cv::Mat& image_src_,
	const cv::Mat& image_dst_,
	cv::Mat& out_image_,
	int circle_radius_,
	const cv::Scalar& color_)
{
	for (const auto& idx : inliers_)
	{
		cv::Point2d pt1(points_.at<double>(idx, 0),
			points_.at<double>(idx, 1));
		cv::Point2d pt2(image_dst_.cols + points_.at<double>(idx, 2),
			points_.at<double>(idx, 3));

		cv::circle(out_image_, pt1, circle_radius_, color_, static_cast<int>(circle_radius_ * 0.4));
		cv::circle(out_image_, pt2, circle_radius_, color_, static_cast<int>(circle_radius_ * 0.4));
		cv::line(out_image_, pt1, pt2, color_, 2);
	}
}