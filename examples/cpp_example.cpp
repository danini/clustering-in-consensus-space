#include <vector>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Eigen>

#include "progx_utils.h"
#include "losses.h"
#include "progressive_x_prime.h"
#include "median_shift_clustering.h"
#include "mean_shift_clustering.h"
#include "dbscan_clustering.h"
#include "distances.h"
#include "utils.h"
#include "GCoptimization.h"
#include "grid_neighborhood_graph.h"
#include "flann_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "solver_homography_three_point.h"
#include "homography_estimator.h"
#include "subspace4_estimator.h"
#include "essential_estimator.h"

#include <ctime>
#include <direct.h>
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

void testMulti6DPoseFitting(
	const std::string& scene_name_, // The name of the current scene 
	const std::string& input_correspondence_path_, // The path of the detected correspondences
	const std::string& camera_intrinsics_path_, // The path of the intrinsic camera parameters
	const std::string& ground_truth_path_, // The path of the ground truth poses 
	const std::string& output_correspondence_path_,  // The path of the correspondences saved with their labels
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const double neighborhood_ball_radius_,
	const double maximum_tanimoto_similarity_,
	const size_t minimum_point_number_);

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
constexpr bool runTestHomography = false;
constexpr bool runTestTwoViewMotion = true;
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
		const std::vector<double> thresholds = { 0.75, 1.0, 1.5, 2.0, 3.0 };
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

#pragma omp parallel for num_threads(24)
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
		const std::vector<double> thresholds = { 3.0, 3.5, 4.0, 5.0, 6.0 };
		const std::vector<double> tanimotoDistances = { 0.75, 0.8, 0.85 };
		const std::vector<int> minimumPoints = { 20, 10, 15 };
		const std::vector<double> confidences = { 0.9, 0.95, 0.99, /*0.9, 0.95, 0.99, 0.999,*/ 0.9999/*, 0.99999*/ };
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

#pragma omp parallel for num_threads(24)
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

void testMulti6DPoseFitting(
	const std::string& scene_name_, // The name of the current scene 
	const std::string& input_correspondence_path_, // The path of the detected correspondences
	const std::string& camera_intrinsics_path_, // The path of the intrinsic camera parameters
	const std::string& ground_truth_path_, // The path of the ground truth poses 
	const std::string& output_correspondence_path_,  // The path of the correspondences saved with their labels
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double neighborhood_ball_radius_, // The radius of the neighborhood ball for determining the neighborhoods.
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const size_t minimum_point_number_) // The minimum number of inlier for a model to be kept.
{
	/*// Loading the 2D-3D correspondences from file
	cv::Mat points;
	gcransac::utils::loadPointsFromFile<5>(points, input_correspondence_path_.c_str());

	// Load the camera matrices
	Eigen::Matrix3d intrinsics;
	gcransac::utils::loadMatrix<double, 3, 3>(camera_intrinsics_path_, intrinsics);

	// Load the ground truth poses
	cv::Mat ground_truth_poses;
	gcransac::utils::loadPointsFromFile<12>(ground_truth_poses, ground_truth_path_.c_str());

	// Normalize the correspondences by the intrinsic camera matrices
	cv::Mat normalized_points(points.rows, 7, CV_64F);
	gcransac::utils::normalizeImagePoints(
		points(cv::Rect(0, 0, 2, points.rows)),
		intrinsics,
		normalized_points(cv::Rect(0, 0, 2, points.rows)));

	// Copy the original 2D points to the last columns. They will be used for degeneracy testing
	points(cv::Rect(0, 0, 2, points.rows)).copyTo(normalized_points(cv::Rect(5, 0, 2, points.rows)));
	// Copy the 3D points to the matrix
	points(cv::Rect(2, 0, 3, points.rows)).copyTo(normalized_points(cv::Rect(2, 0, 3, points.rows)));

	// Normalize the threshold
	const double f =
		(intrinsics(0, 0) + intrinsics(1, 1)) / 2.0;
	const double normalized_threshold =
		inlier_outlier_threshold_ / f;

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius_); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::UniformSampler main_sampler(&points);

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultPnPEstimator, // The type of the used model estimator
		gcransac::sampler::UniformSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings& settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number_;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = normalized_threshold;
	// The required confidence in the results
	settings.setConfidence(confidence_);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity_;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight_;
	// The maximum iteration number of GC-RANSAC
	settings.proposal_engine_settings.max_iteration_number = 1000;

	progressive_x.run(normalized_points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC

	printf("Processing time = %f secs.\n", progressive_x.getStatistics().processing_time);
	printf("Number of found model instances = %d (there are %d instances in the reference labeling).\n", progressive_x.getModelNumber(), ground_truth_poses.rows);

	for (size_t pose_idx = 0; pose_idx < ground_truth_poses.rows; ++pose_idx)
	{
		const Eigen::Map < Eigen::Matrix<double, 3, 4, Eigen::RowMajor> >
			pose(ground_truth_poses.row(pose_idx).ptr<double>());
		const Eigen::Vector3d& gt_translation = pose.leftCols<1>();
		const Eigen::Matrix3d& gt_rotation = pose.block<3, 3>(0, 0);

		double best_rotation_error = std::numeric_limits<double>::max() / 2.0,
			best_translation_error = std::numeric_limits<double>::max() / 2.0;

		for (const auto& model : progressive_x.getModels())
		{
			const Eigen::Vector3d& translation = model.descriptor.leftCols<1>();
			const Eigen::Matrix3d& rotation = model.descriptor.block<3, 3>(0, 0);

			double rotation_error,
				translation_error;

			translation_error = (gt_translation - translation).norm();
			rotation_error = rotationError(gt_rotation, rotation);

			if (translation_error + rotation_error < best_rotation_error + best_translation_error)
			{
				best_translation_error = translation_error;
				best_rotation_error = rotation_error;
			}
		}

		printf("%d-th pose's error\n\tRotation error = %f\370\n\tTranslation error = %f mm\n",
			pose_idx + 1,
			best_rotation_error,
			best_translation_error);
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
			clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultLinearSubspaceEstimator>,
			gcransac::utils::DefaultLinearSubspaceEstimator,
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

		gcransac::utils::DefaultLinearSubspaceEstimator estimator;
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
		double misclassification_error1 = getMisclassificationError<gcransac::utils::DefaultLinearSubspaceEstimator>(
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
				getLabeling<gcransac::utils::DefaultLinearSubspaceEstimator>(
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
		gcransac::sampler::ProgressiveNapsacSampler sampler(&points, // All data points
			{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
			gcransac::utils::DefaultFundamentalMatrixEstimator::sampleSize(), // The size of a minimal sample
			source_image.cols, // The width of the source image
			source_image.rows, // The height of the source image
			destination_image.cols, // The width of the destination image
			destination_image.rows); // The height of the destination image
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
		printf("P-NAPSAC initialization time = %f secs.\n", elapsed_seconds.count());

		// Applying Progressive-X
		typedef progx::ProgressiveXPrime<
			clustering::density::MeanShiftClustering<
			progx::ModelData,
			clustering::distances::TanimotoDistance<progx::ModelData>>,
			clustering::distances::TanimotoDistance<progx::ModelData>,
			clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultFundamentalMatrixEstimator>,
			gcransac::utils::DefaultFundamentalMatrixEstimator,
			gcransac::sampler::ProgressiveNapsacSampler> ProgXPrime;

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
			sampler,
			models,
			modelData);
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		elapsed_seconds = end - start; // The elapsed time in seconds
		double time = elapsed_seconds.count();

		gcransac::utils::DefaultFundamentalMatrixEstimator estimator;
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
				std::ofstream file("tuningF.csv", std::fstream::app);
				file << scene_name_ << ";"
					<< "tanimoto" << ";"
					<< "magsac" << ";"
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
		gcransac::sampler::ProgressiveNapsacSampler sampler(&points, // All data points
			{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
			gcransac::utils::DefaultFundamentalMatrixEstimator::sampleSize(), // The size of a minimal sample
			source_image.cols, // The width of the source image
			source_image.rows, // The height of the source image
			destination_image.cols, // The width of the destination image
			destination_image.rows); // The height of the destination image
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
		printf("P-NAPSAC initialization time = %f secs.\n", elapsed_seconds.count());

		// Applying Progressive-X
		typedef progx::ProgressiveXPrime<
			clustering::density::DBScanClustering<
			progx::ModelData,
			clustering::distances::TanimotoDistance<progx::ModelData>>,
			clustering::distances::TanimotoDistance<progx::ModelData>,
			clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultFundamentalMatrixEstimator>,
			gcransac::utils::DefaultFundamentalMatrixEstimator,
			gcransac::sampler::ProgressiveNapsacSampler> ProgXPrime;

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
			sampler,
			models,
			modelData);
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		elapsed_seconds = end - start; // The elapsed time in seconds
		double time = elapsed_seconds.count();

		gcransac::utils::DefaultFundamentalMatrixEstimator estimator;
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

				// Calculate the misclassification error if a reference labeling is known

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
				std::ofstream file("tuningF.csv", std::fstream::app);
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
	gcransac::sampler::ProgressiveNapsacSampler sampler(&points, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		gcransac::utils::DefaultHomographyEstimator::sampleSize(), // The size of a minimal sample
		source_image.cols, // The width of the source image
		source_image.rows, // The height of the source image
		destination_image.cols, // The width of the destination image
		destination_image.rows); // The height of the destination image
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("P-NAPSAC initialization time = %f secs.\n", elapsed_seconds.count());

	// Applying Progressive-X
	typedef progx::ProgressiveXPrime<
		ClusteringMethod,
		clustering::distances::TanimotoDistance<progx::ModelData>,
		clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultHomographyEstimator>,
		gcransac::utils::DefaultHomographyEstimator,
		gcransac::sampler::ProgressiveNapsacSampler> ProgXPrime;

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
		sampler,
		models,
		modelData);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsed_seconds = end - start; // The elapsed time in seconds
	double time = elapsed_seconds.count();

	gcransac::utils::DefaultHomographyEstimator estimator;
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
	double misclassification_error1 = getMisclassificationError<gcransac::utils::DefaultHomographyEstimator>(
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
			getLabeling<gcransac::utils::DefaultHomographyEstimator>(
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
			std::ofstream file("tuning.csv", std::fstream::app);
			file << scene_name_ << ";"
				<< "tanimoto" << ";"
				<< "magsac" << ";"
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