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
#include "neighborhood/grid_neighborhood_graph.h"
#include "neighborhood/flann_neighborhood_graph.h"
#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"
#include "modified_fundamental_estimator.h"
#include "solver_homography_three_point.h"
#include "modified_homography_estimator.h"
#include "subspace4_estimator.h"
#include "estimators/essential_estimator.h"
#include "preemption/preemption_sprt.h"
#include <gflags/gflags.h>

#include <ctime>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>

struct stat info;

DEFINE_string(dataset_path, "",
	"The path where the data to be tested can be found.");
DEFINE_bool(test_homography_fitting, true,
	"A flag to decide if homographies should be tuned.");
DEFINE_bool(test_two_view_motion_fitting, false,
	"A flag to decide if two-view motions (i.e., fundamental matrices) should be tuned.");
DEFINE_bool(test_motion_fitting, false,
	"A flag to decide if video motions should be tuned.");
DEFINE_bool(visualize, true,
	"A flag to decide if the results should be visualized");
DEFINE_bool(save_statistics, false,
	"A flag to decide if the statistics should be saved.");
DEFINE_int32(core_number, 1,
	"The number of cores to be used when tuning.");
DEFINE_int32(repetitions, 5,
	"The number of repetitions on each scene.");
DEFINE_int32(sampler, 1,
	"(0) Uniform sampler, (1) Progressive NAPSAC, (2) Connected Component sampler");

enum Problem { Homography, TwoViewMotion, RigidMotion, Pose6D };

typedef clustering::density::DBScanClustering<
	progx::ModelData,
	clustering::distances::TanimotoDistance<progx::ModelData>> ClusteringMethod;

template <typename _Sampler>
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

template <typename _Sampler>
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
std::string currentTime;

std::vector<std::string> getAvailableTestScenes(const Problem& problem_);

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Save the current date as a string
	currentTime = getCurrentDateAndTime("%Y_%m_%d_%H_%M_%S");

	// The directory where the 'data' folder is found
	const std::string root_directory = FLAGS_dataset_path;

	const double confidence = 0.99,
		spatial_coherence_weight = 0.1,
		neighborhood_ball_radius = 20,
		maximum_tanimoto_similarity = 0.8;

	const bool visualize_results = true, // A flag to tell if the resulting labeling should be visualized
		visualize_inner_steps = false; // A flag to tell if the steps of the algorithm should be visualized

	progx::MultiModelSettings parameters;
	parameters.minimumInlierNumber = 20;
	parameters.maximumIterations = 75;
	parameters.inlierOutlierThreshold = 2.5;
	parameters.modelDistanceThreshold = 0.85;
	parameters.confidence = confidence;
	parameters.startingHypothesisNumber = 10;
	parameters.addedHypothesisNumber = 10;

	for (int minimumInlierNumber : { 20 })
		for (double inlierOutlierThreshold : { 2.5 })
			for (double modelDistanceThreshold : {  0.85, 0.9 })
				for (const std::string& scene : getAvailableTestScenes(Problem::TwoViewMotion))
				{
					parameters.minimumInlierNumber = minimumInlierNumber;
					parameters.inlierOutlierThreshold = inlierOutlierThreshold;
					parameters.modelDistanceThreshold = modelDistanceThreshold;

					printf("Processed scene = %s.\n", scene.c_str());

					std::string src_image_path, // Path of the source image
						dst_image_path, // Path of the destination image
						input_correspondence_path, // Path where the detected correspondences are saved
						output_correspondence_path, // Path where the inlier correspondences are saved
						output_matched_image_path; // Path where the matched image is saved

					// Initializing the paths 
					if (!initializeScene(scene, // The scene's name
						src_image_path, // The path of the source image
						dst_image_path, // The path of the destination image
						input_correspondence_path, // The path of the detected correspondences
						output_correspondence_path, // The path of the correspondences saved with their labels
						output_matched_image_path, // The path where the images with the labelings are saved
						root_directory, // The root directory where the "results" and "data" folder are
						true)) // In this dataset, the correspondences and a reference labeling are provided
						continue;

					if (FLAGS_sampler == 0)
						testMultiTwoViewMotionFitting<gcransac::sampler::UniformSampler>(
							scene, // The name of the current scene
							src_image_path, // The source image's path
							dst_image_path, // The destination image's path
							input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
							output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
							output_matched_image_path, // The path where the matched image pair will be saved
							parameters.confidence, // The RANSAC confidence value
							parameters.inlierOutlierThreshold, // The used inlier-outlier threshold in GC-RANSAC.
							parameters.maximumIterations,
							parameters.startingHypothesisNumber,
							parameters.addedHypothesisNumber,
							parameters.modelDistanceThreshold,
							parameters.minimumInlierNumber, // The minimum number of inlier for a model to be kept.
							visualize_results, // A flag to determine if the results should be visualized
							visualize_inner_steps, // A flag to determine if the inner steps should be visualized.
							true);  // In this dataset, the correspondences and a reference labeling are provided
					else if (FLAGS_sampler == 1)
						testMultiTwoViewMotionFitting<gcransac::sampler::ProgressiveNapsacSampler<4>>(
							scene, // The name of the current scene
							src_image_path, // The source image's path
							dst_image_path, // The destination image's path
							input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
							output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
							output_matched_image_path, // The path where the matched image pair will be saved
							parameters.confidence, // The RANSAC confidence value
							parameters.inlierOutlierThreshold, // The used inlier-outlier threshold in GC-RANSAC.
							parameters.maximumIterations,
							parameters.startingHypothesisNumber,
							parameters.addedHypothesisNumber,
							parameters.modelDistanceThreshold,
							parameters.minimumInlierNumber, // The minimum number of inlier for a model to be kept.
							visualize_results, // A flag to determine if the results should be visualized
							visualize_inner_steps, // A flag to determine if the inner steps should be visualized.
							true);  // In this dataset, the correspondences and a reference labeling are provided
					else
						testMultiTwoViewMotionFitting<gcransac::sampler::ConnectedComponentSampler>(
							scene, // The name of the current scene
							src_image_path, // The source image's path
							dst_image_path, // The destination image's path
							input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
							output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
							output_matched_image_path, // The path where the matched image pair will be saved
							parameters.confidence, // The RANSAC confidence value
							parameters.inlierOutlierThreshold, // The used inlier-outlier threshold in GC-RANSAC.
							parameters.maximumIterations,
							parameters.startingHypothesisNumber,
							parameters.addedHypothesisNumber,
							parameters.modelDistanceThreshold,
							parameters.minimumInlierNumber, // The minimum number of inlier for a model to be kept.
							visualize_results, // A flag to determine if the results should be visualized
							visualize_inner_steps, // A flag to determine if the inner steps should be visualized.
							true);  // In this dataset, the correspondences and a reference labeling are provided
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
		return { "stairs",  "johnssona", "oldclassicswing", "johnssonb", "unihouse", "bonhall" "unionhouse", "barrsmith",
		 "bonython", "elderhalla",
		 "elderhallb", 
		 "ladysymon", "library", "napiera",
		 "napierb", "neem", "nese", "oldclassicswing",
		 "physics", "sene" };
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
		root_directory_ + scene_name_ + "/" + scene_name_ + "1.jpg";
	if (cv::imread(src_image_path_).empty())
		src_image_path_ = root_directory_ + scene_name_ + "/" + scene_name_ + "1.png";
	if (cv::imread(src_image_path_).empty())
	{
		std::cerr << "Error while loading source image \"" <<
			src_image_path_ << "\"" << std::endl;
		return false;
	}

	// The destination image's path
	dst_image_path_ =
		root_directory_ + scene_name_ + "/" + scene_name_ + "2.jpg";
	if (cv::imread(dst_image_path_).empty())
		dst_image_path_ = root_directory_ + scene_name_ + "/" + scene_name_ + "2.png";
	if (cv::imread(dst_image_path_).empty())
	{
		std::cerr << "Error while loading destination image \"" <<
			dst_image_path_ << "\"" << std::endl;
		return false;
	}

	// The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	if (has_detected_correspondences_) // If the correspondences are provided with the dataset, load them.
		input_correspondence_path_ =
		root_directory_ + scene_name_ + "/" + scene_name_ + ".txt";
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
			for (double labelCost = 0.0; labelCost <= 20.0; labelCost += 1)
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
				std::ofstream file("results/tuning_motion_" + currentTime + ".csv", std::fstream::app);
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

		/*cv::namedWindow("Image 1", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
		cv::namedWindow("Image 2", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
		cv::resizeWindow("Image 1", cv::Size(1024, 1024.0 / source_image.cols * source_image.rows));
		cv::resizeWindow("Image 2", cv::Size(1024, 1024.0 / destination_image.cols * destination_image.rows));
		cv::imshow("Image 1", source_image);
		cv::imshow("Image 2", destination_image);
		cv::waitKey(0);*/
	}
}

template <typename _Sampler>
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

	// Initializing the estimator object
	progx::utils::DefaultFundamentalMatrixEstimator estimator;

	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation

	// Initialize the sampler depending on its type
	std::unique_ptr<_Sampler> sampler;
	if constexpr (std::is_same<gcransac::sampler::ProgressiveNapsacSampler<4>, _Sampler>())
		sampler = std::unique_ptr<_Sampler>(new _Sampler(
			&points, // All data points
			{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
			progx::utils::DefaultFundamentalMatrixEstimator::sampleSize(), // The size of a minimal sample
			{ static_cast<double>(source_image.cols), // The width of the source image
				static_cast<double>(source_image.rows), // The height of the source image
				static_cast<double>(destination_image.cols), // The width of the destination image
				static_cast<double>(destination_image.rows) }));
	else if constexpr (std::is_same<gcransac::sampler::UniformSampler, _Sampler>())
		sampler = std::unique_ptr<_Sampler>(new _Sampler(&points));
	else
		sampler = std::unique_ptr<_Sampler>(new _Sampler(&points,
			estimator.sampleSize(),
			20,
			200,
			5,
			false));
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Sampler initialization time = %f secs.\n", elapsed_seconds.count());

	// Applying Progressive-X
	typedef progx::ProgressiveXPrime<
		clustering::density::DBScanClustering<
		progx::ModelData,
		clustering::distances::TanimotoDistance<progx::ModelData>>,
		clustering::distances::TanimotoDistance<progx::ModelData>,
		clustering::losses::MAGSACLoss<double, progx::utils::DefaultFundamentalMatrixEstimator, 4>,
		progx::utils::DefaultFundamentalMatrixEstimator,
		_Sampler> ProgXPrime;

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
		*sampler,
		models,
		modelData);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsed_seconds = end - start; // The elapsed time in seconds
	double time = elapsed_seconds.count();

	// Get a labeling
	double spatialWeight = 0.0;
	double labelCost = 15.0;

	std::vector<size_t> labels;
	std::vector<int> intLabels;
	size_t max_label;
	getLabeling<progx::utils::DefaultFundamentalMatrixEstimator>(
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

	colors[0] = cv::Scalar(255, 0, 0);
	if (colors.size() > 2)
		colors[1] = cv::Scalar(0, 255, 0);
	if (colors.size() > 3)
		colors[2] = cv::Scalar(0, 0, 255);
	if (colors.size() > 4)
		colors[3] = cv::Scalar(255, 0, 255);
	if (colors.size() > 5)
		colors[4] = cv::Scalar(0, 255, 255);
	colors.back() = cv::Scalar(0, 0, 0);
	
	std::unordered_map<size_t, std::vector<size_t>> label_to_inliers;
	label_to_inliers.reserve(max_label + 1);
 
	for (size_t pointIdx = 0; pointIdx < labels.size(); ++pointIdx)
	{
		intLabels[pointIdx] = labels[pointIdx];
		if (intLabels[pointIdx] == max_label)
			intLabels[pointIdx] = -1;

		cv::circle(source_image,
			cv::Point(points.at<double>(pointIdx, 0), points.at<double>(pointIdx, 1)),
			6,
			colors[labels[pointIdx]],
			-1);

		cv::circle(destination_image,
			cv::Point(points.at<double>(pointIdx, 2), points.at<double>(pointIdx, 3)),
			6,
			colors[labels[pointIdx]],
			-1);

		// Storing the inliers of the current model after labeling
		auto& inliers = label_to_inliers[labels[pointIdx]];
		inliers.emplace_back(pointIdx);
	}

	printf("Processing time = %f secs.\n", time);
	printf("Number of found model instances = %d.\n", modelData.size());
	printf("Number of found model instances after labeling = %d.\n", max_label);
	printf("There are %d instances in the reference labeling.\n", reference_model_number);


	cv::imwrite("results/source_" + scene_name_ + ".jpg", source_image);
	cv::imwrite("results/destination_" + scene_name_ + ".jpg", destination_image);

	cv::namedWindow("Label Image 1", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
	cv::namedWindow("Label Image 2", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
	cv::resizeWindow("Label Image 1", cv::Size(1024, 1024.0 / source_image.cols * source_image.rows));
	cv::resizeWindow("Label Image 2", cv::Size(1024, 1024.0 / destination_image.cols * destination_image.rows));
	cv::imshow("Label Image 1", source_image);
	cv::imshow("Label Image 2", destination_image);
	cv::waitKey(0);

	source_image.release();
	destination_image.release();
}

template <typename _Sampler>
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
			"An error occured while loading image '" << source_path_ << "'" << std::endl;
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		std::cerr <<
			"An error occured while loading image '" << destination_path_ << "'" << std::endl;
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

	// Initializing the estimator object
	progx::utils::DefaultHomographyEstimator estimator;

	// Initialize the sampler depending on its type
	std::unique_ptr<_Sampler> sampler;
	if constexpr (std::is_same<gcransac::sampler::ProgressiveNapsacSampler<4>, _Sampler>())
		sampler = std::unique_ptr<_Sampler>(new _Sampler(
			&points, // All data points
			{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
			progx::utils::DefaultHomographyEstimator::sampleSize(), // The size of a minimal sample
			{ static_cast<double>(source_image.cols), // The width of the source image
				static_cast<double>(source_image.rows), // The height of the source image
				static_cast<double>(destination_image.cols), // The width of the destination image
				static_cast<double>(destination_image.rows) }));
	else if constexpr (std::is_same<gcransac::sampler::UniformSampler, _Sampler>())
		sampler = std::unique_ptr<_Sampler>(new _Sampler(&points));
	else 
		sampler = std::unique_ptr<_Sampler>(new _Sampler(&points,
			estimator.sampleSize(),
			20,
			1000,
			5,
			false));

	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Sampler initialization time = %f secs.\n", elapsed_seconds.count());

	// Applying Progressive-X
	using ProgXPrime = progx::ProgressiveXPrime<
		ClusteringMethod,
		clustering::distances::TanimotoDistance<progx::ModelData>,
		clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultHomographyEstimator, 4>,
		progx::utils::DefaultHomographyEstimator,
		_Sampler>;

	ProgXPrime progressiveXPrime;

	auto& settings = progressiveXPrime.getMutableSettings();
	settings.inlierOutlierThreshold = inlier_outlier_threshold_;
	settings.modelDistanceThreshold = maximum_tanimoto_similarity_;
	settings.maximumIterations = 100;
	settings.minimumInlierNumber = minimum_point_number_;
	settings.startingHypothesisNumber = starting_hypothesis_number_;
	settings.addedHypothesisNumber = added_hypothesis_number_;
	settings.confidence = confidence_;

	std::vector<gcransac::Model> models;
	std::vector<progx::ModelData> modelData;

	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	progressiveXPrime.run(
		points,
		*sampler,
		models,
		modelData);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsed_seconds = end - start; // The elapsed time in seconds
	double time = elapsed_seconds.count();

	{
		double spatialWeight = 0.0;
		double labelCost = 15.0;
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

			colors[0] = cv::Scalar(255, 0, 0);
			if (colors.size() > 2)
				colors[1] = cv::Scalar(0, 255, 0);
			if (colors.size() > 3)
				colors[2] = cv::Scalar(0, 0, 255);
			if (colors.size() > 4)
				colors[3] = cv::Scalar(255, 0, 255);
			if (colors.size() > 5)
				colors[4] = cv::Scalar(0, 255, 255);
			if (colors.size() > 6)
				colors[5] = cv::Scalar(255, 255, 0);
			if (colors.size() > 7)
				colors[6] = cv::Scalar(168, 182, 33);
			if (colors.size() > 8)
				colors[7] = cv::Scalar(33, 119, 182);

			std::vector<size_t> inlierPerModel(0, 0);
			for (size_t pointIdx = 0; pointIdx < labels.size(); ++pointIdx)
			{
				intLabels[pointIdx] = labels[pointIdx];
				if (intLabels[pointIdx] == max_label)
					intLabels[pointIdx] = -1;
				if (inlierPerModel.size() <= labels[pointIdx])
					inlierPerModel.resize(labels[pointIdx] + 1, 0);
				++inlierPerModel[labels[pointIdx]];
			}

			std::vector<std::pair<size_t, size_t>> modelSizes;
			for (size_t modelIdx = 0; modelIdx < inlierPerModel.size(); ++modelIdx)
				modelSizes.emplace_back(std::make_pair(inlierPerModel[modelIdx], modelIdx));
			std::sort(modelSizes.rbegin(), modelSizes.rend());

			cv::Mat outImage(MAX(source_image.rows, destination_image.rows),
				source_image.cols + destination_image.cols,
				source_image.type());
			source_image.copyTo(outImage(cv::Rect(0, 0, source_image.cols, source_image.rows)));
			destination_image.copyTo(outImage(cv::Rect(source_image.cols, 0, destination_image.cols, destination_image.rows)));

			for (size_t pointIdx = 0; pointIdx < labels.size(); ++pointIdx)
			{
				const auto& label = labels[pointIdx];
				size_t colorIdx = 0;
				for (colorIdx = 0; colorIdx < modelSizes.size(); ++colorIdx)
					if (label == modelSizes[colorIdx].second)
						break;
				if (label == max_label)
				{
					colorIdx = colors.size() - 1;
					continue;

					cv::line(outImage,
						cv::Point(points.at<double>(pointIdx, 0), points.at<double>(pointIdx, 1)),
						cv::Point(source_image.cols + points.at<double>(pointIdx, 2), points.at<double>(pointIdx, 3)),
						cv::Scalar(0, 0, 0),
						2);
				}

				cv::circle(outImage,
					cv::Point(points.at<double>(pointIdx, 0), points.at<double>(pointIdx, 1)),
					15,
					colors[colorIdx],
					-1);

				cv::circle(outImage,
					cv::Point(source_image.cols + points.at<double>(pointIdx, 2), points.at<double>(pointIdx, 3)),
					15,
					colors[colorIdx],
					-1);
			}

			printf("Processing time = %f secs.\n", time);
			printf("Number of found model instances = %d (there are %d instances in the reference labeling).\n", modelData.size(), reference_model_number);

			if (FLAGS_visualize)
			{
				cv::imwrite("results/result_" + scene_name_ + ".png", outImage);

				cv::namedWindow("Image", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
				cv::resizeWindow("Image", cv::Size(1024, 1024.0 / outImage.cols * outImage.rows));
				cv::imshow("Image", outImage);
				cv::waitKey(0);
			}
		}
	}
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