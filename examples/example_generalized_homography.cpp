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
#include "uniform_generalized_sampler.h"
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
#include "generalized_homography_estimator.h"
#include "solver_generalized_homography_ceres.h"
#include "solver_generalized_homography_three_two_point.h"

#include <ctime>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>

#include <gflags/gflags.h>
#include <sophus/se3.hpp>

//#include "pose_averaging/openMVG.hpp"

#include <mutex>

DEFINE_double(inlier_outlier_threshold, 0.002,
	"The inlier-outlier threshold for the robust fitting.");
DEFINE_int32(sequence_idx, -1,
	"The sequence index which is supposed to be processed. If -1, all will be processed.");
DEFINE_int32(frame_step_size, 1,
	"Step size when processing KITTI.");
DEFINE_string(image_path, "",
	"The path where the KITTI images are stored.");
DEFINE_string(reference_pose_path, "",
	"The path where the reference poses for KITTI are stored.");
DEFINE_string(workspace_path, "",
	"This is where the algorithm saves the results.");
DEFINE_double(model_to_model_distance, 0.8,
	"The maximum accepted similarity in the clustering. A values in-between [0, 1].");
DEFINE_int32(minimum_point_number, 20,
	"The minimum number of points required to keep a model.");

// Typedef for the clustering in the high-dimensional consensus space
typedef clustering::density::DBScanClustering<
	progx::ModelData,
	clustering::distances::TanimotoDistance<progx::ModelData>> ClusteringMethod;

// The default estimator for generalized homography fitting
typedef gcransac::estimator::GeneralizedHomographyEstimator<
	gcransac::estimator::solver::GeneralizedHomographyThreeTwoPointSolver, // The solver used for fitting a model to a minimal sample
	gcransac::estimator::solver::GeneralizedHomographyCeresSolver<0>> // The solver used for fitting a model to a non-minimal sample
	GeneralizedEstimator32;

// The exact type of the multi-model fitting algorithm
typedef progx::ProgressiveXPrime<
	// The currently used clustering technique
	ClusteringMethod, 
	// Using Tanimoto distance as the model-to-model distances
	clustering::distances::TanimotoDistance<progx::ModelData>, 
	// Using MAGSAC++-based weights both in the iteratively re-weighted LSQ fitting and for the model representation in the consensus space.
	clustering::losses::TukeyBisquareLoss<double>, 
	// We are looking for homographies, thus the homography estimator as specified here.
	GeneralizedEstimator32,
	// The sampler used for finding minimal samples
	gcransac::sampler::UniformGeneralizedSampler> ProgXPrime;

bool loadGroundTruthPoses(
	const std::string& posePath_,
	const std::string& sequence_,
	std::vector<Sophus::SE3d>& groundTruthPoses_);

void processFramePair(
	const std::string& imagePath_,
	const std::string& sequence_,
	const size_t& frameSource_,
	const size_t& frameDestination_,
	const Eigen::Matrix3d intrinsicsSource_,
	const Eigen::Matrix3d intrinsicsDestination_,
	const Sophus::SE3d& rigPose_,
	const std::vector<Sophus::SE3d>& groundTruthPoses_);

void loadCorrespondences(
	const std::string& imagePath_,
	const std::string& sequence_,
	const size_t& frame_1_,
	const size_t& frame_2_,
	cv::Mat& correspondencesLeftRight_,
	cv::Mat& correspondencesLeftNextLeft_,
	cv::Mat& correspondencesRightNextLeft_,
	cv::Mat* imageLeft_,
	cv::Mat* imageRight_,
	cv::Mat* imageNextLeft_);

template<class _Estimator, class ... _EstimatorTypes>
void estimateHomography(
	const std::string& sequence_,
	const size_t& frame1_,
	const size_t& frame2_,
	const cv::Mat& imageLeft_,
	const cv::Mat& imageRight_,
	const cv::Mat& imageNextLeft_,
	const cv::Mat& dataLeftRight_,
	const cv::Mat& dataLeftNextLeft_,
	const cv::Mat& dataRightNextLeft_,
	const cv::Mat& normalizedDataLeftRight_,
	const cv::Mat& normalizedDataLeftNextLeft_,
	const cv::Mat& normalizedDataRightNextLeft_,
	const Eigen::Matrix3d& rotationLeft_,
	const Eigen::Matrix3d& rotationRight_,
	const Eigen::Vector3d& translationLeft_,
	const Eigen::Vector3d& translationRight_,
	const Sophus::SE3d& groundTruthPose_,
	const double& inlierOutlierThreshold_);

void processKITTI();

std::mutex writing_mutex;

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	processKITTI();

	return 0;
}

void processFramePair(
	const std::string& imagePath_,
	const std::string& sequence_,
	const size_t& frameSource_,
	const size_t& frameDestination_,
	const Eigen::Matrix3d intrinsicsSource_,
	const Eigen::Matrix3d intrinsicsDestination_,
	const Sophus::SE3d &rigPose_,
	const std::vector<Sophus::SE3d>& groundTruthPoses_)
{
	printf("Process frames %d -> %d.\n", frameSource_, frameDestination_);

	const Eigen::Matrix3d inverseInstrinsicsSource =
		intrinsicsSource_.inverse();
	const Eigen::Matrix3d inverseInstrinsicsDestination =
		intrinsicsDestination_.inverse();

	const Eigen::Vector3d &translationLeftRight =
		rigPose_.translation();
	Eigen::Matrix3d translationLeftRightCross;
	translationLeftRightCross <<
		0, translationLeftRight(2), -translationLeftRight(1),
		-translationLeftRight(2), 0, translationLeftRight(0),
		translationLeftRight(1), -translationLeftRight(0), 0;

	const Eigen::Matrix3d essentialMatrix =
		rigPose_.rotationMatrix() * translationLeftRightCross;
	const Eigen::Matrix3d fundamentalMatrix =
		inverseInstrinsicsDestination.transpose() * essentialMatrix * inverseInstrinsicsSource;

	// Detecting the point correspondences
	cv::Mat correspondencesLeftRight,
		correspondencesLeftNextLeft,
		correspondencesRightNextLeft,
		normalizedCorrespondencesLeftRight,
		normalizedCorrespondencesLeftNextLeft,
		normalizedCorrespondencesRightNextLeft,
		imageLeft,
		imageRight,
		imageNextLeft;

	loadCorrespondences(
		imagePath_,
		sequence_,
		frameSource_,
		frameDestination_,
		correspondencesLeftRight,
		correspondencesLeftNextLeft,
		correspondencesRightNextLeft,
		&imageLeft,
		&imageRight,
		&imageNextLeft);

	// Normalizing the correspondences by the intrinsic matrices
	normalizedCorrespondencesLeftRight.create(correspondencesLeftRight.size(), correspondencesLeftRight.type());
	normalizedCorrespondencesLeftNextLeft.create(correspondencesLeftNextLeft.size(), correspondencesLeftNextLeft.type());
	normalizedCorrespondencesRightNextLeft.create(correspondencesRightNextLeft.size(), correspondencesRightNextLeft.type());

	gcransac::utils::normalizeCorrespondences(
		correspondencesLeftRight,
		intrinsicsSource_,
		intrinsicsDestination_,
		normalizedCorrespondencesLeftRight);

	gcransac::utils::normalizeCorrespondences(
		correspondencesLeftNextLeft,
		intrinsicsSource_,
		intrinsicsDestination_,
		normalizedCorrespondencesLeftNextLeft);

	gcransac::utils::normalizeCorrespondences(
		correspondencesRightNextLeft,
		intrinsicsSource_,
		intrinsicsDestination_,
		normalizedCorrespondencesRightNextLeft);

	// Calculating the ground truth relative pose
	Sophus::SE3d relativePose = 
		groundTruthPoses_[frameDestination_] * groundTruthPoses_[frameSource_].inverse();

	estimateHomography<
		GeneralizedEstimator32>(
			sequence_,
			frameSource_,
			frameDestination_,
			imageLeft,
			imageRight,
			imageNextLeft,
			correspondencesLeftRight,
			correspondencesLeftNextLeft,
			correspondencesRightNextLeft,
			normalizedCorrespondencesLeftRight,
			normalizedCorrespondencesLeftNextLeft,
			normalizedCorrespondencesRightNextLeft,
			Eigen::Matrix3d::Identity(),
			Eigen::Matrix3d::Identity(),
			Eigen::Vector3d::Zero(),
			translationLeftRight,
			relativePose,
			FLAGS_inlier_outlier_threshold);
}



template<class _Estimator, class ... _EstimatorTypes>
void estimateHomography(
	const std::string& sequence_,
	const size_t& frame1_,
	const size_t& frame2_,
	const cv::Mat& imageLeft_,
	const cv::Mat& imageRight_,
	const cv::Mat& imageNextLeft_,
	const cv::Mat& dataLeftRight_,
	const cv::Mat& dataLeftNextLeft_,
	const cv::Mat& dataRightNextLeft_,
	const cv::Mat& normalizedDataLeftRight_,
	const cv::Mat& normalizedDataLeftNextLeft_,
	const cv::Mat& normalizedDataRightNextLeft_,
	const Eigen::Matrix3d& rotationLeft_,
	const Eigen::Matrix3d& rotationRight_,
	const Eigen::Vector3d& translationLeft_,
	const Eigen::Vector3d& translationRight_,
	const Sophus::SE3d& groundTruthPose_,
	const double& inlierOutlierThreshold_)
{
	gcransac::Model model;

	// Initialize the estimator
	std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses(2);
	generalizedCameraPoses[0] << rotationLeft_, translationLeft_;
	generalizedCameraPoses[1] << rotationRight_, translationRight_;
	_Estimator estimator;
	estimator.setRigPoses(generalizedCameraPoses);

	// Compose the data matrix
	cv::Mat dataMatrix(dataLeftNextLeft_.rows + dataRightNextLeft_.rows, 11, CV_64F);
	const Eigen::Vector3d leftCenter = -rotationLeft_.transpose() * translationLeft_;
	const Eigen::Vector3d rightCenter = -rotationRight_.transpose() * translationRight_;
	Eigen::Vector3d pt1, pt2;

	for (size_t rowIdx = 0; rowIdx < normalizedDataLeftNextLeft_.rows; ++rowIdx)
	{
		pt1 <<
			normalizedDataLeftNextLeft_.at<double>(rowIdx, 0),
			normalizedDataLeftNextLeft_.at<double>(rowIdx, 1),
			1;
		pt1 = rotationLeft_.transpose() * pt1;

		dataMatrix.at<double>(rowIdx, 0) = normalizedDataLeftNextLeft_.at<double>(rowIdx, 2);
		dataMatrix.at<double>(rowIdx, 1) = normalizedDataLeftNextLeft_.at<double>(rowIdx, 3);
		dataMatrix.at<double>(rowIdx, 2) = pt1(0);
		dataMatrix.at<double>(rowIdx, 3) = pt1(1);
		dataMatrix.at<double>(rowIdx, 4) = pt1(2);
		dataMatrix.at<double>(rowIdx, 5) = leftCenter(0);
		dataMatrix.at<double>(rowIdx, 6) = leftCenter(1);
		dataMatrix.at<double>(rowIdx, 7) = leftCenter(2);
		dataMatrix.at<double>(rowIdx, 8) = 0;
		dataMatrix.at<double>(rowIdx, 9) = normalizedDataLeftNextLeft_.at<double>(rowIdx, 0);
		dataMatrix.at<double>(rowIdx, 10) = normalizedDataLeftNextLeft_.at<double>(rowIdx, 1);
	}

	for (size_t rowIdx = 0; rowIdx < normalizedDataRightNextLeft_.rows; ++rowIdx)
	{
		const size_t realIdx = dataLeftNextLeft_.rows + rowIdx;

		pt2 <<
			normalizedDataRightNextLeft_.at<double>(rowIdx, 0),
			normalizedDataRightNextLeft_.at<double>(rowIdx, 1),
			1;
		pt2 = rotationRight_.transpose() * pt2;

		dataMatrix.at<double>(realIdx, 0) = normalizedDataRightNextLeft_.at<double>(rowIdx, 2);
		dataMatrix.at<double>(realIdx, 1) = normalizedDataRightNextLeft_.at<double>(rowIdx, 3);
		dataMatrix.at<double>(realIdx, 2) = pt2(0);
		dataMatrix.at<double>(realIdx, 3) = pt2(1);
		dataMatrix.at<double>(realIdx, 4) = pt2(2);
		dataMatrix.at<double>(realIdx, 5) = rightCenter(0);
		dataMatrix.at<double>(realIdx, 6) = rightCenter(1);
		dataMatrix.at<double>(realIdx, 7) = rightCenter(2);
		dataMatrix.at<double>(realIdx, 8) = 1;
		dataMatrix.at<double>(realIdx, 9) = normalizedDataRightNextLeft_.at<double>(rowIdx, 0);
		dataMatrix.at<double>(realIdx, 10) = normalizedDataRightNextLeft_.at<double>(rowIdx, 1);
	}

	gcransac::sampler::UniformGeneralizedSampler sampler(
		&dataMatrix,
		{ std::make_pair(0, dataLeftNextLeft_.rows - 1),
			std::make_pair(dataLeftNextLeft_.rows, dataMatrix.rows - 1) },
		{ 3, 2 }); // The main sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!sampler.isInitialized())
	{
		fprintf(stderr, "The sampler is not initialized successfully.\n");
		return;
	}

	ProgXPrime progressiveXPrime;

	progressiveXPrime.getMutableEstimator().setRigPoses(generalizedCameraPoses);

	auto& settings = progressiveXPrime.getMutableSettings();
	settings.inlierOutlierThreshold = inlierOutlierThreshold_;
	settings.modelDistanceThreshold = FLAGS_model_to_model_distance;
	settings.maximumIterations = 100;
	settings.minimumInlierNumber = FLAGS_minimum_point_number;
	settings.startingHypothesisNumber = 10;
	settings.addedHypothesisNumber = 10;
	settings.confidence = 0.999;

	std::vector<gcransac::Model> models;
	std::vector<progx::ModelData> modelData;

	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	progressiveXPrime.run(
		dataMatrix,
		sampler,
		models,
		modelData);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	double time = elapsed_seconds.count();

	// Recovering the pose from the obtained homographies


	/*printf("Inlier number = %d\nProcessing time = %f\nIteration number = %d\nRotation error = %f\nTranslation error = %f\nPosition error = %f\n",
		inlierNumber,
		statistics_.processing_time,
		statistics_.iteration_number,
		bestRotationError,
		bestTranslationError,
		bestPositionError);*/

	/*savingMutex.lock();
	std::ofstream resultFile("results.csv", std::fstream::app);
	resultFile
		<< sequence_ << ";"
		<< frame1_ << ";"
		<< frame2_ << ";"
		<< "H" << ";"
		<< estimator.sampleSize() << ";"
		<< bestRotationError << ";"
		<< bestTranslationError << ";"
		<< bestPositionError << ";"
		<< inlierNumber << ";"
		<< statistics_.processing_time << ";"
		<< statistics_.iteration_number << "\n";
	resultFile.close();
	savingMutex.unlock();*/

	cv::Mat tmp1 = imageLeft_.clone();
	cv::Mat tmp2 = imageRight_.clone();
	cv::Mat tmp3 = imageNextLeft_.clone();

	for (const auto& data : modelData)
	{
		cv::Scalar color(
			255.0 * (double)rand() / RAND_MAX,
			255.0 * (double)rand() / RAND_MAX,
			255.0 * (double)rand() / RAND_MAX);

		for (const auto& inlierIdx : data.inliers)
		{
			if ((int)dataMatrix.at<double>(inlierIdx, 8) == 0)
			{
				cv::circle(tmp1,
					cv::Point2d(dataLeftNextLeft_.at<double>(inlierIdx, 0), dataLeftNextLeft_.at<double>(inlierIdx, 1)),
					3,
					color, -1);

				cv::circle(tmp3,
					cv::Point2d(dataLeftNextLeft_.at<double>(inlierIdx, 2), dataLeftNextLeft_.at<double>(inlierIdx, 3)),
					3,
					color, -1);
			}
			else
			{
				cv::circle(tmp2,
					cv::Point2d(dataRightNextLeft_.at<double>(inlierIdx - dataLeftNextLeft_.rows, 0), dataRightNextLeft_.at<double>(inlierIdx - dataLeftNextLeft_.rows, 1)),
					3,
					color, -1);

				cv::circle(tmp3,
					cv::Point2d(dataRightNextLeft_.at<double>(inlierIdx - dataLeftNextLeft_.rows, 2), dataRightNextLeft_.at<double>(inlierIdx - dataLeftNextLeft_.rows, 3)),
					3,
					color, -1);
			}
		}
	}

	cv::imshow("Image 1", tmp1);
	cv::imshow("Image 2", tmp2);
	cv::imshow("Image 3", tmp3);
	cv::waitKey(0);

	

	// If there is an untested estimator run the function again
	if constexpr (sizeof...(_EstimatorTypes) > 0) {
		estimateHomography<_EstimatorTypes...>(
			sequence_,
			frame1_,
			frame2_,
			imageLeft_,
			imageRight_,
			imageNextLeft_,
			dataLeftRight_,
			dataLeftNextLeft_,
			dataRightNextLeft_,
			normalizedDataLeftRight_,
			normalizedDataLeftNextLeft_,
			normalizedDataRightNextLeft_,
			rotationLeft_,
			rotationRight_,
			translationLeft_,
			translationRight_,
			groundTruthPose_,
			inlierOutlierThreshold_);
	}
}

void processKITTI()
{
	const std::vector<std::string> sequences = 
		{ "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11" };
	constexpr int kStepSize = 1;

	// The calibration matrices of the left-right cameras
	Eigen::Matrix<double, 3, 4> P1, P2;
	P1 << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
		0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
		0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;
	P2 << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02,
		0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
		0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;

	const Eigen::Matrix3d
		& intrinsicsSource = P1.block<3, 3>(0, 0),
		& intrinsicsDestination = P2.block<3, 3>(0, 0);

	const Sophus::SE3d rigPose(Eigen::Matrix3d::Identity(), P2.rightCols<1>());

	// Selecting the sequences to be processed
	std::vector<std::string> sequencesToProcess;
	if (FLAGS_sequence_idx == -1)
		sequencesToProcess = sequences;
	else if (FLAGS_sequence_idx < sequences.size())
		sequencesToProcess.emplace_back(sequences[FLAGS_sequence_idx]);
	else
	{
		fprintf(stderr, "The allowed sequence indices are -1 (all) or from interval [0, %d].\n", sequences.size() - 1);
		return;
	}

	// Iterating through the sequences
	for (const auto& sequence : sequences)
	{
		// Loading the ground truth poses for the current sequence
		std::vector<Sophus::SE3d> groundTruthPoses;
		if (!loadGroundTruthPoses(
			FLAGS_reference_pose_path,
			sequence,
			groundTruthPoses))
		{
			printf("An error occured when loading the poses for sequence '%s'\n", sequence.c_str());
			continue;
		}

		// The number of frames in the current sequence
		const size_t &frameNumber = groundTruthPoses.size();

		for (int frame = 0; frame < frameNumber - FLAGS_frame_step_size; frame += FLAGS_frame_step_size)
			processFramePair(
				FLAGS_image_path,
				sequence,
				frame,
				frame + FLAGS_frame_step_size,
				intrinsicsSource,
				intrinsicsDestination,
				rigPose,
				groundTruthPoses);
	}
}

void loadCorrespondences(
	const std::string& imagePath_,
	const std::string& sequence_,
	const size_t& frameSource_,
	const size_t& frameDestination_,
	cv::Mat& correspondencesLeftRight_,
	cv::Mat& correspondencesLeftNextLeft_,
	cv::Mat& correspondencesRightNextLeft_,
	cv::Mat* imageLeft_,
	cv::Mat* imageRight_,
	cv::Mat* imageNextLeft_)
{
	// Loading the images of the first frame
	std::string imageName = std::to_string(frameSource_);
	constexpr size_t filenameLength = 6;
	const size_t currentSize = imageName.size();
	for (int i = 0; i < filenameLength - currentSize; ++i)
		imageName = "0" + imageName;

	// Loading the left image of the next frame
	std::string imageNameDestination = std::to_string(frameDestination_);
	const size_t currentSizeDestination = imageNameDestination.size();
	for (int i = 0; i < filenameLength - currentSizeDestination; ++i)
		imageNameDestination = "0" + imageNameDestination;

	*imageLeft_ = cv::imread(imagePath_ + sequence_ + "/image_0/" + imageName + ".png");
	*imageRight_ = cv::imread(imagePath_ + sequence_ + "/image_1/" + imageName + ".png");
	*imageNextLeft_ = cv::imread(imagePath_ + sequence_ + "/image_0/" + imageNameDestination + ".png");

	// Detecting point correspondences between the left and right images
	gcransac::utils::detectFeatures(
		FLAGS_workspace_path + sequence_ + "_" + std::to_string(frameSource_) + "_left_right.txt", // The path where the correspondences are read from or saved to.
		*imageLeft_, // The source image
		*imageRight_, // The destination image
		correspondencesLeftRight_); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	// Detecting point correspondences between the left and next left images
	gcransac::utils::detectFeatures(
		FLAGS_workspace_path + sequence_ + "_" + std::to_string(frameSource_) + "_left_" + std::to_string(frameDestination_) + "_left.txt", // The path where the correspondences are read from or saved to.
		*imageLeft_, // The source image
		*imageNextLeft_, // The destination image
		correspondencesLeftNextLeft_); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	// Detecting point correspondences between the right and next images
	gcransac::utils::detectFeatures(
		FLAGS_workspace_path + sequence_ + "_" + std::to_string(frameSource_) + "_right_" + std::to_string(frameDestination_) + "_left.txt", // The path where the correspondences are read from or saved to.
		*imageRight_, // The source image
		*imageNextLeft_, // The destination image
		correspondencesRightNextLeft_); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"
}

bool loadGroundTruthPoses(
	const std::string& posePath_,
	const std::string& sequence_,
	std::vector<Sophus::SE3d>& groundTruthPoses_)
{
	std::ifstream file(posePath_ + sequence_ + ".txt");
	if (!file.is_open())
		return false;

	double scale = 1;
	Sophus::SE3d pose;
	Eigen::Matrix<double, 3, 4> projectionMatrix;
	while (file >> projectionMatrix(0, 0) >> projectionMatrix(0, 1) >> projectionMatrix(0, 2) >> projectionMatrix(0, 3)
		>> projectionMatrix(1, 0) >> projectionMatrix(1, 1) >> projectionMatrix(1, 2) >> projectionMatrix(1, 3)
		>> projectionMatrix(2, 0) >> projectionMatrix(2, 1) >> projectionMatrix(2, 2) >> projectionMatrix(2, 3))
	{
		Eigen::Matrix3d rotationMatrix =
			projectionMatrix.block<3, 3>(0, 0);
		pose::convertToRotationMatrix(
			rotationMatrix,
			&scale);
		groundTruthPoses_.emplace_back(Sophus::SE3d(
			rotationMatrix,
			projectionMatrix.rightCols<1>()));
	}

	file.close();
	return groundTruthPoses_.size() > 0;
}