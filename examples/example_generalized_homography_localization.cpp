#include <regex>
#include <vector>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"
#include "progx_utils.h"
#include "losses.h"
#include "utils.h"
#include "progressive_x_prime.h"
#include "mean_shift_clustering.h"
#include "dbscan_clustering.h"
#include "distances.h"
#include "GCoptimization.h"
#include "flann_neighborhood_graph.h"
#include "uniform_generalized_sampler.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "modified_fundamental_estimator.h"
#include "modified_homography_estimator.h"
#include "preemption_sprt.h"
#include "generalized_homography_estimator.h"
#include "solver_generalized_homography_ceres.h"
#include "solver_generalized_homography_three_two_point.h"
#include "generalized_essential_estimator.h"
#include "solver_generalized_essential_matrix_ceres.h"
#include "solver_gen_essential_matrix_six_point.h"
#include "pose_utils.h"

#include <ctime>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>

#include <gflags/gflags.h>
#include <sophus/se3.hpp>

#include <mutex>

DEFINE_double(inlier_outlier_threshold, 0.02,
	"The inlier-outlier threshold for the robust homography fitting.");
DEFINE_double(essential_inlier_outlier_threshold, 0.0002,
	"The inlier-outlier threshold for the robust essential matrix fitting.");
DEFINE_bool(estimate_essential_matrix, true,
	"A flag to determine if essential matrices should also be estimated.");
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
DEFINE_int32(core_number, 26,
	"The minimum number of points required to keep a model.");

// Typedef for the clustering in the high-dimensional consensus space
typedef clustering::density::DBScanClustering<
	// The object containing the information regarding a model instance
	progx::ModelData,
	// The model-to-model distance function defined over the preference vectors
	clustering::distances::TanimotoDistance<progx::ModelData>> ClusteringMethod;

// The default estimator for generalized homography fitting
typedef gcransac::estimator::GeneralizedHomographyEstimator<
	// The solver used for fitting a model to a minimal sample
	gcransac::estimator::solver::GeneralizedHomographyThreeTwoPointSolver, 
	// The solver used for fitting a model to a non-minimal sample
	gcransac::estimator::solver::GeneralizedHomographyCeresSolver> 
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

// The exact type of the multi-model fitting algorithm
typedef progx::ProgressiveXPrime<
	// The currently used clustering technique
	ClusteringMethod,
	// Using Tanimoto distance as the model-to-model distances
	clustering::distances::TanimotoDistance<progx::ModelData>,
	// Using MAGSAC++-based weights both in the iteratively re-weighted LSQ fitting and for the model representation in the consensus space.
	clustering::losses::TukeyBisquareLoss<double>,
	// We are looking for homographies, thus the homography estimator as specified here.
	gcransac::utils::DefaultHomographyEstimator,
	// The sampler used for finding minimal samples
	gcransac::sampler::UniformSampler> ProgXPrimeTraditionalHomography;


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

void poseSelectionFromHomographies(
	const cv::Mat& dataPoints_,
	const std::vector<gcransac::Model>& models_,
	const std::vector<progx::ModelData>& modelData_,
	const Eigen::Matrix3d& sourceIntrinsics_,
	const Eigen::Matrix3d& destinationIntrinsics_,
	const std::vector<Eigen::Matrix<double, 3, 4>>& generalizedCameraPoses_,
	const Sophus::SE3d& relativePose_,
	std::vector<size_t>& inliers_,
	double& score_,
	Sophus::SE3d& estimatedPose_);

template<class _Estimator, class ... _EstimatorTypes>
void estimateHomography(
	const std::string& sequence_,
	const size_t& frame1_,
	const size_t& frame2_,
	const cv::Mat& imageLeft_,
	const cv::Mat& imageRight_,
	const cv::Mat& imageNextLeft_,
	const cv::Mat& dataMatrix_, // The matrix consisting of the correspondences and everything that is needed for the estimation
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
	Sophus::SE3d& estimatedPose_, // The estimated relative pose
	std::vector<size_t> inliers_,
	double& poseScore_,
	const double& inlierOutlierThreshold_);

template<class _Estimator, class ... _EstimatorTypes>
void estimateEssentialMatrix(
	const std::string& sequence_,
	const size_t& frame1_,
	const size_t& frame2_,
	const cv::Mat& imageLeft_,
	const cv::Mat& imageRight_,
	const cv::Mat& imageNextLeft_,
	const cv::Mat& dataMatrix_,  // The matrix consisting of the correspondences and everything that is needed for the estimation
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
	std::vector<Sophus::SE3d>& estimatedPoses_, // The estimated relative pose
	const double& inlierOutlierThreshold_);

void poseSelection(
	const cv::Mat& dataPoints_,
	const std::vector<Sophus::SE3d>& poses_,
	const Eigen::Matrix3d& sourceIntrinsics_,
	const Eigen::Matrix3d& destinationIntrinsics_,
	const std::vector<Eigen::Matrix<double, 3, 4>>& generalizedCameraPoses_,
	std::vector<size_t>& inliers_,
	double& score_,
	Sophus::SE3d& estimatedPose_);

void processKITTI();

// int main(int argc, char** argv)
// {
// 	// Parsing the flags
// 	gflags::ParseCommandLineFlags(&argc, &argv, true);

// 	// Running tests on the KITTI dataset
// 	processKITTI();

// 	return 0;
// }

// Estimating the relative pose from a pair of consecutive frames
void processFramePair(
	const std::string& imagePath_, // The path where the images are stored
	const std::string& sequence_, // The KITTI sequence index
	const size_t& frameSource_, // The source frame's index
	const size_t& frameDestination_, // The destination frame's index
	const Eigen::Matrix3d intrinsicsSource_, // The intrinsic camera parameters of the source camera
	const Eigen::Matrix3d intrinsicsDestination_, // The intrinsic camera parameters of the destination camera
	const Sophus::SE3d &rigPose_, // The pose from the left to the right camera 
	const std::vector<Sophus::SE3d>& groundTruthPoses_) // The ground truth absolute poses
{
	printf("Process frames %d -> %d.\n", frameSource_, frameDestination_);

	// The inverse intrinsic parameter matrix of the source view
	const Eigen::Matrix3d inverseInstrinsicsSource =
		intrinsicsSource_.inverse();
	// The inverse intrinsic parameter matrix of the destination view
	const Eigen::Matrix3d inverseInstrinsicsDestination =
		intrinsicsDestination_.inverse();

	// Ther translation from the left to the right image
	const Eigen::Vector3d &translationLeftRight =
		rigPose_.translation();
	// The cross-product matrix of the translation vector
	Eigen::Matrix3d translationLeftRightCross;
	translationLeftRightCross <<
		0, translationLeftRight(2), -translationLeftRight(1),
		-translationLeftRight(2), 0, translationLeftRight(0),
		translationLeftRight(1), -translationLeftRight(0), 0;

	// The essential matrix describing the relative pose of the left and right images in the rig
	const Eigen::Matrix3d essentialMatrix =
		rigPose_.rotationMatrix() * translationLeftRightCross;
	// The implied fundamental matrix
	const Eigen::Matrix3d fundamentalMatrix =
		inverseInstrinsicsDestination.transpose() * essentialMatrix * inverseInstrinsicsSource;

	// Detecting the point correspondences
	cv::Mat correspondencesLeftRight, // The point correspondences in the left and right images
		correspondencesLeftNextLeft, // The point correspondences in the left and next left images
		correspondencesRightNextLeft, // The point correspondences in the right and next left images
		normalizedCorrespondencesLeftRight, // The normalized point correspondences in the left and right images
		normalizedCorrespondencesLeftNextLeft, // The normalized point correspondences in the left and next left images
		normalizedCorrespondencesRightNextLeft, // The normalized point correspondences in the right and next left images
		imageLeft, // The left image
		imageRight, // The right image
		imageNextLeft; // The next left image

	// Finding the required correspondences
	loadCorrespondences(
		imagePath_, // The path were the images are stored
		sequence_, // The current KITTI sequence index
		frameSource_, // The current frame
 		frameDestination_, // The next frame
		correspondencesLeftRight, // The point correspondences in the left and right images
		correspondencesLeftNextLeft, // The point correspondences in the left and next left images
		correspondencesRightNextLeft, // The point correspondences in the right and next left images
		&imageLeft, // The left image
		&imageRight, // The right image
		&imageNextLeft); // The next left image

	// Continue if we don't have at least 20 inliers.
	if (correspondencesLeftRight.rows < 20 ||
		correspondencesLeftNextLeft.rows < 20 ||
		correspondencesRightNextLeft.rows < 20)
		return;

	// Normalizing the correspondences by the intrinsic matrices
	normalizedCorrespondencesLeftRight.create(correspondencesLeftRight.size(), correspondencesLeftRight.type());
	normalizedCorrespondencesLeftNextLeft.create(correspondencesLeftNextLeft.size(), correspondencesLeftNextLeft.type());
	normalizedCorrespondencesRightNextLeft.create(correspondencesRightNextLeft.size(), correspondencesRightNextLeft.type());

	gcransac::utils::normalizeCorrespondences(
		correspondencesLeftRight, // The point correspondences in the left and right images
		intrinsicsSource_, // The intrinsic parameters of the source view
		intrinsicsDestination_, // The intrinsic parameters of the destination view
		normalizedCorrespondencesLeftRight); // The normalized point correspondences in the left and right images

	gcransac::utils::normalizeCorrespondences(
		correspondencesLeftNextLeft, // The point correspondences in the left and next left images
		intrinsicsSource_, // The intrinsic parameters of the source view
		intrinsicsDestination_, // The intrinsic parameters of the destination view
		normalizedCorrespondencesLeftNextLeft); // The normalized point correspondences in the left and next left images

	gcransac::utils::normalizeCorrespondences(
		correspondencesRightNextLeft, // The point correspondences in the right and next left images
		intrinsicsSource_, // The intrinsic parameters of the source view
		intrinsicsDestination_, // The intrinsic parameters of the destination view
		normalizedCorrespondencesRightNextLeft); // The normalized point correspondences in the right and next left images

	// Composing the data matrix
	const Eigen::Matrix3d rotationLeft = Eigen::Matrix3d::Identity(), // The relative rotation of the left camera
		rotationRight = Eigen::Matrix3d::Identity(); // The relative rotaiton of the right camera w.r.t. the left one
	const Eigen::Vector3d translationLeft = Eigen::Vector3d::Zero(), // The relative translation of the left camera
		& translationRight = translationLeftRight; // The relative translation of the right camera w.r.t. the left one

	cv::Mat dataMatrix(normalizedCorrespondencesLeftNextLeft.rows + normalizedCorrespondencesRightNextLeft.rows, 11, CV_64F);
	const Eigen::Vector3d leftCenter = -rotationLeft.transpose() * translationLeft;
	const Eigen::Vector3d rightCenter = -rotationRight.transpose() * translationRight;
	Eigen::Vector3d pt1, pt2;

	for (size_t rowIdx = 0; rowIdx < normalizedCorrespondencesLeftNextLeft.rows; ++rowIdx)
	{
		pt1 <<
			normalizedCorrespondencesLeftNextLeft.at<double>(rowIdx, 0),
			normalizedCorrespondencesLeftNextLeft.at<double>(rowIdx, 1),
			1;
		pt1 = rotationLeft.transpose() * pt1;

		dataMatrix.at<double>(rowIdx, 0) = normalizedCorrespondencesLeftNextLeft.at<double>(rowIdx, 2);
		dataMatrix.at<double>(rowIdx, 1) = normalizedCorrespondencesLeftNextLeft.at<double>(rowIdx, 3);
		dataMatrix.at<double>(rowIdx, 2) = pt1(0);
		dataMatrix.at<double>(rowIdx, 3) = pt1(1);
		dataMatrix.at<double>(rowIdx, 4) = pt1(2);
		dataMatrix.at<double>(rowIdx, 5) = leftCenter(0);
		dataMatrix.at<double>(rowIdx, 6) = leftCenter(1);
		dataMatrix.at<double>(rowIdx, 7) = leftCenter(2);
		dataMatrix.at<double>(rowIdx, 8) = 0;
		dataMatrix.at<double>(rowIdx, 9) = normalizedCorrespondencesLeftNextLeft.at<double>(rowIdx, 0);
		dataMatrix.at<double>(rowIdx, 10) = normalizedCorrespondencesLeftNextLeft.at<double>(rowIdx, 1);
	}

	for (size_t rowIdx = 0; rowIdx < normalizedCorrespondencesRightNextLeft.rows; ++rowIdx)
	{
		const size_t realIdx = normalizedCorrespondencesLeftNextLeft.rows + rowIdx;

		pt2 <<
			normalizedCorrespondencesRightNextLeft.at<double>(rowIdx, 0),
			normalizedCorrespondencesRightNextLeft.at<double>(rowIdx, 1),
			1;
		pt2 = rotationRight.transpose() * pt2;

		dataMatrix.at<double>(realIdx, 0) = normalizedCorrespondencesRightNextLeft.at<double>(rowIdx, 2);
		dataMatrix.at<double>(realIdx, 1) = normalizedCorrespondencesRightNextLeft.at<double>(rowIdx, 3);
		dataMatrix.at<double>(realIdx, 2) = pt2(0);
		dataMatrix.at<double>(realIdx, 3) = pt2(1);
		dataMatrix.at<double>(realIdx, 4) = pt2(2);
		dataMatrix.at<double>(realIdx, 5) = rightCenter(0);
		dataMatrix.at<double>(realIdx, 6) = rightCenter(1);
		dataMatrix.at<double>(realIdx, 7) = rightCenter(2);
		dataMatrix.at<double>(realIdx, 8) = 1;
		dataMatrix.at<double>(realIdx, 9) = normalizedCorrespondencesRightNextLeft.at<double>(rowIdx, 0);
		dataMatrix.at<double>(realIdx, 10) = normalizedCorrespondencesRightNextLeft.at<double>(rowIdx, 1);
	}

	// Calculating the ground truth relative pose
	Sophus::SE3d relativePose = 
		groundTruthPoses_[frameDestination_] * groundTruthPoses_[frameSource_].inverse();
	// Eigen::Vector3d t;
	// t << 0, 0, 1;
	// t *= (groundTruthPoses_[frameDestination_].translation() - groundTruthPoses_[frameSource_].translation()).norm();
	// relativePose = Sophus::SE3d(relativePose.rotationMatrix(), t);

	// The estimated relative pose
	Sophus::SE3d estimatedPose;
	// The inliers of the pose determined by thresholding the re-projection error
	std::vector<size_t> poseInliers;
	// The score of the pose
	double poseScore;

	// Estimating multiple homographies
	estimateHomography<
		GeneralizedEstimator32>(
			sequence_, // The current KITTI sequence index
			frameSource_, // The current frame
			frameDestination_, // The next frame
			imageLeft, // The left image
			imageRight, // The right image
			imageNextLeft, // The next left image
			dataMatrix,  // The matrix consisting of the correspondences and everything that is needed for the estimation
			correspondencesLeftRight, // The point correspondences in the left and right images
			correspondencesLeftNextLeft, // The point correspondences in the left and next left images
			correspondencesRightNextLeft, // The point correspondences in the right and next left images
			normalizedCorrespondencesLeftRight, // The normalized point correspondences in the left and right images
			normalizedCorrespondencesLeftNextLeft, // The normalized point correspondences in the left and next left images
			normalizedCorrespondencesRightNextLeft, // The normalized point correspondences in the right and next left images
			Eigen::Matrix3d::Identity(), // The relative rotation of the left camera
			Eigen::Matrix3d::Identity(), // The relative rotaiton of the right camera w.r.t. the left one
			Eigen::Vector3d::Zero(), // The relative translation of the left camera
			translationLeftRight, // The relative translation of the right camera w.r.t. the left one
			relativePose, // The ground truth relative pose
			estimatedPose, // The estimated relative pose
			poseInliers,
			poseScore, 
			FLAGS_inlier_outlier_threshold); // The used inlier-outlier threshold

	// Estimating the relative pose from essential matrices as well
	if (FLAGS_estimate_essential_matrix)
	{
		// Estimating the generalized essential matrix as well
		// The default estimator for homography fitting
		typedef gcransac::estimator::GeneralizedEssentialMatrixEstimator<
			gcransac::estimator::solver::GeneralizedEssentialMatrixSixPointSolver, // The solver used for fitting a model to a minimal sample
			gcransac::estimator::solver::GeneralizedEssentialMatrixCeresSolver<0>> // The solver used for fitting a model to a non-minimal sample
			GeneralizedEstimator42;

		std::vector<Sophus::SE3d> estimatedPosesFromEssentialMatrix;
		estimateEssentialMatrix<
			GeneralizedEstimator42>(
				sequence_, // The current KITTI sequence index
				frameSource_, // The current frame
				frameDestination_, // The next frame
				imageLeft, // The left image
				imageRight, // The right image
				imageNextLeft, // The next left image
				dataMatrix,  // The matrix consisting of the correspondences and everything that is needed for the estimation
				correspondencesLeftRight, // The point correspondences in the left and right images
				correspondencesLeftNextLeft, // The point correspondences in the left and next left images
				correspondencesRightNextLeft, // The point correspondences in the right and next left images
				normalizedCorrespondencesLeftRight, // The normalized point correspondences in the left and right images
				normalizedCorrespondencesLeftNextLeft, // The normalized point correspondences in the left and next left images
				normalizedCorrespondencesRightNextLeft, // The normalized point correspondences in the right and next left images
				Eigen::Matrix3d::Identity(), // The relative rotation of the left camera
				Eigen::Matrix3d::Identity(), // The relative rotaiton of the right camera w.r.t. the left one
				Eigen::Vector3d::Zero(), // The relative translation of the left camera
				translationLeftRight, // The relative translation of the right camera w.r.t. the left one
				relativePose, // The ground truth relative pose
				estimatedPosesFromEssentialMatrix, // The estimated relative pose
				FLAGS_essential_inlier_outlier_threshold); // The used inlier-outlier threshold

		// Initialize the estimator
		std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses(2);
		generalizedCameraPoses[0] << rotationLeft, translationLeft;
		generalizedCameraPoses[1] << rotationRight, translationRight;

		poseSelection(
			dataMatrix,
			estimatedPosesFromEssentialMatrix,
			intrinsicsSource_,
			intrinsicsDestination_,
			generalizedCameraPoses,
			poseInliers,
			poseScore,
			estimatedPose);
	}

	// Calculating the error of the estimated pose
	constexpr double rad2Deg = 180.0 / M_PI;

	// Compute the angular translation error by using the dot product between translations.
	const Eigen::Vector3d referenceTranslationDstSrc = relativePose.translation().normalized();
	const Eigen::Vector3d translationDstSrc = estimatedPose.translation().normalized();

	double positionError =
		(-estimatedPose.rotationMatrix().transpose() * relativePose.translation().norm() * translationDstSrc.normalized() -
			(-relativePose.rotationMatrix().transpose() * relativePose.translation())).norm();

	// Floating point errors can put the cosTheta slightly outside of (-1, 1) so we need to clamp.
	double translationError = rad2Deg * std::acos(std::clamp(referenceTranslationDstSrc.dot(translationDstSrc), -1.0, 1.0));
	translationError = MIN(rad2Deg * std::acos(std::clamp(referenceTranslationDstSrc.dot(-translationDstSrc), -1.0, 1.0)), translationError);

	Sophus::SO3d rotationDifference = Sophus::SO3d(relativePose.rotationMatrix()).inverse() * Sophus::SO3d(estimatedPose.rotationMatrix());
	double rotationError = rad2Deg * rotationDifference.log().norm();

	// Recovering the pose from the obtained homographies
	LOG(INFO) << "Translation error = " << translationError << " degrees";
	LOG(INFO) << "Rotation error = " << rotationError << " degrees";
	LOG(INFO) << "Position error = " << rotationError << " meters";

	static std::mutex savingMutex;

	savingMutex.lock();
	std::ofstream resultFile("results.csv", std::fstream::app);
	resultFile
		<< sequence_ << ";"
		<< frameSource_ << ";"
		<< frameDestination_ << ";"
		<< rotationError << ";"
		<< translationError << ";"
		<< positionError << "\n";
	resultFile.close();
	savingMutex.unlock();
}


template<class _Estimator, class ... _EstimatorTypes>
void estimateEssentialMatrix(
	const std::string& sequence_,
	const size_t& frame1_,
	const size_t& frame2_,
	const cv::Mat& imageLeft_,
	const cv::Mat& imageRight_,
	const cv::Mat& imageNextLeft_,
	const cv::Mat& dataMatrix_,  // The matrix consisting of the correspondences and everything that is needed for the estimation
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
	std::vector<Sophus::SE3d>& estimatedPoses_, // The estimated relative pose
	const double& inlierOutlierThreshold_)
{
	gcransac::Model model;

	// Initialize the estimator
	std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses(2);
	generalizedCameraPoses[0] << rotationLeft_, translationLeft_;
	generalizedCameraPoses[1] << rotationRight_, translationRight_;
	_Estimator estimator(generalizedCameraPoses);

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&dataMatrix_, 20);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	gcransac::preemption::SPRTPreemptiveVerfication<_Estimator> sprt(
		dataMatrix_,
		estimator);
	gcransac::preemption::EmptyPreemptiveVerfication<_Estimator> emptyPreemption;

	gcransac::GCRANSAC<_Estimator,
		gcransac::neighborhood::FlannNeighborhoodGraph,
		gcransac::MSACScoringFunction<_Estimator>,
		gcransac::preemption::SPRTPreemptiveVerfication<_Estimator>> gcransac;
	gcransac.settings.threshold = inlierOutlierThreshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
	gcransac.settings.confidence = 0.99; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 5; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 20; // The radius of the neighborhood ball
	gcransac.settings.do_local_optimization = false;
	gcransac.settings.do_final_iterated_least_squares = false;

	gcransac::sampler::UniformGeneralizedSampler main_sampler(
		&dataMatrix_,
		{ std::make_pair(0, dataLeftNextLeft_.rows - 1),
			std::make_pair(dataLeftNextLeft_.rows, dataMatrix_.rows - 1) },
		{ 4, 2 }); // The main sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&dataMatrix_); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	// Start GC-RANSAC
	gcransac.run(dataMatrix_,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model,
		sprt);

	// Get the statistics of the results
	const gcransac::utils::RANSACStatistics& statistics_ =
		gcransac.getRansacStatistics();

	const size_t& inlierNumber = 
		statistics_.inliers.size();

	std::vector<size_t> inliers;
	inliers.reserve(inlierNumber);
	for (const auto& inlierIdx : statistics_.inliers)
		if (inlierIdx < normalizedDataLeftNextLeft_.rows)
			inliers.emplace_back(inlierIdx);

	if (inliers.size() >= 6 &&
		model.descriptor.rows() == 3)
	{
		// Decomposing the essential matrix
		std::vector<Eigen::Matrix3d> rotations(2);
		std::vector<Eigen::Vector3d> translations(2);

		pose::decomposeEssentialMatrix(
			model.descriptor.block<3, 3>(0, 4).transpose(),
			rotations[0],
			rotations[1],
			translations[0]);
		translations[1] = -translations[0];

		double scale;
		pose::convertToRotationMatrix(rotations[0], &scale);
		pose::convertToRotationMatrix(rotations[1], &scale);

		estimatedPoses_.emplace_back(Sophus::SE3d(rotations[0], translations[0]));
		estimatedPoses_.emplace_back(Sophus::SE3d(rotations[0], -translations[0]));
		estimatedPoses_.emplace_back(Sophus::SE3d(rotations[1], translations[0]));
		estimatedPoses_.emplace_back(Sophus::SE3d(rotations[1], -translations[0]));
	}

	// If there is an untested estimator run the function again
	if constexpr (sizeof...(_EstimatorTypes) > 0) {
		estimateEssentialMatrix<_EstimatorTypes...>(
			sequence_,
			frame1_,
			frame2_,
			imageLeft_,
			imageRight_,
			imageNextLeft_,
			dataMatrix_,
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
			estimatedPoses_,
			inlierOutlierThreshold_);
	}
}


template<class _Estimator, class ... _EstimatorTypes>
void estimateHomography(
	const std::string& sequence_, // The current KITTI sequence index
	const size_t& frame1_,// The current frame
	const size_t& frame2_, // The next frame
	const cv::Mat& imageLeft_, // The left image
	const cv::Mat& imageRight_, // The right image
	const cv::Mat& imageNextLeft_, // The next left image
	const cv::Mat& dataMatrix_, // The matrix consisting of the correspondences and everything that is needed for the estimation
	const cv::Mat& dataLeftRight_, // The point correspondences in the left and right images
	const cv::Mat& dataLeftNextLeft_, // The point correspondences in the left and next left images
	const cv::Mat& dataRightNextLeft_, // The point correspondences in the right and next left images
	const cv::Mat& normalizedDataLeftRight_, // The normalized point correspondences in the left and right images
	const cv::Mat& normalizedDataLeftNextLeft_, // The normalized point correspondences in the left and next left images
	const cv::Mat& normalizedDataRightNextLeft_, // The normalized point correspondences in the right and next left images
	const Eigen::Matrix3d& rotationLeft_, // The relative rotation of the left camera
	const Eigen::Matrix3d& rotationRight_, // The relative rotaiton of the right camera w.r.t. the left one
	const Eigen::Vector3d& translationLeft_, // The relative translation of the left camera
	const Eigen::Vector3d& translationRight_, // The relative translation of the right camera w.r.t. the left one
	const Sophus::SE3d& groundTruthPose_, // The ground truth relative pose
	Sophus::SE3d& estimatedPose_, // The estimated relative pose
	std::vector<size_t> inliers_,
	double &poseScore_,
	const double& inlierOutlierThreshold_) // The used inlier-outlier threshold
{
	gcransac::Model model;

	// Initialize the estimator
	std::vector<Eigen::Matrix<double, 3, 4>> generalizedCameraPoses(2);
	generalizedCameraPoses[0] << rotationLeft_, translationLeft_;
	generalizedCameraPoses[1] << rotationRight_, translationRight_;

	if constexpr (_Estimator::isGeneralizedHomography())
	{
		_Estimator estimator;
		estimator.setRigPoses(generalizedCameraPoses);

		// Compose the data matrix
		/*cv::Mat dataMatrix(dataLeftNextLeft_.rows + dataRightNextLeft_.rows, 11, CV_64F);
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
		}*/

		gcransac::sampler::UniformGeneralizedSampler sampler(
			&dataMatrix_,
			{ std::make_pair(0, dataLeftNextLeft_.rows - 1),
				std::make_pair(dataLeftNextLeft_.rows, dataMatrix_.rows - 1) },
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
		settings.modelDistanceThreshold = 1.0;
		settings.maximumIterations = 5;
		settings.minimumInlierNumber = FLAGS_minimum_point_number;
		settings.startingHypothesisNumber = 10;
		settings.addedHypothesisNumber = 10;
		settings.confidence = 0.999;

		std::vector<gcransac::Model> models;
		std::vector<progx::ModelData> modelData;

		std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
		start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
		progressiveXPrime.run(
			dataMatrix_,
			sampler,
			models,
			modelData);
		end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
		std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
		double time = elapsed_seconds.count();

		poseSelectionFromHomographies(
			dataMatrix_,
			models,
			modelData,
			Eigen::Matrix3d::Identity(),
			Eigen::Matrix3d::Identity(),
			generalizedCameraPoses,
			groundTruthPose_,
			inliers_,
			poseScore_,
			estimatedPose_);
	} else
	{
		// Compose the data matrix
		cv::Mat dataMatrix(dataLeftNextLeft_.rows, 4, CV_64F);
		Eigen::Vector3d pt1, pt2;

		for (size_t rowIdx = 0; rowIdx < normalizedDataLeftNextLeft_.rows; ++rowIdx)
		{
			dataMatrix.at<double>(rowIdx, 0) = normalizedDataLeftNextLeft_.at<double>(rowIdx, 0);
			dataMatrix.at<double>(rowIdx, 1) = normalizedDataLeftNextLeft_.at<double>(rowIdx, 1);
			dataMatrix.at<double>(rowIdx, 2) = normalizedDataLeftNextLeft_.at<double>(rowIdx, 2);
			dataMatrix.at<double>(rowIdx, 3) = normalizedDataLeftNextLeft_.at<double>(rowIdx, 3);
		}

		gcransac::sampler::UniformSampler sampler(&dataMatrix); // The main sampler is used inside the local optimization

		// Checking if the samplers are initialized successfully.
		if (!sampler.isInitialized())
		{
			fprintf(stderr, "The sampler is not initialized successfully.\n");
			return;
		}

		ProgXPrimeTraditionalHomography progressiveXPrime;

		auto& settings = progressiveXPrime.getMutableSettings();
		settings.inlierOutlierThreshold = inlierOutlierThreshold_;
		settings.modelDistanceThreshold = FLAGS_model_to_model_distance;
		settings.maximumIterations = 5;
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

		poseSelectionFromHomographies(
			dataMatrix,
			models,
			modelData,
			Eigen::Matrix3d::Identity(),
			Eigen::Matrix3d::Identity(),
			generalizedCameraPoses,
			groundTruthPose_,
			inliers_,
			poseScore_,
			estimatedPose_);
	}

	// If there is an untested estimator run the function again
	if constexpr (sizeof...(_EstimatorTypes) > 0) {
		estimateHomography<_EstimatorTypes...>(
			sequence_,
			frame1_,
			frame2_,
			imageLeft_,
			imageRight_,
			imageNextLeft_,
			dataMatrix_,
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
			estimatedPose_,
			inliers_,
			poseScore_,
			inlierOutlierThreshold_);
	}
}

void poseSelection(
	const cv::Mat& dataPoints_,
	const std::vector<Sophus::SE3d>& poses_,
	const Eigen::Matrix3d& sourceIntrinsics_,
	const Eigen::Matrix3d& destinationIntrinsics_,
	const std::vector<Eigen::Matrix<double, 3, 4>>& generalizedCameraPoses_,
	std::vector<size_t>& inliers_,
	double& score_,
	Sophus::SE3d& estimatedPose_)
{
	double scale;
	Eigen::Matrix3d rotation = generalizedCameraPoses_[1].block<3, 3>(0, 0);
	pose::convertToRotationMatrix(rotation, &scale);

	if (rotation.hasNaN())
		return;

	Sophus::SE3d leftToRightCameraPose(
		rotation,
		generalizedCameraPoses_[1].rightCols<1>());
	Sophus::SE3d rightToLeftCameraPose =
		leftToRightCameraPose.inverse();

	size_t mostPoints = 0,
		currentInlierNumber = 0;
	double poseScore = -1e10,
		currentPoseScore = 0.0;
	std::vector<size_t> tmpInliers;

	const Eigen::Matrix<double, 3, 4> projectionSource =
		Eigen::Matrix<double, 3, 4>::Identity();
	Eigen::Matrix<double, 3, 4> projectionDestinationLeft,
		projectionDestinationRight;
	cv::Mat point(1, 4, dataPoints_.type());
	
	// Iterate through the poses to select the best one with the most inliers
	for (size_t poseIdx = 0; poseIdx < poses_.size(); ++poseIdx)
	{
		// Clearing the vector of inliers
		tmpInliers.clear();
		// The score of the current pose
		currentPoseScore = 0.0;
		// The inlier number of the current pose
		currentInlierNumber = 0;
		// The current pose
		const auto &currentPose = poses_[poseIdx];
		// The pose from the right image to the next left
		const auto rightImagePose = currentPose * rightToLeftCameraPose;

		// Projection matrix between the two left images
		projectionDestinationLeft << 
			currentPose.rotationMatrix(), currentPose.translation();
		// Projection matrix between the left and right images
		projectionDestinationRight << 
			rightImagePose.rotationMatrix(), rightImagePose.translation();

		// Iterating through all points to select the inliers
		for (size_t pointIdx = 0; pointIdx < dataPoints_.rows; ++pointIdx)
		{
			// The point-to-pose residual
			double residual;
			// The triangulated 3D point
			Eigen::Matrix<double, 4, 1> triangulatedPoint;
			// For some problems, not the generalized setup is used and, thus,
			// the structure of the correspondence matrix is different.
			if (dataPoints_.cols == 4)
			{
				// The coordinates of the current point
				point.at<double>(0) = dataPoints_.at<double>(pointIdx, 0);
				point.at<double>(1) = dataPoints_.at<double>(pointIdx, 1);
				point.at<double>(2) = dataPoints_.at<double>(pointIdx, 2);
				point.at<double>(3) = dataPoints_.at<double>(pointIdx, 3);

				// Linear triangulation
				// TODO(danini): replace with midpoint which is faster and probably is similarly good for this taks
				pose::linearTriangulation(
					projectionSource, // The projection matrix of the soirce image
					projectionDestinationLeft, // The projection matrix of the destination image
					point, // The point correspondence
					triangulatedPoint); // The triangulated 3D point

				// The 3D point projected in the source image
				Eigen::Vector3d projectedSource =
					projectionSource * triangulatedPoint;

				// If the homogeneous coordinate is negative, the 3D point falls behind the camera.
				if (projectedSource(2) < 0)
					continue;

				// The 3D point projected in the destination image
				Eigen::Vector3d projectedDestination =
					projectionDestinationLeft * triangulatedPoint;

				// If the homogeneous coordinate is negative, the 3D point falls behind the camera.
				if (projectedDestination(2) < 0)
					continue;

				// Calculating the point-to-pose residual
				Eigen::Vector2d pt1(
					point.at<double>(0), point.at<double>(1));
				Eigen::Vector2d pt2(
					point.at<double>(2), point.at<double>(3));

				residual =
					(projectedSource.hnormalized() - pt1).norm() +
					(projectedDestination.hnormalized() - pt2).norm();
			}
			else
			{
				// The coordinates of the current point
				point.at<double>(0) = dataPoints_.at<double>(pointIdx, 9);
				point.at<double>(1) = dataPoints_.at<double>(pointIdx, 10);
				point.at<double>(2) = dataPoints_.at<double>(pointIdx, 0);
				point.at<double>(3) = dataPoints_.at<double>(pointIdx, 1);

				// Selecting the index of the camera in which the point is visible
				const double& cameraIdx = dataPoints_.at<double>(pointIdx, 8);

				// Selecting the projection matrix depending on the camera index
				const Eigen::Matrix<double, 3, 4>& projectionMatrix =
					cameraIdx == 0 ?
					projectionDestinationLeft :
					projectionDestinationRight;

				// The 3D point projected in the source image
				pose::linearTriangulation(
				projectionSource, // The projection matrix of the soirce image
					projectionMatrix, // The projection matrix of the destination image
					point, // The point correspondence
					triangulatedPoint); // The triangulated 3D point

				Eigen::Vector3d projectedSource =
					projectionSource * triangulatedPoint;

				// If the homogeneous coordinate is negative, the 3D point falls behind the camera.
				if (projectedSource(2) < 0)
					continue;

				// The 3D point projected in the destination image
				Eigen::Vector3d projectedDestination =
					projectionMatrix * triangulatedPoint;

				// If the homogeneous coordinate is negative, the 3D point falls behind the camera.
				if (projectedDestination(2) < 0)
					continue;

				// Calculating the point-to-pose residual
				Eigen::Vector2d pt1(
					point.at<double>(0), point.at<double>(1));
				Eigen::Vector2d pt2(
					point.at<double>(2), point.at<double>(3));

				residual =
					(projectedSource.hnormalized() - pt1).norm() +
					(projectedDestination.hnormalized() - pt2).norm();
			}

			// Consider a point inlier if the residual is greater than two times the threshold.
			// Symmetric error is used, thus, the two times. 
			if (residual < 2 * FLAGS_inlier_outlier_threshold)
			{
				// Store the inlier
				tmpInliers.emplace_back(pointIdx);
				// Increase the score of the model accordingly. Currently, MSAC scoring.
				currentPoseScore += 1.0 - residual / (2 * FLAGS_inlier_outlier_threshold);
				// Increase the inlier number
				++currentInlierNumber;
			}
		}

		LOG(INFO) << poseIdx << ". pose has " << currentInlierNumber << " inliers and its score is " <<
			currentPoseScore << ". Maximum is (" << inliers_.size() << ", " << score_ << ")";

		// Store the pose and its inliers if it is the new best
		if (currentPoseScore >= score_)
		{
			inliers_ = tmpInliers;
			score_ = currentPoseScore;
			estimatedPose_ = currentPose;
		}
	}
}

void poseSelectionFromHomographies(
	const cv::Mat &dataPoints_,
	const std::vector<gcransac::Model>& models_,
	const std::vector<progx::ModelData>& modelData_,
	const Eigen::Matrix3d & sourceIntrinsics_,
	const Eigen::Matrix3d & destinationIntrinsics_,
	const std::vector<Eigen::Matrix<double, 3, 4>> &generalizedCameraPoses_,
	const Sophus::SE3d &relativePose_,
	std::vector<size_t> &inliers_,
	double &score_,
	Sophus::SE3d &estimatedPose_)
{
	const size_t & kHomographyNumber = models_.size();

	// Initializing the vector consisting of the poses decomposed from the homographies and, possibly, the essential matrix
	std::vector<Sophus::SE3d> poses;
	poses.reserve(kHomographyNumber * 4);

	// Decompose homographies
	for (size_t homographyIdx = 0; homographyIdx < kHomographyNumber; ++homographyIdx)
	{
		std::vector<Eigen::Vector3d> points3d;
		const Eigen::Matrix3d& homography = 
			models_[homographyIdx].descriptor.cols() > 3 ?
			models_[homographyIdx].descriptor.block<3, 3>(0, 4) :
			models_[homographyIdx].descriptor.block<3, 3>(0, 0);

		const auto& inliers = modelData_[homographyIdx].inliers;
		Eigen::Matrix3d rotation;
		Eigen::Vector3d translation,
			normal;

		std::vector<Eigen::Matrix3d> R_cmbs;
		std::vector<Eigen::Vector3d> t_cmbs;
		std::vector<Eigen::Vector3d> n_cmbs;

		pose::decomposeHomographyMatrix(
			homography,
			sourceIntrinsics_,
			destinationIntrinsics_,
			R_cmbs,
			t_cmbs,
			n_cmbs);

		double scale;
		for (size_t poseIdx = 0; poseIdx < R_cmbs.size(); ++poseIdx)
		{
			if (R_cmbs[poseIdx].hasNaN())
				continue;
			if (R_cmbs[poseIdx].determinant() < 0)
				continue;
			pose::convertToRotationMatrix(R_cmbs[poseIdx], &scale);
			poses.emplace_back(Sophus::SE3d(R_cmbs[poseIdx], t_cmbs[poseIdx]));
		}
	}

	// Select the best pose from the set of homographies
	size_t mostPoints = 0,
		currentInlierNumber = 0;
	double poseScore = -1e10,
		currentPoseScore = 0.0;
	std::vector<size_t> inliers,
		tmpInliers;

	const Eigen::Matrix<double, 3, 4> projectionSource =
		Eigen::Matrix<double, 3, 4>::Identity();
	Eigen::Matrix<double, 3, 4> projectionDestinationLeft,
		projectionDestinationRight;

	double scale;
	Eigen::Matrix3d rotation = generalizedCameraPoses_[1].block<3, 3>(0, 0);
	pose::convertToRotationMatrix(rotation, &scale);

	if (rotation.hasNaN())
		return;

	Sophus::SE3d leftToRightCameraPose(
		rotation,
		generalizedCameraPoses_[1].rightCols<1>());
	Sophus::SE3d rightToLeftCameraPose =
		leftToRightCameraPose.inverse();

	for (size_t poseIdx = 0; poseIdx < poses.size(); ++poseIdx)
	{
		currentPoseScore = 0.0;
		currentInlierNumber = 0;
		const auto currentPose = poses[poseIdx]; // .inverse();
		const auto rightImagePose = currentPose * rightToLeftCameraPose;

		projectionDestinationLeft << currentPose.rotationMatrix(), currentPose.translation();
		projectionDestinationRight << rightImagePose.rotationMatrix(), rightImagePose.translation();

		tmpInliers.clear();

		cv::Mat point(1, 4, dataPoints_.type());

		for (size_t pointIdx = 0; pointIdx < dataPoints_.rows; ++pointIdx)
		{
			double residual;
			Eigen::Matrix<double, 4, 1> triangulatedPoint;
			if (dataPoints_.cols == 4)
			{
				point.at<double>(0) = dataPoints_.at<double>(pointIdx, 0);
				point.at<double>(1) = dataPoints_.at<double>(pointIdx, 1);
				point.at<double>(2) = dataPoints_.at<double>(pointIdx, 2);
				point.at<double>(3) = dataPoints_.at<double>(pointIdx, 3);

				pose::linearTriangulation(
					projectionSource,
					projectionDestinationLeft,
					point,
					triangulatedPoint);

				Eigen::Vector3d projectedSource =
					projectionSource * triangulatedPoint;

				if (projectedSource(2) < 0)
					continue;

				Eigen::Vector3d projectedDestination =
					projectionDestinationLeft * triangulatedPoint;

				if (projectedDestination(2) < 0)
					continue;

				Eigen::Vector2d pt1(
					point.at<double>(0), point.at<double>(1));
				Eigen::Vector2d pt2(
					point.at<double>(2), point.at<double>(3));

				residual =
					(projectedSource.hnormalized() - pt1).norm() +
					(projectedDestination.hnormalized() - pt2).norm();
			}
			else
			{
				point.at<double>(0) = dataPoints_.at<double>(pointIdx, 9);
				point.at<double>(1) = dataPoints_.at<double>(pointIdx, 10);
				point.at<double>(2) = dataPoints_.at<double>(pointIdx, 0);
				point.at<double>(3) = dataPoints_.at<double>(pointIdx, 1);

				const double& cameraIdx = dataPoints_.at<double>(pointIdx, 8);

				const Eigen::Matrix<double, 3, 4>& projectionMatrix =
					cameraIdx == 0 ?
					projectionDestinationLeft :
					projectionDestinationRight;

				pose::linearTriangulation(
					projectionSource,
					projectionMatrix,
					point,
					triangulatedPoint);

				Eigen::Vector3d projectedSource =
					projectionSource * triangulatedPoint;

				if (projectedSource(2) < 0)
					continue;

				Eigen::Vector3d projectedDestination =
					projectionMatrix * triangulatedPoint;

				if (projectedDestination(2) < 0)
					continue;

				Eigen::Vector2d pt1(
					point.at<double>(0), point.at<double>(1));
				Eigen::Vector2d pt2(
					point.at<double>(2), point.at<double>(3));

				residual =
					(projectedSource.hnormalized() - pt1).norm() +
					(projectedDestination.hnormalized() - pt2).norm();
			}

			if (residual < 2 * FLAGS_inlier_outlier_threshold)
			{
				tmpInliers.emplace_back(pointIdx);
				currentPoseScore += 1.0 - residual / (2 * FLAGS_inlier_outlier_threshold);
				++currentInlierNumber;
			}
		}

		LOG(INFO) << poseIdx << ". pose has " << currentInlierNumber << " inliers and its score is " <<
			currentPoseScore << ". Maximum is (" << mostPoints << ", " << poseScore << ")";

		// Compute the angular translation error by using the dot product between translations.
		/*const Eigen::Vector3d referenceTranslationDstSrc = relativePose_.translation().normalized();
		const Eigen::Vector3d translationDstSrc = currentPose.translation().normalized();

		constexpr double rad2Deg = 180.0 / M_PI;
		double translationError = rad2Deg * std::acos(std::clamp(referenceTranslationDstSrc.dot(translationDstSrc), -1.0, 1.0));
		translationError = MIN(rad2Deg * std::acos(std::clamp(referenceTranslationDstSrc.dot(-translationDstSrc), -1.0, 1.0)), translationError);

		Sophus::SO3d rotationDifference = Sophus::SO3d(relativePose_.rotationMatrix()).inverse() * Sophus::SO3d(currentPose.rotationMatrix());
		double rotationError = rad2Deg * rotationDifference.log().norm();
		currentPoseScore = -(translationError + rotationError);*/
		

		if (currentPoseScore >= poseScore)
		{
			inliers = tmpInliers;
			poseScore = currentPoseScore;
			mostPoints = currentInlierNumber;
			estimatedPose_ = currentPose;
		}
	}

	inliers_ = inliers;
	score_ = poseScore;
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

#pragma omp parallel for num_threads(FLAGS_core_number)
		for (int frame = 0; frame < frameNumber; ++frame)
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

////////////////////////////////////////////////////////////////////////////////
// Code from Torsten.
////////////////////////////////////////////////////////////////////////////////

DEFINE_string(colmap_data_path, "d:/Datasets/CambridgeColmapModel/cambridge_1024_colmap_models_orig_gt/ShopFacade/empty_all/", "directory with colmap data");
DEFINE_string(match_dir_path, "d:/Datasets/CambridgeColmapModel/posenet_800_matches/matches_shop/", "directory with the matches");
DEFINE_string(loc_results_outfile, "", "output file for results");
DEFINE_bool(resize_intrinsics, true,
	"Set to true for experiments on Cambridge Landmarks.");

// At the moment, we are ignoring every model that is too complicated.
struct ColmapCamera {
  std::string camera_model;
  int camera_id;
  int width;
  int height;
  std::vector<double> parameters;
};

struct ColmapObservation {
  Eigen::Vector2d x;
  int point_id;
};

struct ColmapImage {
  std::string image_name;
  int image_id;
  int camera_id;
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
  std::vector<ColmapObservation> observations;
};

struct ColmapTrack {
  int image_id;
  int feature_id;
};

struct ColmapPoint {
  int point_id;
  Eigen::Vector3d X;
  Eigen::Vector3i color;
  double error;
  std::vector<ColmapTrack> track;
};

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> Points2D;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Points3D;

struct Match2Dto2D {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Represented by a 2D point in the query image and a 3D ray and ray origin
  // for the 2D point in the reference image. The origin of the ray is the center
  // of the camera in the generalized camera coordinate system.
  Eigen::Vector2d p2D_query;
  Eigen::Vector3d ref_ray_dir;
  Eigen::Vector3d ref_center;
  // 2D feature position in the generalized camera cameras.
  Eigen::Vector2d p2D_db_normalized;
  int camera_id;
};
typedef std::vector<Match2Dto2D, Eigen::aligned_allocator<Match2Dto2D>> Matches2Dto2D;

typedef Eigen::Matrix<double, 3, 4> CameraPose;
typedef std::vector<CameraPose> CameraPoses;

void LoadCamera(const std::string& line, bool load_camera_id,
                ColmapCamera* camera) {
  std::stringstream s_stream(line);

  ColmapCamera& cam = *camera;
  if (load_camera_id) s_stream >> cam.camera_id;
  s_stream >> cam.camera_model >> cam.width >> cam.height;
  if (cam.camera_model.compare("SIMPLE_RADIAL") == 0 ||
      cam.camera_model.compare("VSFM") == 0 ||
      cam.camera_model.compare("PINHOLE") == 0) {
    cam.parameters.resize(4);
    s_stream >> cam.parameters[0] >> cam.parameters[1] >> cam.parameters[2]
             >> cam.parameters[3];
  } else if (cam.camera_model.compare("SIMPLE_PINHOLE") == 0) {
    cam.parameters.resize(3);
    s_stream >> cam.parameters[0] >> cam.parameters[1] >> cam.parameters[2];
  } else {
    std::cout << " camera model " << cam.camera_model << " not supported"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

bool LoadColmapCamera(const std::string& filename,
                      std::vector<ColmapCamera>* cameras) {
  cameras->clear();
  std::ifstream ifs(filename.c_str(), std::ios::in);
  if (!ifs.is_open()){
    std::cout << " cannot read from " << filename << std::endl;
    return false;
  }
  std::string line;

  // Skips the first three lines containing the header.
  for (int i = 0; i < 3; ++i) {
    std::getline(ifs, line);
  }

  while (std::getline(ifs, line)) {
    ColmapCamera cam;
    LoadCamera(line, true, &cam);
    cameras->push_back(cam);
  }

  ifs.close();
  return true;
}

bool LoadColmapImages(const std::string& filename,
                      std::vector<ColmapImage>* images) {
  images->clear();
  std::ifstream ifs(filename.c_str(), std::ios::in);
  if (!ifs.is_open()){
    std::cout << " cannot read from " << filename << std::endl;
    return false;
  }
  std::string line;

  // Skips the first four lines containing the header.
  for (int i = 0; i < 4; ++i) {
    std::getline(ifs, line);
  }

  int count = 0;
  while (std::getline(ifs, line)) {
    std::stringstream s_stream(line);

    ColmapImage img;
    s_stream >> img.image_id >> img.q.w() >> img.q.x() >> img.q.y() >> img.q.z()
             >> img.t[0] >> img.t[1] >> img.t[2] >> img.camera_id
             >> img.image_name;
    img.q.normalize();
    {
      // Skips feature associations.
      std::getline(ifs, line);
      // std::stringstream s_stream2(line);
      // while (!s_stream2.eof()) {
      //   ColmapObservation obs;
      //   s_stream2 >> obs.x[0] >> obs.x[1] >> obs.point_id;
      //   if (obs.point_id != -1) {
      //     img.observations.push_back(obs); 
      //   }
      // }
    }
    
    images->push_back(img);
    ++count;
  }

  ifs.close();
  return true;
}

template <typename T>
double ComputeMedian(std::vector<T>* data) {
  T mean = static_cast<T>(0.0);
  for (size_t i = 0; i < data->size(); ++i) {
    mean += (*data)[i];
  }
  mean /= static_cast<T>(data->size());
  std::cout << " mean : " << mean << std::endl;

  std::sort(data->begin(), data->end());
  if (data->size() % 2u == 1u) {
    return static_cast<double>((*data)[data->size() / 2]);
  } else {
    double a = static_cast<double>((*data)[data->size() / 2 - 1]);
    double b = static_cast<double>((*data)[data->size() / 2]);
    return (a + b) * 0.5;
  }
}

struct QueryData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string name;

  ColmapCamera camera;

  Eigen::Quaterniond q;
  Eigen::Vector3d t;
};

typedef std::vector<QueryData, Eigen::aligned_allocator<QueryData>> Queries;

// Loads the list of query images together with their extrinsics and intrinsics.
bool LoadQueries(const std::string& filename,
                 Queries* query_images) {
  std::ifstream ifs(filename.c_str(), std::ios::in);
  if (!ifs.is_open()) {
    std::cerr << " ERROR: Cannot read the image list from " << filename
              << std::endl;
    return false;
  }
  std::string line;

  query_images->clear();

  while (std::getline(ifs, line)) {
    std::stringstream s_stream(line);

    QueryData q;
    s_stream >> q.name;
    s_stream >> q.camera.camera_model >> q.camera.width >> q.camera.height;
    if (q.camera.camera_model.compare("SIMPLE_RADIAL") == 0 ||
        q.camera.camera_model.compare("VSFM") == 0 ||
        q.camera.camera_model.compare("PINHOLE") == 0) {
      q.camera.parameters.resize(4);
      s_stream >> q.camera.parameters[0] >> q.camera.parameters[1]
               >> q.camera.parameters[2] >> q.camera.parameters[3];
    } else {
      std::cout << " camera model " << q.camera.camera_model << " not supported"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    s_stream >> q.q.w() >> q.q.x() >> q.q.y() >> q.q.z() >> q.t[0] >> q.t[1]
             >> q.t[2];
    q.q.normalize();
    // std::string line;
    // std::getline(s_stream, line);
    // LoadCamera(line, false, &(q.camera));
    query_images->push_back(q);
  }

  ifs.close();

  return true;
}

// Loads 2D-2D matches for two given image names.
bool LoadMatches2D2D(const std::string& match_dir, const std::string& q_name,
                     const std::string& db_name,  
                     Points2D* points2D_q, Points2D* points2D_ref) {
  points2D_q->clear();
  points2D_ref->clear();

  std::stringstream s_stream;
  s_stream << "matches_" << q_name << "_"
           << db_name << ".txt";
  std::string ending(s_stream.str());
  std::string filename(match_dir);
  filename.append(std::regex_replace(ending, std::regex("/"), "_"));

  // std::cout << " Loading matches from " << filename << std::endl;

  std::ifstream ifs(filename.c_str(), std::ios::in);
  if (!ifs.is_open()) {
    std::cerr << " ERROR: Cannot read the matches from " << filename
              << std::endl;
    return false;
  }
  std::string line;

  while (std::getline(ifs, line)) {
    std::stringstream s_stream(line);

    Eigen::Vector2d p_q;
    Eigen::Vector2d p_db;
    s_stream >> p_q[0] >> p_q[1] >> p_db[0] >> p_db[1];

    points2D_q->push_back(p_q);
    points2D_ref->push_back(p_db);
  }
  // std::cout << points2D_q->size() << std::endl;

  return true;
}

bool LoadPairs(const std::string& filename,
               std::unordered_map<std::string, std::vector<std::string>>* pairs) {
  pairs->clear();

  std::ifstream ifs(filename.c_str(), std::ios::in);
  if (!ifs.is_open()) {
    std::cerr << " ERROR: Cannot read the matches from " << filename
              << std::endl;
    return false;
  }
  std::string line;

  while (std::getline(ifs, line)) {
    std::stringstream s_stream(line);

    std::string q_name, db_name;
    s_stream >> q_name >> db_name;

    (*pairs)[q_name].push_back(db_name);
  }

  return true;
}

bool DecomposeHomography(const Eigen::Matrix3d& H,
                         std::vector<Eigen::Matrix3d>* Rs,
                         std::vector<Eigen::Vector3d>* ts,
                         std::vector<Eigen::Vector3d>* ns) {
  Rs->clear();
  ts->clear();
  ns->clear();
  // The following code is based on The Robotics Toolbox for Matlab (RTB) and
  // the corresponding copyright note is replicated below.
  // Copyright (C) 1993-2011, by Peter I. Corke
  //
  // This file is part of The Robotics Toolbox for Matlab (RTB).
  //
  // RTB is free software: you can redistribute it and/or modify
  // it under the terms of the GNU Lesser General Public License as published by
  // the Free Software Foundation, either version 3 of the License, or
  // (at your option) any later version.
  //
  // RTB is distributed in the hope that it will be useful,
  // but WITHOUT ANY WARRANTY; without even the implied warranty of
  // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  // GNU Lesser General Public License for more details.
  //
  // You should have received a copy of the GNU Leser General Public License
  // along with RTB.  If not, see <http://www.gnu.org/licenses/>.
  
  // normalize H so that the second singular value is one
  Eigen::JacobiSVD<Eigen::Matrix3d> svd1(H);
  Eigen::Matrix3d H2 = H / svd1.singularValues()[1];

  // compute the SVD of the symmetric matrix H'*H = VSV'
  Eigen::JacobiSVD<Eigen::Matrix3d> svd2(H2.transpose() * H2,
                                         Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd2.matrixV();
  
  // ensure V is right-handed
  if (V.determinant() < 0.0) V *= -1.0;

  // get the squared singular values
  Eigen::Vector3d S = svd2.singularValues();
  double s1 = S[0];
  double s3 = S[2];

  Eigen::Vector3d v1 = V.col(0);
  Eigen::Vector3d v2 = V.col(1);
  Eigen::Vector3d v3 = V.col(2);

  // pure the case of pure rotation all the singular values are equal to 1
  if (fabs(s1-s3) < 1e-14) {
    return false;
  } else {
    // compute orthogonal unit vectors
    Eigen::Vector3d u1 = (sqrt(1.0-s3)*v1 + sqrt(s1-1.0)*v3) / sqrt(s1-s3);
    Eigen::Vector3d u2 = (sqrt(1.0-s3)*v1 - sqrt(s1-1.0)*v3) / sqrt(s1-s3);

    Eigen::Matrix3d U1, W1, U2, W2;
    U1.col(0) = v2;
    U1.col(1) = u1;
    U1.col(2) = v2.cross(u1);

    W1.col(0) = H2 * v2;
    W1.col(1) = H2 * u1;
    W1.col(2) = (H2 * v2).cross(H2 * u1);

    U2.col(0) = v2;
    U2.col(1) = u2;
    U2.col(2) = v2.cross(u2);

    W2.col(0) = H2 * v2;
    W2.col(1) = H2 * u2;
    W2.col(2) = (H2 * v2).cross(H2 * u2);
    
    // compute the rotation matrices
    Eigen::Matrix3d R1 = W1 * U1.transpose();
    Eigen::Matrix3d R2 = W2 * U2.transpose();
    
    // build the solutions, discard those with negative plane normals
    // Compare to the original code, we do not invert the transformation.
    // Furthermore, we multiply t with -1.
    Eigen::Vector3d n = v2.cross(u1);
    ns->push_back(n);
    Rs->push_back(R1);
    Eigen::Vector3d t = -(H2-R1)*n;
    ts->push_back(t);

    ns->push_back(-n);
    t = (H2-R1)*n;
    Rs->push_back(R1);
    ts->push_back(t);  
    

    n = v2.cross(u2);
    ns->push_back(n);
    t = -(H2-R2)*n;
    Rs->push_back(R2);
    ts->push_back(t);
    
    ns->push_back(-n);
    t = (H2-R2)*n;
    ts->push_back(t);
    Rs->push_back(R2);
  }
  return true;
}

double ComputeTriangulationError(const Eigen::Matrix3d& R1,
	                               const Eigen::Vector3d& c1,
	                               const Eigen::Matrix3d& R2,
	                               const Eigen::Vector3d& c2,
	                               const Eigen::Vector2d& i1,
	                               const Eigen::Vector2d& i2) {
    // Computes the closest point on one line to the second one, then checks
    // if the point is in front of both cameras.
    // See https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points for details.
    Eigen::Vector3d p1 = c1;
    Eigen::Vector3d d1 = R1 * i1.homogeneous();
    d1.normalize();
    Eigen::Vector3d p2 = c2;
    Eigen::Vector3d d2 = R2.transpose() * i2.homogeneous();
    d2.normalize();

    // std::cout << acos(d1.dot(d2)) * 180.0 / M_PI << " degree" << std::endl;

    Eigen::Vector3d n = d1.cross(d2);
    n.normalize();
    Eigen::Vector3d n1 = d1.cross(n);
    n1.normalize();
    Eigen::Vector3d n2 = d2.cross(n);
    n2.normalize();
    
    Eigen::Vector3d pt1 = p1 + (p2 - p1).dot(n2) / (d1.dot(n2)) * d1;

    Eigen::Vector3d pt2 = p2 + (p1 - p2).dot(n1) / (d2.dot(n1)) * d2;

    // std::cout << (pt1-pt2).norm() << std::endl;


    double error = std::numeric_limits<double>::max();

    if ((d1.dot(pt1 - p1) < 0.0) || (d2.dot(pt2 - p2) < 0.0)) {
      return std::numeric_limits<double>::max();
    }

  //   std::cout << pt1.transpose() << std::endl;
  //   std::cout << pt2.transpose() << std::endl;
  //   std::cout << (pt1-pt2).norm() << std::endl;

  //   cv::Mat point(1, 4, CV_64F);
		// point.at<double>(0) = i1[0];
		// point.at<double>(1) = i1[1];
		// point.at<double>(2) = i2[0];
		// point.at<double>(3) = i2[1];
		// Eigen::Matrix<double, 3, 4> P1, P2;
		// P1.topLeftCorner<3, 3>() = R1.transpose();
		// P1.col(3) = -R1.transpose() * c1;
		// P2.topLeftCorner<3, 3>() = R2;
		// P2.col(3) = -R2 * c2;
		// Eigen::Matrix<double, 4, 1> triangulatedPoint;
		// pose::linearTriangulation(
		// 			P1, // The projection matrix of the soirce image
		// 			P2, // The projection matrix of the destination image
		// 			point, // The point correspondence
		// 			triangulatedPoint); 
		// std::cout << triangulatedPoint.hnormalized().transpose() << std::endl;


    Eigen::Vector3d p = R2 * (pt1 - c2);
    if (p[2] < 0.0) return std::numeric_limits<double>::max();
    error = (p.hnormalized() - i2).norm();
    // std::cout << error << " ";
    p = R1.transpose() * (pt2 - c1);
    if (p[2] < 0.0) return std::numeric_limits<double>::max();
    error += (p.hnormalized() - i1).norm();
    // std::cout << (p.hnormalized() - i1).norm() << std::endl;

    return error;
  }

void SelectBestGeneralizedPose(
	const cv::Mat &dataPoints_,
	const std::vector<gcransac::Model>& models_,
	const std::vector<progx::ModelData>& modelData_,
	const std::vector<Eigen::Matrix<double, 3, 4>> &generalizedCameraPoses_,
	const Eigen::Matrix3d& R_gt,
	const Eigen::Vector3d& c_gt,
	std::vector<size_t> &inliers_,
	double &score_,
	Eigen::Matrix3d* R_est,
	Eigen::Vector3d* c_est) {
	const size_t & kHomographyNumber = models_.size();

	// Initializing the vector consisting of the poses decomposed from the
	// homographies and, possibly, the essential matrix
	std::vector<Eigen::Matrix3d> estimated_rotations;
	std::vector<Eigen::Vector3d> estimated_positions;
	estimated_rotations.reserve(kHomographyNumber * 4);
	estimated_positions.reserve(kHomographyNumber * 4);

	// Decompose homographies
	for (size_t homographyIdx = 0; homographyIdx < kHomographyNumber; ++homographyIdx) {
		std::cout << homographyIdx << " : " << modelData_[homographyIdx].inlierNumber() << " inliers" << std::endl;
		gcransac::estimator::solver::GeneralizedHomographyCeresSolver solver(
		  generalizedCameraPoses_, generalizedCameraPoses_.size());
		// std::cout << " calling solver" << std::endl;

		/*std::vector<gcransac::Model> outputModels;
		outputModels.emplace_back(models_[homographyIdx]);
		solver.estimateModel(dataPoints_, &(modelData_[homographyIdx].inliers[0]),
		                     modelData_[homographyIdx].inliers.size(),
		                     outputModels);
		// std::cout << " solver finished" << std::endl; 
		// std::cout << outputModels.size() << " " << outputModels[0].descriptor.cols() << std::endl;

		// const Eigen::Matrix3d& homography = 
		// 	models_[homographyIdx].descriptor.cols() > 3 ?
		// 	models_[homographyIdx].descriptor.block<3, 3>(0, 4) :
		// 	models_[homographyIdx].descriptor.block<3, 3>(0, 0);
		// Eigen::Vector3d h_N = models_[homographyIdx].descriptor.col(3);

		// const auto& inliers = modelData_[homographyIdx].inliers;

		const Eigen::Matrix3d& homography = 
			outputModels[0].descriptor.block<3, 3>(0, 0);
		Eigen::Vector3d h_N = outputModels[0].descriptor.col(3);
		std::cout << h_N.norm() << std::endl;

		const auto& inliers = modelData_[homographyIdx].inliers;

		std::vector<Eigen::Matrix3d> R_cmbs;
		std::vector<Eigen::Vector3d> t_cmbs;
		std::vector<Eigen::Vector3d> n_cmbs;

		DecomposeHomography(homography, &R_cmbs, &t_cmbs, &n_cmbs);*/

		/*Eigen::Vector3d h_N = models_[homographyIdx].descriptor.col(3);
		double scale = h_N.norm();
		*R_est = models_[homographyIdx].descriptor.block<3, 3>(0, 4);
		*c_est = -*R_est * models_[homographyIdx].descriptor.col(7); // model.descriptor.col(7) / scale;*/

		estimated_rotations.emplace_back(models_[homographyIdx].descriptor.block<3, 3>(0, 4));
		estimated_positions.emplace_back(-*R_est * models_[homographyIdx].descriptor.col(7));

		/*double scale = h_N.norm();
		for (size_t poseIdx = 0; poseIdx < R_cmbs.size(); ++poseIdx) {
			if (R_cmbs[poseIdx].hasNaN()) continue;
			if (R_cmbs[poseIdx].determinant() < 0) continue;
			estimated_rotations.emplace_back(R_cmbs[poseIdx]);
			estimated_positions.emplace_back(t_cmbs[poseIdx]/scale);
		}*/
	}

	// Select the best pose from the set of homographies
	double poseScore = -1e10,
		bestError = std::numeric_limits<double>::max(),
		currentPoseScore = 0.0;
	std::vector<size_t> inliers,
		tmpInliers;
	size_t mostPoints = 0,
		currentInlierNumber = 0;


	for (size_t poseIdx = 0; poseIdx < estimated_rotations.size(); ++poseIdx) {
		currentPoseScore = 0.0;
		currentInlierNumber = 0;

		const auto R1 = estimated_rotations[poseIdx];
		const auto c1 = estimated_positions[poseIdx];

		//std::cout << R1 << std::endl << c1.transpose() << std::endl;
// 
		tmpInliers.clear();

		for (size_t pointIdx = 0; pointIdx < dataPoints_.rows; ++pointIdx) {
			
			Eigen::Vector2d p_img1, p_img2;
			// p_img1 << dataPoints_.at<double>(pointIdx, 9), dataPoints_.at<double>(pointIdx, 10);
			// p_img2 << dataPoints_.at<double>(pointIdx, 0), dataPoints_.at<double>(pointIdx, 1);
			p_img1 << dataPoints_.at<double>(pointIdx, 0), dataPoints_.at<double>(pointIdx, 1);
			p_img2 << dataPoints_.at<double>(pointIdx, 9), dataPoints_.at<double>(pointIdx, 10);
			Eigen::Vector3d c2;
			c2 << dataPoints_.at<double>(pointIdx, 5),
			      dataPoints_.at<double>(pointIdx, 6),
			      dataPoints_.at<double>(pointIdx, 7);

			int img_id = static_cast<int>(dataPoints_.at<double>(pointIdx, 8));
			// std::cout << " " << img_id << std::flush;

			Eigen::Matrix3d R2;
			R2 = generalizedCameraPoses_[img_id].topLeftCorner<3, 3>();
			// std::cout << c2.transpose() << std::endl;
		  c2 = -R2.transpose() * generalizedCameraPoses_[img_id].col(3);
		  // std::cout << c2.transpose() << std::endl;
	
	    double residual = ComputeTriangulationError(R1, c1, R2, c2, p_img1, p_img2);

	    // if (pointIdx % 100u == 0) {
	    // 	std::cout << R1 << std::endl << c1.transpose() << std::endl << R2 << std::endl << c2.transpose() << std::endl;
	    // 	std::cout << pointIdx << " " << residual << " " << p_img1.transpose() << " " << p_img2.transpose() << " " << img_id << std::endl;
	    // }

			if (residual < 2 * FLAGS_inlier_outlier_threshold) {
				tmpInliers.emplace_back(pointIdx);
				currentPoseScore += 1.0 - residual / (2 * FLAGS_inlier_outlier_threshold);
				++currentInlierNumber;
			}
		}

		LOG(INFO) << poseIdx << ". pose has " << currentInlierNumber << " inliers and its score is " <<
			currentPoseScore << ". Maximum is (" << mostPoints << ", " << poseScore << ")";		

		Eigen::AngleAxisd aax(R1 * R_gt);
    double rot_error = aax.angle() * 180.0 / M_PI;
    std::cout << " rotation error: " << rot_error << std::endl;
    std::cout << " position error: " << (c1 - c_gt).norm() << std::endl;

		if (bestError >= rot_error) {
			inliers = tmpInliers;
			bestError = rot_error;
			poseScore = currentPoseScore;
			mostPoints = currentInlierNumber;
			*R_est = estimated_rotations[poseIdx];
			*c_est = estimated_positions[poseIdx];
		}
	}

	inliers_ = inliers;
	score_ = poseScore;
}

template<class _Estimator, class ... _EstimatorTypes>
void estimateHomography(const cv::Mat& dataMatrix_,
	                      const CameraPoses& gen_camera_poses_,
	                      const std::vector<std::pair<size_t, size_t>>& match_ranges_,
	                      const Eigen::Matrix3d& R_gt,
												const Eigen::Vector3d& c_gt,
	                      std::vector<size_t>& inliers_,
	                      double& poseScore_,
	                      const double& inlierOutlierThreshold_,
	                      Eigen::Matrix3d* R_est,
	                      Eigen::Vector3d* c_est) {
	gcransac::Model model;

	// Initialize the estimator
	if constexpr (_Estimator::isGeneralizedHomography()) {

		const size_t kNumGenCameras = match_ranges_.size();
		std::vector<size_t> sample_sizes(2, 0u);
		sample_sizes[0] = 3u;
		sample_sizes[1] = 2u;

		if (false)
		{
			_Estimator estimator;
			estimator.setRigPoses(gen_camera_poses_);

			// The main sampler is used inside the local optimization
			gcransac::sampler::UniformGeneralizedSampler sampler(
				&dataMatrix_, match_ranges_, sample_sizes);

			// Checking if the samplers are initialized successfully.
			if (!sampler.isInitialized()) {
				fprintf(stderr, "The sampler is not initialized successfully.\n");
				return;
			}

			ProgXPrime progressiveXPrime;

			progressiveXPrime.getMutableEstimator().setRigPoses(gen_camera_poses_);

			auto& settings = progressiveXPrime.getMutableSettings();
			settings.inlierOutlierThreshold = inlierOutlierThreshold_;
			settings.modelDistanceThreshold = FLAGS_model_to_model_distance;
			settings.maximumIterations = 3;
			settings.minimumInlierNumber = FLAGS_minimum_point_number;
			settings.startingHypothesisNumber = 10;
			settings.addedHypothesisNumber = 10;
			settings.confidence = 0.999;

			std::vector<gcransac::Model> models;
			std::vector<progx::ModelData> modelData;

			std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
			start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
			progressiveXPrime.run(
				dataMatrix_,
				sampler,
				models,
				modelData);
			end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
			std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
			double time = elapsed_seconds.count();

			SelectBestGeneralizedPose(dataMatrix_, models, modelData, gen_camera_poses_,
				R_gt, c_gt, inliers_, poseScore_, R_est, c_est);
		}

		if (true)
		{  
			cv::Mat dataMatrix = dataMatrix_.clone();
			std::vector<gcransac::Model> models;
			std::vector<std::pair<size_t, size_t>> match_ranges = match_ranges_;

			for (size_t iterations = 0; iterations < 10; ++iterations)
			{
				// Simply test GC RANSAC on its own
				using namespace gcransac;

				gcransac::Model tmpModel;
				neighborhood::FlannNeighborhoodGraph neighborhood(new cv::Mat(0, 0, CV_64F), 20);

				_Estimator estimator;
				estimator.setRigPoses(gen_camera_poses_);

				preemption::SPRTPreemptiveVerfication<_Estimator> sprt(
					dataMatrix,
					estimator);
				preemption::EmptyPreemptiveVerfication<_Estimator> emptyPreemption;

				GCRANSAC<_Estimator,
					neighborhood::FlannNeighborhoodGraph,
					MSACScoringFunction<_Estimator>,
					preemption::SPRTPreemptiveVerfication<_Estimator>> gcransac;
				gcransac.settings.threshold = FLAGS_inlier_outlier_threshold; // The inlier-outlier threshold
				gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
				gcransac.settings.confidence = 0.99; // The required confidence in the results
				gcransac.settings.max_local_optimization_number = 20; // The maximum number of local optimizations
				gcransac.settings.max_iteration_number = 10000; // The maximum number of iterations
				gcransac.settings.min_iteration_number = 10000; // The minimum number of iterations
				gcransac.settings.neighborhood_sphere_radius = 20; // The radius of the neighborhood ball
				gcransac.settings.do_local_optimization = true;
				gcransac.settings.do_final_iterated_least_squares = false;

				sampler::UniformGeneralizedSampler main_sampler(
					&dataMatrix,
					match_ranges,
					sample_sizes); // The main sampler is used inside the local optimization
				sampler::UniformSampler local_optimization_sampler(&dataMatrix); // The local optimization sampler is used inside the local optimization

				// Checking if the samplers are initialized successfully.
				if (!main_sampler.isInitialized() ||
					!local_optimization_sampler.isInitialized())
				{
					fprintf(stderr, "One of the samplers is not initialized successfully.\n");
					return;
				}

				// Start GC-RANSAC
				gcransac.run(dataMatrix,
					estimator,
					&main_sampler,
					&local_optimization_sampler,
					&neighborhood,
					tmpModel,
					sprt);

				// Get the statistics of the results

				// Get the statistics of the results
				const utils::RANSACStatistics& statistics_ =
					gcransac.getRansacStatistics();

				const size_t& inlierNumber = statistics_.inliers.size();
				std::cout << " GC-RANSAC: " << inlierNumber << std::endl;

				if (inlierNumber < 10)
					break;

				models.emplace_back(tmpModel);

				std::vector<bool> mask(dataMatrix.rows, true);
				for (const auto& inlierIdx : statistics_.inliers)
					mask[inlierIdx] = false;

				cv::Mat newDataMatrix(dataMatrix.rows - inlierNumber, dataMatrix.cols, dataMatrix.type());
				size_t currentRow = 0;
				for (size_t pointIdx = 0; pointIdx < dataMatrix.rows; ++pointIdx)
				{
					if (!mask[pointIdx])
						continue;
					dataMatrix.row(pointIdx).copyTo(newDataMatrix.row(currentRow));
					++currentRow;
				}

				dataMatrix.release();
				dataMatrix = newDataMatrix.clone();

				match_ranges.clear();
				match_ranges.resize(2);
				match_ranges[0].first = 0;
				size_t currentImage = 0;
				for (size_t pointIdx = 0; pointIdx < dataMatrix.rows; ++pointIdx)
				{
					const size_t cameraIdx = dataMatrix.at<double>(pointIdx, 8);
					if (cameraIdx != currentImage)
					{
						++currentImage;
						match_ranges[currentImage].first = pointIdx;
					}

					match_ranges[currentImage].second = pointIdx;
				}
			}

			double bestScale = 0;
			double bestError = std::numeric_limits<double>::max();
			for (const auto& currentModel : models)
			{
				Eigen::Vector3d h_N = currentModel.descriptor.col(3);
				double scale = h_N.norm();

				Eigen::AngleAxisd aax(currentModel.descriptor.block<3, 3>(0, 4) * R_gt);
				double rotError = aax.angle() * 180.0 / M_PI;

				if (rotError < bestError)
				{
					model = currentModel;
					*R_est = currentModel.descriptor.block<3, 3>(0, 4);
					*c_est = -*R_est * currentModel.descriptor.col(7); // model.descriptor.col(7) / scale;
					bestError = rotError;
					bestScale = scale;
				}

			}

			static double
				s1 = 0, s2 = 0, s3 = 0, s4 = 0;
			static int cnt = 0;

			s1 += abs((-*R_est * model.descriptor.col(7) / bestScale - c_gt).norm());
			s2 += abs((*R_est * model.descriptor.col(7) / bestScale - c_gt).norm());
			s3 += abs((-*R_est * model.descriptor.col(7) - c_gt).norm());
			s4 += abs((*R_est * model.descriptor.col(7) - c_gt).norm());
			++cnt;

			printf("Errors = %f\t%f\t%f\t%f\n", 
				s1 / cnt, s2 / cnt, s3 / cnt, s4 / cnt);


			/*std::vector<Eigen::Matrix3d> R_cmbs;
			std::vector<Eigen::Vector3d> t_cmbs;
			std::vector<Eigen::Vector3d> n_cmbs;

			DecomposeHomography(
				model.descriptor.block<3, 3>(0, 0), 
				&R_cmbs, 
				&t_cmbs, 
				&n_cmbs);

			size_t rNum = R_cmbs.size();
			for (size_t poseIdx = 0; poseIdx < rNum; ++poseIdx) {
				R_cmbs.emplace_back(R_cmbs[poseIdx].transpose());
				t_cmbs.emplace_back(t_cmbs[poseIdx]);
				n_cmbs.emplace_back(n_cmbs[poseIdx]);
			}

			Eigen::Vector3d h_N = model.descriptor.col(3);
			double scale = h_N.norm();
			double bestError = std::numeric_limits<double>::max();
			for (size_t poseIdx = 0; poseIdx <  R_cmbs.size(); ++poseIdx) {
				if (R_cmbs[poseIdx].hasNaN()) continue;
				if (R_cmbs[poseIdx].determinant() < 0) continue;

				Eigen::AngleAxisd aax(R_cmbs[poseIdx] * R_gt);
				double rot_error = aax.angle() * 180.0 / M_PI;

				if (rot_error < bestError)
				{
					*R_est = R_cmbs[poseIdx];
					*c_est = t_cmbs[poseIdx] / scale;
					bestError = rot_error;
				}
			}*/
    }
	} else {
		std::cerr << " Unsupported solver type" << std::endl;
	}

	// If there is an untested estimator run the function again
	if constexpr (sizeof...(_EstimatorTypes) > 0) {
		estimateHomography<_EstimatorTypes...>(
			dataMatrix_, gen_camera_poses_, match_ranges_, R_gt, c_gt, 
			inliers_,
			poseScore_,
			inlierOutlierThreshold_, R_est, c_est);
	}
}


int ProcessStructureLessLocalization() {
	std::vector<ColmapImage> images;
  {
    std::string image_file(FLAGS_colmap_data_path);
    image_file.append("images.txt");
    if (!LoadColmapImages(image_file, &images)) {
      std::cerr << " ERROR: Cannot load images from " << image_file << std::endl;
      return -1;
    }
  }

  std::unordered_map<int, int> map_image_id_to_idx;
  std::unordered_map<std::string, int> map_image_name_to_idx;
  std::vector<Eigen::Vector3d> camera_centers(images.size());
  for (int i = 0; i < static_cast<int>(images.size()); ++i) {
    map_image_id_to_idx[images[i].image_id] = i;
    map_image_name_to_idx[images[i].image_name] = i;
    camera_centers[i] = -Eigen::Matrix3d(images[i].q).transpose() * images[i].t;
  }

  std::vector<ColmapCamera> cameras;
  {
    std::string cam_file(FLAGS_colmap_data_path);
    cam_file.append("cameras.txt");
    if (!LoadColmapCamera(cam_file, &cameras)) {
      std::cerr << " ERROR: Cannot load cameras from " << cam_file << std::endl;
      return -1;
    }
  }
  std::unordered_map<int, int> map_cam_id_to_idx;
  for (int i = 0; i < static_cast<int>(cameras.size()); ++i) {
    map_cam_id_to_idx[cameras[i].camera_id] = i;
    if (FLAGS_resize_intrinsics) {
    	// std::cout << "Resizing" << std::endl;
	    double scaling = 800.0 / static_cast<double>(std::max(cameras[i].width,
	                                                          cameras[i].height));
	    // std::cout << scaling << std::endl;
	    cameras[i].width = static_cast<int>(static_cast<double>(cameras[i].width)
	    	                                                             * scaling);
	    cameras[i].height = static_cast<int>(static_cast<double>(cameras[i].height)
	    	                                                             * scaling);
	    // Assumes that the camera has four parameters.
	    cameras[i].parameters[0] *= scaling;
	    cameras[i].parameters[1] *= scaling;
	    cameras[i].parameters[2] *= scaling;
	    cameras[i].parameters[3] *= scaling;
		}
  }

  std::cout << "  -> done loading COLMAP data" << std::endl;
  std::cout << "      Found " << images.size() << " images and "
            << cameras.size() << " cameras" << std::endl;

  std::string pairs_file(FLAGS_match_dir_path);
  pairs_file.append("densevlad_top10.txt");
  std::unordered_map<std::string, std::vector<std::string>> pairs;
  if (!LoadPairs(pairs_file, &pairs)) {
    std::cerr << " ERROR: Could not read the pairs from " << pairs_file
              << std::endl;
    return -1;
  }

  // std::uniform_int_distribution<int> distribution(1, 5);

  Queries query_data;
  std::string list(FLAGS_match_dir_path);
  list.append("list_queries_with_intrinsics_and_gt_pose.txt");


  if (!LoadQueries(list, &query_data)) {
    std::cerr << " ERROR: Could not read the data from " << list << std::endl;
    return -1;
  }
  const int kNumQuery = static_cast<int>(query_data.size());
  std::cout << " Found " << kNumQuery << " query images " << std::endl;


  std::vector<double> orientation_errors, position_errors;

  double average_ransac_time = 0.0;
  int num_valid = 0;

#pragma omp parallel for num_threads(FLAGS_core_number)
  for (int i = 0; i < kNumQuery; ++i) {
  	std::cout << " Processing " << query_data[i].name << std::endl;
    const std::vector<std::string>& pairs_i = pairs[query_data[i].name];

    Matches2Dto2D matches;
    CameraPoses gen_camera_poses;
    std::vector<std::pair<size_t, size_t>> match_ranges;

    double q_fx = query_data[i].camera.parameters[0];
    double q_fy = q_fx;

    int num_inliers = 0;

    Eigen::Matrix3d R1 = query_data[i].q.toRotationMatrix().transpose(); 
    //Eigen::Vector3d c1 = R1 * query_data[i].t;
	Eigen::Vector3d c1 = -R1 * query_data[i].t;

    //std::cout << FLAGS_inlier_outlier_threshold << std::endl;

    for (const std::string& p : pairs_i) 
	{
      Points2D points2d_q;
      Points2D points2d_db;
      if (!LoadMatches2D2D(FLAGS_match_dir_path, query_data[i].name, p,
      	                   &points2d_q, &points2d_db)) {
        std::cerr << "  ERROR: Could not load matches for " << query_data[i].name
                  << " and reference image " << p << std::endl;
        continue;
      }

      const int kNumMatches = static_cast<int>(points2d_q.size());
      if (kNumMatches <= 5) {
      	continue;
      }

      match_ranges.push_back(std::make_pair(matches.size(), 0u));

      const int kImgIdx = map_image_name_to_idx[p];
      const int kCamIdx = map_cam_id_to_idx[images[kImgIdx].camera_id];
      CameraPose P;
      P.topLeftCorner<3, 3>() = images[kImgIdx].q.toRotationMatrix();
      P.col(3) = images[kImgIdx].t;
      // std::cout << p << std::endl;
      // std::cout << P << std::endl;
      // std::cout << images[kImgIdx].q.toRotationMatrix() << std::endl;

      Eigen::Vector3d c_db = -P.topLeftCorner<3, 3>().transpose() * P.col(3);


      // Prepare the keypoints.
      Eigen::Vector2d pp_q, pp_db;
      double fx_db = cameras[kCamIdx].parameters[0];
      double fy_db = fx_db;
      
      if (query_data[i].camera.camera_model.compare("PINHOLE") == 0) {
        q_fy = query_data[i].camera.parameters[1];
        pp_q << query_data[i].camera.parameters[2],
                query_data[i].camera.parameters[3];
      } else {
        pp_q << query_data[i].camera.parameters[1],
                query_data[i].camera.parameters[2];
      }
      if (cameras[kCamIdx].camera_model.compare("PINHOLE") == 0) {
      	fy_db = cameras[kCamIdx].parameters[1];
        pp_db << cameras[kCamIdx].parameters[2], 
			cameras[kCamIdx].parameters[3];
      } else {
        pp_db << cameras[kCamIdx].parameters[1], cameras[kCamIdx].parameters[2];
      }

     // std::cout << fx_db << " " << fy_db << std::endl;

      for (int j = 0; j < kNumMatches; ++j) {
        Match2Dto2D m;
        m.p2D_query = points2d_q[j] - pp_q;
        m.p2D_query[0] /= q_fx;  //query_data[i].camera.parameters[0];
        m.p2D_query[1] /= q_fy;
        m.p2D_db_normalized = points2d_db[j] - pp_db;
        m.p2D_db_normalized[0] /= fx_db;
        m.p2D_db_normalized[1] /= fy_db; 
        // std::cout << m.p2D_query.norm() << " " << m.p2D_db_normalized.norm() << std::endl;
        m.ref_ray_dir = P.topLeftCorner<3, 3>().transpose() *  m.p2D_db_normalized.homogeneous();
        m.ref_ray_dir.normalize();
        // std::cout << "2D measurement: " << points2d_db[j].transpose() << std::endl;
        // std::cout << m.p2D_db_normalized.transpose() << std::endl;
        // std::cout << m.ref_ray_dir.transpose() << std::endl;
        // std::cout << P.topLeftCorner<3, 3>() << std::endl;
        m.ref_center = c_db;
        m.camera_id = static_cast<int>(gen_camera_poses.size());

        matches.push_back(m);

        Eigen::Vector3d c2 = -images[kImgIdx].q.toRotationMatrix().transpose() * images[kImgIdx].t;
        double error = ComputeTriangulationError(R1, c1, images[kImgIdx].q.toRotationMatrix(), c2, m.p2D_query, m.p2D_db_normalized);
        // if ((matches.size() - 1u) % 100u == 0) {
        // 	std::cout << matches.size() - 1u << " " << error << " " << m.p2D_query.transpose() << " " << m.p2D_db_normalized.transpose() << " " << m.camera_id << std::endl;
        // }
        if (error < 2.0 * FLAGS_inlier_outlier_threshold) ++num_inliers;
      }
      match_ranges.back().second = matches.size() - 1u;
      gen_camera_poses.push_back(P);

      if (kNumMatches >= 5) ++num_valid;
    }
    std::cout << "   Found " << num_inliers << " inliers" << std::endl;
    // return -1;

    // if (num_valid < 2) {
    //   std::cerr << " Warning: Found matches only with one or less images."
    //             << " Skipping this test image."
    // }
    const int kNumMatches = static_cast<int>(matches.size());

    std::cout << "  Applying homography fitting to " << kNumMatches << " matches"
              << std::endl;
    

    if (kNumMatches < 10 || num_valid < 2) {
      orientation_errors.push_back(std::numeric_limits<double>::max());
      position_errors.push_back(std::numeric_limits<double>::max());
      continue;
    }

    // Initializes the OpenCV matrix that is used as input to progressive
    // x-prime.
    num_inliers = 0;
    cv::Mat dataMatrix(kNumMatches, 11, CV_64F);
    for (int j = 0; j < kNumMatches; ++j) {
    	// dataMatrix.at<double>(j, 0) = matches[j].p2D_db_normalized[0];
    	// dataMatrix.at<double>(j, 1) = matches[j].p2D_db_normalized[1];
    	dataMatrix.at<double>(j, 0) = matches[j].p2D_query[0];
    	dataMatrix.at<double>(j, 1) = matches[j].p2D_query[1];
    	dataMatrix.at<double>(j, 2) = matches[j].ref_ray_dir[0];
    	dataMatrix.at<double>(j, 3) = matches[j].ref_ray_dir[1];
    	dataMatrix.at<double>(j, 4) = matches[j].ref_ray_dir[2];
    	dataMatrix.at<double>(j, 5) = matches[j].ref_center[0];
    	dataMatrix.at<double>(j, 6) = matches[j].ref_center[1];
    	dataMatrix.at<double>(j, 7) = matches[j].ref_center[2];
    	dataMatrix.at<double>(j, 8) = matches[j].camera_id;
    	// dataMatrix.at<double>(j, 9) = matches[j].p2D_query[0];
    	// dataMatrix.at<double>(j, 10) = matches[j].p2D_query[1];
    	dataMatrix.at<double>(j, 9) = matches[j].p2D_db_normalized[0];
    	dataMatrix.at<double>(j, 10) = matches[j].p2D_db_normalized[1];

    	// Eigen::Vector2d p_img1, p_img2;
    	// p_img1 << dataMatrix.at<double>(j, 0), dataMatrix.at<double>(j, 1);
    	// p_img2 << dataMatrix.at<double>(j, 9), dataMatrix.at<double>(j, 10);
    	// Eigen::Matrix3d R2 = gen_camera_poses[dataMatrix.at<double>(j, 8)].topLeftCorner<3, 3>();
    	// Eigen::Vector3d c2 = -R2.transpose() * gen_camera_poses[dataMatrix.at<double>(j, 8)].col(3);
     //  double error = ComputeTriangulationError(R1, c1, R2, c2, p_img1, p_img2);
     //  if (j % 100 == 0) {
     //  	std::cout << R1 << std::endl << c1.transpose() << std::endl << R2 << std::endl << c2.transpose() << std::endl;
     //  	std::cout << j << " " << error << " " << p_img1.transpose() << " " << p_img2.transpose() << " " << dataMatrix.at<double>(j, 8) << std::endl;
     //  }
     //  if (error < 2.0 * FLAGS_inlier_outlier_threshold) ++num_inliers;
    }
    // std::cout << "  new inlier count " << num_inliers << std::endl;

    double pose_score = -1.0;
    std::vector<size_t> inliers;
    Eigen::Matrix3d R_est;
    Eigen::Vector3d c_est;

   // std::cout << "  GT Pose: " << std::endl;
    Eigen::Matrix3d R_gt = query_data[i].q.toRotationMatrix();
   // std::cout << R_gt.transpose() << std::endl;
    Eigen::Vector3d c_gt = -query_data[i].q.toRotationMatrix().transpose() * query_data[i].t;
   /* std::cout << c_gt.transpose() << std::endl;
    std::cout << query_data[i].q.coeffs().transpose() << std::endl;
    std::cout << query_data[i].t.transpose() << std::endl;*/

    estimateHomography<GeneralizedEstimator32>(dataMatrix, gen_camera_poses,
	                      match_ranges, // The estimated relative pose
	                      R_gt, c_gt, 
	                      inliers, pose_score,
	                      FLAGS_inlier_outlier_threshold,
	                      &R_est, &c_est);

    // orientation_errors.push_back(std::numeric_limits<double>::max());
    // position_errors.push_back(std::numeric_limits<double>::max());

    Eigen::AngleAxisd aax(R_est * R_gt);
    double rot_error = aax.angle() * 180.0 / M_PI;  //std::acos(0.5 * ((R_gt * best_model.R).trace() - 1)) * 180.0 / M_PI;
    std::cout << "orientation: " << aax.angle() * 180.0 / M_PI << " deg, ";
    std::cout << "position: " << (c_est - c_gt).norm() << std::endl;
    // // std::cout << std::endl;
    // // std::cout << std::acos(0.5 * ((R_gt * best_model.R).trace() - 1)) * 180.0 / M_PI << std::endl;
    // // std::cout << aax.axis().transpose() << std::endl;

    orientation_errors.push_back(rot_error);
    position_errors.push_back((c_est - c_gt).norm());
  }

  std::cout << std::endl << std::endl;
  std::cout << "/////////////////////" << std::endl;
  std::cout << "// Statistics" << std::endl;
  double median_pos = ComputeMedian(&position_errors);
  double median_orient = ComputeMedian(&orientation_errors);
  std::cout << "// average RANSAC time: " << average_ransac_time << std::endl;
  std::cout << "// median position error: " << median_pos << std::endl;
  std::cout << "// median orientation error: " << median_orient << std::endl;
  std::ofstream ofs(FLAGS_loc_results_outfile.c_str(), std::ios::out);
  if (!ofs.is_open()) {
    std::cerr << " ERROR: Cannot write to " << FLAGS_loc_results_outfile
              << std::endl;
    return -1;
  }
  ofs << median_pos << " / " << median_orient << " & " << std::endl;
  ofs.close();
	
	return 0;
}

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Running tests on the KITTI dataset
	ProcessStructureLessLocalization();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// End code from Torsten.
////////////////////////////////////////////////////////////////////////////////
