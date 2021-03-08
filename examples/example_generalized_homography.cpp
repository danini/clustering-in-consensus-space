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

DEFINE_double(inlier_outlier_threshold, 0.0005,
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

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Running tests on the KITTI dataset
	processKITTI();

	return 0;
}

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
	Eigen::Vector3d t;
	t << 0, 0, 1;
	t *= (groundTruthPoses_[frameDestination_].translation() - groundTruthPoses_[frameSource_].translation()).norm();
	relativePose = Sophus::SE3d(relativePose.rotationMatrix(), t);

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