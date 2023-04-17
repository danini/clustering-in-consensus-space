#include "pymulticonsensus_python.h"
#include <iostream>
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include "GCRANSAC.h"
#include "mcons_utils.h"
#include "losses.h"
#include "multi_consensus_fitting.h"

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
#include "samplers/napsac_sampler.h"
#include "connected_component_sampler.h"

#include "modified_fundamental_estimator.h"
#include "solver_homography_three_point.h"
#include "modified_homography_estimator.h"
#include "subspace4_estimator.h"
#include "estimators/essential_estimator.h"
#include "vanishing_point_estimator.h"
#include "solver_vanishing_point_two_lines.h"

#include "preemption/preemption_sprt.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>

typedef gcransac::estimator::VanishingPointEstimator<
	gcransac::estimator::solver::VanishingPointTwoLineSolver, // The solver used for fitting a model to a minimal sample
	gcransac::estimator::solver::VanishingPointTwoLineSolver> // The solver used for fitting a model to a non-minimal sample
	DefaultVanishingPointEstimator;

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& poses,
	const double& spatial_coherence_weight,
	const double& threshold,
	const double& confidence,
	const double& neighborhood_ball_radius,
	const double& maximum_tanimoto_similarity,
	const size_t& max_iters,
	const size_t& minimum_point_number,
	const int& maximum_model_number)
{
	fprintf(stderr, "Multiple 6D pose fitting is not yet implemented.");
	return 0;
}

int findVanishingPoints_(
	std::vector<double>& lines_,
	std::vector<double>& vanishingPoints_,
	const size_t& kImageWidth_,
	const size_t& kImageHeight_,
	const double& kInlierOutlierThreshold_,
	const double& kConfidence_,
	const double& kNeighborhoodRadius_,
	const double& kMaximumTanimotoSimilarity_,
	const size_t& kStartingHypothesisNumber_,
	const size_t& kAddedHypothesisNumber_,
	const size_t& kMaximumIterations_,
	const size_t& kMinimumPointNumber_,
	const size_t& kSamplerId_)
{
	fprintf(stderr, "Multiple vanishing point fitting is not yet implemented.");
	return 0;

	// Initializing the estimator object
	DefaultVanishingPointEstimator estimator;

	const size_t num_tents = lines_.size() / 4;
	cv::Mat lines(num_tents, 4, CV_64F, &lines_[0]);
	
	// Initializing the neighborhood structure based on the provided paramereters
	typedef gcransac::neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = DefaultVanishingPointEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (kSamplerId_ == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&lines));
	else 
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling)\n",
			kSamplerId_);
		return 0;
	}

	std::vector<gcransac::Model> models;
	std::vector<mcons::ModelData> modelData;

	typedef mcons::MultiConsensusFitting<
		clustering::density::DBScanClustering<
		mcons::ModelData,
		clustering::distances::TanimotoDistance<mcons::ModelData>>,
		clustering::distances::TanimotoDistance<mcons::ModelData>,
		clustering::losses::MAGSACLoss<double, DefaultVanishingPointEstimator, 4>,
		DefaultVanishingPointEstimator,
		AbstractSampler> ActualMultiConsensusMethod;

	ActualMultiConsensusMethod multiConsensusMethod;

	auto& settings = multiConsensusMethod.getMutableSettings();
	settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
	settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
	settings.maximumIterations = kMaximumIterations_;
	settings.minimumInlierNumber = kMinimumPointNumber_;
	settings.startingHypothesisNumber = kStartingHypothesisNumber_;
	settings.addedHypothesisNumber = kAddedHypothesisNumber_;
	settings.confidence = kConfidence_;

	multiConsensusMethod.run(
		lines,
		*main_sampler,
		models,
		modelData);

	vanishingPoints_.reserve(3 * models.size());

	// Saving the homography parameters
	for (const auto &model : models)
	{
		vanishingPoints_.emplace_back(model.descriptor(0));
		vanishingPoints_.emplace_back(model.descriptor(1));
		vanishingPoints_.emplace_back(model.descriptor(2));
	}

	return models.size();
}


int findHomographies_(
	std::vector<double>& correspondences_,
	std::vector<double>& homographies_,
	const size_t& kSourceImageWidth_,
	const size_t& kSourceImageHeight_,
	const size_t& kDestinationImageWidth_,
	const size_t& kDestinationImageHeight_,
	const double& kInlierOutlierThreshold_,
	const double& kConfidence_,
	const double& kNeighborhoodRadius_,
	const double& kMaximumTanimotoSimilarity_,
	const size_t& kStartingHypothesisNumber_,
	const size_t& kAddedHypothesisNumber_,
	const size_t& kMaximumIterations_,
	const size_t& kMinimumPointNumber_,
	const double &kMinimumComponentDistance_,
	const double &kMaximumComponentDistance_,
	const int &kComponentPartition_,
	const size_t& kSamplerId_)
{
	// Initializing the estimator object
	mcons::utils::DefaultHomographyEstimator estimator;

	const size_t num_tents = correspondences_.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences_[0]);
	
	// Initializing the neighborhood structure based on the provided paramereters
	typedef gcransac::neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	if (kSamplerId_ == 2)
		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new gcransac::neighborhood::FlannNeighborhoodGraph(&points, kNeighborhoodRadius_));
	else if (kSamplerId_ == 3)
		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new gcransac::neighborhood::BruteForceNeighborhoodGraph(&points, kNeighborhoodRadius_));
			
	std::vector<gcransac::Model> models;
	std::vector<mcons::ModelData> modelData;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultHomographyEstimator::sampleSize();

	if (kSamplerId_ == 0) // Initializing a RANSAC-like uniformly random sampler
	{
		gcransac::sampler::UniformSampler main_sampler(&points);

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
				mcons::ModelData,
				clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, mcons::utils::DefaultHomographyEstimator, 4>,
			mcons::utils::DefaultHomographyEstimator,
			gcransac::sampler::UniformSampler> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	

	} else if (kSamplerId_ == 1) // Initializing a Progressive NAPSAC sampler. This requires the points to be ordered according to the quality.
	{		
		gcransac::sampler::ProgressiveNapsacSampler<4> main_sampler(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (kSourceImageWidth_ / 16) * (kSourceImageHeight_ / 16)  * (kDestinationImageWidth_ / 16)  (kDestinationImageHeight_ / 16), etc.
			kSampleSize, // The size of a minimal sample
			{ static_cast<double>(kSourceImageWidth_), // The width of the source image
				static_cast<double>(kSourceImageHeight_), // The height of the source image
				static_cast<double>(kDestinationImageWidth_), // The width of the destination image
				static_cast<double>(kDestinationImageHeight_) } // The height of the destination image
			); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
				mcons::ModelData,
				clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, mcons::utils::DefaultHomographyEstimator, 4>,
			mcons::utils::DefaultHomographyEstimator,
			gcransac::sampler::ProgressiveNapsacSampler<4>> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	
	}
	else if (kSamplerId_ == 2) 
	{
		if (kMinimumComponentDistance_ > kMaximumComponentDistance_ ||
			kComponentPartition_ < 0)
		{
			fprintf(stderr, "Incorrect parameters for the Connected Component sampler. ");
			return 0;			
		}

		gcransac::sampler::ConnectedComponentSampler main_sampler(&points,
				estimator.sampleSize(),
				kMinimumComponentDistance_,
				kMaximumComponentDistance_,
				kComponentPartition_);

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
				mcons::ModelData,
				clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, mcons::utils::DefaultHomographyEstimator, 4>,
			mcons::utils::DefaultHomographyEstimator,
			gcransac::sampler::ConnectedComponentSampler> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (P-NAPSAC sampling), 2 (NAPSAC sampling on FLANN neighborhood), 3 (NAPSAC sampling on BF neighborhood), 4 (CC Sampler)\n",
			kSamplerId_);
		return 0;
	}

	homographies_.reserve(9 * models.size());

	// Saving the homography parameters
	for (const auto &model : models)
	{
		homographies_.emplace_back(model.descriptor(0, 0));
		homographies_.emplace_back(model.descriptor(0, 1));
		homographies_.emplace_back(model.descriptor(0, 2));
		homographies_.emplace_back(model.descriptor(1, 0));
		homographies_.emplace_back(model.descriptor(1, 1));
		homographies_.emplace_back(model.descriptor(1, 2));
		homographies_.emplace_back(model.descriptor(2, 0));
		homographies_.emplace_back(model.descriptor(2, 1));
		homographies_.emplace_back(model.descriptor(2, 2));
	}

	return models.size();
}

int findTwoViewMotions_(
	std::vector<double>& correspondences_,
	std::vector<double>& motions_,
	const size_t& kSourceImageWidth_,
	const size_t& kSourceImageHeight_,
	const size_t& kDestinationImageWidth_,
	const size_t& kDestinationImageHeight_,
	const double& kInlierOutlierThreshold_,
	const double& kConfidence_,
	const double& kNeighborhoodRadius_,
	const double& kMaximumTanimotoSimilarity_,
	const size_t& kStartingHypothesisNumber_,
	const size_t& kAddedHypothesisNumber_,
	const size_t& kMaximumIterations_,
	const size_t& kMinimumPointNumber_,
	const double &kMinimumComponentDistance_,
	const double &kMaximumComponentDistance_,
	const int &kComponentPartition_,
	const size_t& kSamplerId_)
{
	// Initializing the estimator object
	mcons::utils::DefaultFundamentalMatrixEstimator estimator;

	const size_t num_tents = correspondences_.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences_[0]);
	
	// Initializing the neighborhood structure based on the provided paramereters
	typedef gcransac::neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultFundamentalMatrixEstimator::sampleSize();

	std::vector<gcransac::Model> models;
	std::vector<mcons::ModelData> modelData;

	if (kSamplerId_ == 0) // Initializing a RANSAC-like uniformly random sampler
	{
		gcransac::sampler::UniformSampler main_sampler(&points);

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
			mcons::ModelData,
			clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, mcons::utils::DefaultFundamentalMatrixEstimator, 4>,
			mcons::utils::DefaultFundamentalMatrixEstimator,
			gcransac::sampler::UniformSampler> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	

	} else if (kSamplerId_ == 1) // Initializing a Progressive NAPSAC sampler. This requires the points to be ordered according to the quality.
	{		
		gcransac::sampler::ProgressiveNapsacSampler<4> main_sampler(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (kSourceImageWidth_ / 16) * (kSourceImageHeight_ / 16)  * (kDestinationImageWidth_ / 16)  (kDestinationImageHeight_ / 16), etc.
			kSampleSize, // The size of a minimal sample
			{ static_cast<double>(kSourceImageWidth_), // The width of the source image
				static_cast<double>(kSourceImageHeight_), // The height of the source image
				static_cast<double>(kDestinationImageWidth_), // The width of the destination image
				static_cast<double>(kDestinationImageHeight_) } // The height of the destination image
			); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
			mcons::ModelData,
			clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, mcons::utils::DefaultFundamentalMatrixEstimator, 4>,
			mcons::utils::DefaultFundamentalMatrixEstimator,
			gcransac::sampler::ProgressiveNapsacSampler<4>> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	
	}
	else if (kSamplerId_ == 2) 
	{
		gcransac::sampler::ConnectedComponentSampler main_sampler(&points,
				estimator.sampleSize(),
				kMinimumComponentDistance_,
				kMaximumComponentDistance_,
				kComponentPartition_);

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
			mcons::ModelData,
			clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, mcons::utils::DefaultFundamentalMatrixEstimator, 4>,
			mcons::utils::DefaultFundamentalMatrixEstimator,
			gcransac::sampler::ConnectedComponentSampler> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (P-NAPSAC sampling), 2 (NAPSAC sampling on FLANN neighborhood), 3 (NAPSAC sampling on BF neighborhood), 4 (CC Sampler)\n",
			kSamplerId_);
		return 0;
	}	

	motions_.reserve(9 * models.size());

	// Saving the homography parameters
	for (const auto &model : models)
	{
		motions_.emplace_back(model.descriptor(0, 0));
		motions_.emplace_back(model.descriptor(0, 1));
		motions_.emplace_back(model.descriptor(0, 2));
		motions_.emplace_back(model.descriptor(1, 0));
		motions_.emplace_back(model.descriptor(1, 1));
		motions_.emplace_back(model.descriptor(1, 2));
		motions_.emplace_back(model.descriptor(2, 0));
		motions_.emplace_back(model.descriptor(2, 1));
		motions_.emplace_back(model.descriptor(2, 2));
	}

	return models.size();
}

int findRigidMotions_(
	std::vector<double>& subspaces_,
	std::vector<double>& motions_,
	const double& kInlierOutlierThreshold_,
	const double& kConfidence_,
	const double& kNeighborhoodRadius_,
	const double& kMaximumTanimotoSimilarity_,
	const size_t& kStartingHypothesisNumber_,
	const size_t& kAddedHypothesisNumber_,
	const size_t& kMaximumIterations_,
	const size_t& kMinimumPointNumber_,
	const double &kMinimumComponentDistance_,
	const double &kMaximumComponentDistance_,
	const int &kComponentPartition_,
	const size_t& kSamplerId_)
{
	fprintf(stderr, "Multiple rigid motion fitting is not yet implemented.");
	return 0;

	// Initializing the estimator object
	mcons::utils::DefaultLinearSubspaceEstimator estimator;

	const size_t num_tents = subspaces_.size() / 5;
	cv::Mat points(num_tents, 5, CV_64F, &subspaces_[0]);
	
	// Initializing the neighborhood structure based on the provided paramereters
	typedef gcransac::neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = mcons::utils::DefaultLinearSubspaceEstimator::sampleSize();

	std::vector<gcransac::Model> models;
	std::vector<mcons::ModelData> modelData;

	if (kSamplerId_ == 0) // Initializing a RANSAC-like uniformly random sampler
	{
		gcransac::sampler::UniformSampler main_sampler(&points);

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
			mcons::ModelData,
			clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, mcons::utils::DefaultLinearSubspaceEstimator, 4>,
			mcons::utils::DefaultLinearSubspaceEstimator,
			gcransac::sampler::UniformSampler> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	

		std::cout << modelData[0].inliers.size() << std::endl;
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling)\n",
			kSamplerId_);
		return 0;
	}	

	motions_.reserve(5 * models.size());

	// Saving the homography parameters
	for (const auto &model : models)
	{
		motions_.emplace_back(model.descriptor(0));
		motions_.emplace_back(model.descriptor(1));
		motions_.emplace_back(model.descriptor(2));
		motions_.emplace_back(model.descriptor(3));
		motions_.emplace_back(model.descriptor(4));
	}

	return models.size();	
}


int findPlanes_(
	std::vector<double>& correspondences_,
	std::vector<double>& planes_,
	const double& kInlierOutlierThreshold_,
	const double& kConfidence_,
	const double& kNeighborhoodRadius_,
	const double& kMaximumTanimotoSimilarity_,
	const size_t& kStartingHypothesisNumber_,
	const size_t& kAddedHypothesisNumber_,
	const size_t& kMaximumIterations_,
	const size_t& kMinimumPointNumber_,
	const double &kMinimumComponentDistance_,
	const double &kMaximumComponentDistance_,
	const int &kComponentPartition_,
	const size_t& kSamplerId_)
{
	fprintf(stderr, "Multiple plane fitting is not yet implemented.");
	return 0;

	// Initializing the estimator object
	gcransac::utils::Default3DPlaneEstimator estimator;

	const size_t num_tents = correspondences_.size() / 3;
	cv::Mat points(num_tents, 3, CV_64F, &correspondences_[0]);

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::Default3DPlaneEstimator::sampleSize();

	std::vector<gcransac::Model> models;
	std::vector<mcons::ModelData> modelData;

	if (kSamplerId_ == 0) // Initializing a RANSAC-like uniformly random sampler
	{
		gcransac::sampler::UniformSampler main_sampler(&points);

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
			mcons::ModelData,
			clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, gcransac::utils::Default3DPlaneEstimator, 4>,
			gcransac::utils::Default3DPlaneEstimator,
			gcransac::sampler::UniformSampler> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	

	} else if (kSamplerId_ == 1) // Initializing a Progressive NAPSAC sampler. This requires the points to be ordered according to the quality.
	{		
		gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, kNeighborhoodRadius_);
		gcransac::sampler::NapsacSampler main_sampler(&points, &neighborhood); 

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
			mcons::ModelData,
			clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, gcransac::utils::Default3DPlaneEstimator, 4>,
			gcransac::utils::Default3DPlaneEstimator,
			gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	
	}
	else if (kSamplerId_ == 2) 
	{
		gcransac::neighborhood::BruteForceNeighborhoodGraph neighborhood(&points, kNeighborhoodRadius_);
		gcransac::sampler::NapsacSampler main_sampler(&points, &neighborhood); 

		typedef mcons::MultiConsensusFitting<
			clustering::density::DBScanClustering<
			mcons::ModelData,
			clustering::distances::TanimotoDistance<mcons::ModelData>>,
			clustering::distances::TanimotoDistance<mcons::ModelData>,
			clustering::losses::MAGSACLoss<double, gcransac::utils::Default3DPlaneEstimator, 4>,
			gcransac::utils::Default3DPlaneEstimator,
			gcransac::sampler::NapsacSampler<gcransac::neighborhood::BruteForceNeighborhoodGraph>> ActualMultiConsensusMethod;

		ActualMultiConsensusMethod multiConsensusMethod;

		auto& settings = multiConsensusMethod.getMutableSettings();
		settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
		settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
		settings.maximumIterations = kMaximumIterations_;
		settings.minimumInlierNumber = kMinimumPointNumber_;
		settings.startingHypothesisNumber = kStartingHypothesisNumber_;
		settings.addedHypothesisNumber = kAddedHypothesisNumber_;
		settings.confidence = kConfidence_;

		multiConsensusMethod.run(
			points,
			main_sampler,
			models,
			modelData);	
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (P-NAPSAC sampling), 2 (NAPSAC sampling on FLANN neighborhood), 3 (NAPSAC sampling on BF neighborhood), 4 (CC Sampler)\n",
			kSamplerId_);
		return 0;
	}	

	planes_.reserve(4 * models.size());

	// Saving the homography parameters
	for (const auto &model : models)
	{
		planes_.emplace_back(model.descriptor(0));
		planes_.emplace_back(model.descriptor(1));
		planes_.emplace_back(model.descriptor(2));
		planes_.emplace_back(model.descriptor(3));
	}

	return models.size();	
}

void getLabeling_(
	std::vector<double>& points_,
	std::vector<double>& models_,
	const int &kModelType_,
	const double &kInlierOutlierThreshold_,
	const double &kNeighborhoodSize_,
	const double &kSpatialWeight_,
	const double &kLabelCost_,
	std::vector<size_t> &labeling_,
	size_t &maxLabel_)
{
	// Retrieving the data and model dimensions depending on the model type
	size_t dataDimension;
	size_t modelDimensionRow, modelDimensionCol;
	if (kModelType_ == 0 || kModelType_ == 1)
	{
		dataDimension = 4;
		modelDimensionRow = 3;
		modelDimensionCol = 3;
	} else if (kModelType_ == 2)
	{
		dataDimension = 4;
		modelDimensionRow = 3;
		modelDimensionCol = 1;		
	} else if (kModelType_ == 3)
	{
		dataDimension = 3;
		modelDimensionRow = 4;
		modelDimensionCol = 1;		
	} else if (kModelType_ == 4)
	{
		dataDimension = 5;
		modelDimensionRow = 5;
		modelDimensionCol = 1;		
	}

	// Calculating the point number
	const size_t kNumPoints = points_.size() / dataDimension;
	// Calculating the model number
	const size_t kNumModels = models_.size() / (modelDimensionRow * modelDimensionCol);

	// Converting the points to OpenCV format
	cv::Mat points(kNumPoints, dataDimension, CV_64F, &points_[0]);

	// Converting the models to GC-RANSAC format
	std::vector<gcransac::Model> models(kNumModels);
	size_t idx = 0;
	for (size_t modelIdx = 0; modelIdx < kNumModels; ++modelIdx)
	{
		models[modelIdx].descriptor.resize(modelDimensionRow, modelDimensionCol);
		for (size_t row = 0; row < modelDimensionRow; ++row)
			for (size_t col = 0; col < modelDimensionCol; ++col)
				models[modelIdx].descriptor(row, col) = models_[idx++];
	}

	switch (kModelType_)
	{
		case 0:
			getLabeling<mcons::utils::DefaultHomographyEstimator>(
				points,
				models,
				kNeighborhoodSize_,
				kInlierOutlierThreshold_,
				kSpatialWeight_,
				kLabelCost_,
				labeling_,
				maxLabel_);
			break;
		case 1:
			getLabeling<mcons::utils::DefaultFundamentalMatrixEstimator>(
				points,
				models,
				kNeighborhoodSize_,
				kInlierOutlierThreshold_,
				kSpatialWeight_,
				kLabelCost_,
				labeling_,
				maxLabel_);
			break;
		case 2:
			getLabeling<DefaultVanishingPointEstimator>(
				points,
				models,
				kNeighborhoodSize_,
				kInlierOutlierThreshold_,
				kSpatialWeight_,
				kLabelCost_,
				labeling_,
				maxLabel_);
			break;
		case 3:
			getLabeling<gcransac::utils::Default3DPlaneEstimator>(
				points,
				models,
				kNeighborhoodSize_,
				kInlierOutlierThreshold_,
				kSpatialWeight_,
				kLabelCost_,
				labeling_,
				maxLabel_);
			break;
		case 4:
			getLabeling<mcons::utils::DefaultLinearSubspaceEstimator>(
				points,
				models,
				kNeighborhoodSize_,
				kInlierOutlierThreshold_,
				kSpatialWeight_,
				kLabelCost_,
				labeling_,
				maxLabel_);
			break;
		default:
			fprintf(stderr, "Model type %d is not yet implemented.\n", kModelType_);
	}
}

void getSoftLabeling_(
	std::vector<double>& points_,
	std::vector<double>& models_,
	const int &kModelType_,
	const double &kInlierOutlierThreshold_,
	std::vector<double> &labeling_)
{
	// Retrieving the data and model dimensions depending on the model type
	size_t dataDimension;
	size_t modelDimensionRow, modelDimensionCol;
	if (kModelType_ == 0 || kModelType_ == 1)
	{
		dataDimension = 4;
		modelDimensionRow = 3;
		modelDimensionCol = 3;
	} else if (kModelType_ == 2)
	{
		dataDimension = 4;
		modelDimensionRow = 3;
		modelDimensionCol = 1;		
	} else if (kModelType_ == 3)
	{
		dataDimension = 3;
		modelDimensionRow = 4;
		modelDimensionCol = 1;		
	} else if (kModelType_ == 4)
	{
		dataDimension = 5;
		modelDimensionRow = 5;
		modelDimensionCol = 1;		
	}

	// Calculating the point number
	const size_t kNumPoints = points_.size() / dataDimension;
	// Calculating the model number
	const size_t kNumModels = models_.size() / (modelDimensionRow * modelDimensionCol);

	// Converting the points to OpenCV format
	cv::Mat points(kNumPoints, dataDimension, CV_64F, &points_[0]);

	// Converting the models to GC-RANSAC format
	std::vector<gcransac::Model> models(kNumModels);
	size_t idx = 0;
	for (size_t modelIdx = 0; modelIdx < kNumModels; ++modelIdx)
	{
		models[modelIdx].descriptor.resize(modelDimensionRow, modelDimensionCol);
		for (size_t row = 0; row < modelDimensionRow; ++row)
			for (size_t col = 0; col < modelDimensionCol; ++col)
				models[modelIdx].descriptor(row, col) = models_[idx++];
	}

	// The estimator object
	typedef gcransac::estimator::Estimator < cv::Mat, gcransac::Model > AbstractEstimator;
	std::unique_ptr<AbstractEstimator> estimator;

	// Use homography model
	if (kModelType_ == 0)
		estimator = std::unique_ptr<AbstractEstimator>(new mcons::utils::DefaultHomographyEstimator());
	else if (kModelType_ == 1)
		estimator = std::unique_ptr<AbstractEstimator>(new mcons::utils::DefaultFundamentalMatrixEstimator());
	else if (kModelType_ == 2)
		estimator = std::unique_ptr<AbstractEstimator>(new DefaultVanishingPointEstimator());
	else if (kModelType_ == 3)
		estimator = std::unique_ptr<AbstractEstimator>(new gcransac::utils::Default3DPlaneEstimator());
	else if (kModelType_ == 4)
		estimator = std::unique_ptr<AbstractEstimator>(new mcons::utils::DefaultLinearSubspaceEstimator());
		
	// Calculating the squared threshold
	const double kSquaredThreshold = kInlierOutlierThreshold_ * kInlierOutlierThreshold_;

	// Initializing the soft labeling
	labeling_.resize(kNumPoints * kNumModels, 0.0);

	// Iterate through all points to get the point-to-model residuals
	double squaredResidual;
	idx = 0;
	for (size_t pointIdx = 0; pointIdx < kNumPoints; ++pointIdx)
	{
		// The current point's coordinates
		const cv::Mat &kPoint = points.row(pointIdx);

		// Iterate through all models
		for (size_t modelIdx = 0; modelIdx < kNumModels; ++modelIdx)
		{
			squaredResidual = estimator->squaredResidual(kPoint, models[modelIdx]);
			if (squaredResidual > kSquaredThreshold)
			{
				++idx;
				continue;
			}

			labeling_[idx++] = 
				exp(-0.5 * squaredResidual / kSquaredThreshold);
		}
	}
}