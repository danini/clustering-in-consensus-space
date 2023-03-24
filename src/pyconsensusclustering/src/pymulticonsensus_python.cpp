#include "pymulticonsensus_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

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

#include <ctime>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>

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
	return 0;
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
	const size_t& kSamplerId_)
{
	// Initializing the estimator object
	progx::utils::DefaultHomographyEstimator estimator;

	const size_t num_tents = correspondences_.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences_[0]);

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		kNeighborhoodRadius_); // The radius of the neighborhood ball for determining the neighborhoods.

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	constexpr size_t kSampleSize = gcransac::utils::DefaultHomographyEstimator::sampleSize();
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (kSamplerId_ == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (kSamplerId_ == 1) // Initializing a Progressive NAPSAC sampler. This requires the points to be ordered according to the quality.
	{
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (kSourceImageWidth_ / 16) * (kSourceImageHeight_ / 16)  * (kDestinationImageWidth_ / 16)  (kDestinationImageHeight_ / 16), etc.
			kSampleSize, // The size of a minimal sample
			{ static_cast<double>(kSourceImageWidth_), // The width of the source image
				static_cast<double>(kSourceImageHeight_), // The height of the source image
				static_cast<double>(kDestinationImageWidth_), // The width of the destination image
				static_cast<double>(kDestinationImageHeight_) }, // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	}
	else if (kSamplerId_ == 2) // Initializing a NAPSAC sampler
	{
		//main_sampler = std::unique_ptr<AbstractSampler>(
		//	new gcransac::sampler::NapsacSampler<gcransac::neighborhood::FlannNeighborhoodGraph>(&points, &neighborhood));
	}
	else if (kSamplerId_ == 3) 
	{
		main_sampler = std::unique_ptr<AbstractSampler>(
			new gcransac::sampler::ConnectedComponentSampler(&points,
				estimator.sampleSize(),
				20,
				200,
				5,
				false));
	}
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			kSamplerId_);
		return 0;
	}

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	typedef progx::MultiConsensusFitting<
		clustering::density::DBScanClustering<
		progx::ModelData,
		clustering::distances::TanimotoDistance<progx::ModelData>>,
		clustering::distances::TanimotoDistance<progx::ModelData>,
		clustering::losses::MAGSACLoss<double, progx::utils::DefaultFundamentalMatrixEstimator, 4>,
		progx::utils::DefaultFundamentalMatrixEstimator,
		AbstractSampler> ActualMultiConsensusMethod;

	ActualMultiConsensusMethod progressiveXPrime;

	auto& settings = progressiveXPrime.getMutableSettings();
	settings.inlierOutlierThreshold = kInlierOutlierThreshold_;
	settings.modelDistanceThreshold = kMaximumTanimotoSimilarity_;
	settings.maximumIterations = kMaximumIterations_;
	settings.minimumInlierNumber = kMinimumPointNumber_;
	settings.startingHypothesisNumber = kStartingHypothesisNumber_;
	settings.addedHypothesisNumber = kAddedHypothesisNumber_;
	settings.confidence = kConfidence_;

	std::vector<gcransac::Model> models;
	std::vector<progx::ModelData> modelData;

	progressiveXPrime.run(
		points,
		*main_sampler,
		models,
		modelData);

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
	const std::vector<double>& sourcePoints,
	const std::vector<double>& destinationPoints,
	std::vector<size_t>& labeling,
	std::vector<double>& homographies,
	const double& spatial_coherence_weight,
	const double& threshold,
	const double& confidence,
	const double& neighborhood_ball_radius,
	const double& maximum_tanimoto_similarity,
	const size_t& max_iters,
	const size_t& minimum_point_number,
	const int& maximum_model_number)
{
	return 0;
}