#include <vector>
#include <string>

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& homographies,
	const double& spatial_coherence_weight,
	const double& threshold,
	const double& confidence,
	const double& neighborhood_ball_radius,
	const double& maximum_tanimoto_similarity,
	const size_t& max_iters,
	const size_t& minimum_point_number,
	const int& maximum_model_number);

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
	const size_t& kSamplerId);

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
	const int& maximum_model_number);