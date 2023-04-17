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
	const size_t& kSamplerId_);

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
	const size_t& kSamplerId_);

int findTwoViewMotions_(
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
	const size_t& kSamplerId_);

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
	const size_t& kSamplerId_);
	
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
	const size_t& kSamplerId_);

void getLabeling_(
	std::vector<double>& points_,
	std::vector<double>& models_,
	const int &kModelType_,
	const double &kInlierOutlierThreshold_,
	const double &kNeighborhoodSize_,
	const double &kSpatialWeight_,
	const double &kLabelCost_,
	std::vector<size_t> &labeling_,
	size_t &maxLabel_);


void getSoftLabeling_(
	std::vector<double>& points_,
	std::vector<double>& models_,
	const int &kModelType_,
	const double &kInlierOutlierThreshold_,
	std::vector<double> &labeling_);