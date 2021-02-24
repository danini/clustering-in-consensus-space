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

#include <gflags/gflags.h>
#include <sophus/se3.hpp>

#include "pose_averaging/openMVG.hpp"

#include <mutex>

DEFINE_string(data_path, "",
	"The path where all the data are found.");
DEFINE_string(source_image_path, "",
	"The path to the source image.");
DEFINE_string(destination_image_path, "",
	"The path to the destination image.");
DEFINE_string(source_intrinsics_path, "",
	"The path to the intrinsic parameters of the source image.");
DEFINE_string(destination_intrinsics_path, "",
	"The path to the intrinsic parameters of the destination image.");
DEFINE_string(source_pose_path, "",
	"The path to the pose parameters of the source image.");
DEFINE_string(destination_pose_path, "",
	"The path to the pose parameters of the destination image.");
DEFINE_string(workspace_path, "",
	"This is where the algorithm saves the results.");

DEFINE_double(confidence, 0.9999,
	"The confidence of the multi-model fitting.");
DEFINE_double(threshold, 2.0,
	"The inlier-outlier threshold for homography fitting.");
DEFINE_double(epipolar_geometry_threshold, 0.75,
	"The inlier-outlier threshold for essential matrix fitting.");
DEFINE_int32(maximum_iterations, 100,
	"The maximum round of multi-model fitting.");
DEFINE_int32(starting_hypothesis_number, 10,
	"The number of hypotheses proposed before the optimization starts.");
DEFINE_int32(added_hypothesis_number, 10,
	"The number of hypotheses proposed in each round.");
DEFINE_double(model_to_model_distance, 0.8,
	"The maximum accepted similarity in the clustering. A values in-between [0, 1].");
DEFINE_int32(minimum_point_number, 20,
	"The minimum number of points required to keep a model.");
DEFINE_bool(visualize_results, true,
	"A flag to determine if the results should be visualized.");
DEFINE_bool(estimate_essential_matrix, true,
	"A flag to determine if the essential matrix should also be estimated.");

// Typedef for the clustering in the high-dimensional consensus space
typedef clustering::density::DBScanClustering<
	progx::ModelData,
	clustering::distances::TanimotoDistance<progx::ModelData>> ClusteringMethod;

// The exact type of the multi-model fitting algorithm
typedef progx::ProgressiveXPrime<
	// The currently used clustering technique
	ClusteringMethod, 
	// Using Tanimoto distance as the model-to-model distances
	clustering::distances::TanimotoDistance<progx::ModelData>, 
	// Using MAGSAC++-based weights both in the iteratively re-weighted LSQ fitting and for the model representation in the consensus space.
	clustering::losses::MAGSACLoss<double, gcransac::utils::DefaultHomographyEstimator, 4>, 
	// We are looking for homographies, thus the homography estimator as specified here.
	progx::utils::DefaultHomographyEstimator,
	// The sampler used for finding minimal samples
	gcransac::sampler::ProgressiveNapsacSampler<4>> ProgXPrime;

std::mutex writing_mutex;

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	

	return 0;
}