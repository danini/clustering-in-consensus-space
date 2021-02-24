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

#include <mutex>

/*
	Initializing the flags
*/
DEFINE_string(data_path, "d:/Kutatas/ProgressiveXPrime/matlab/multilinesegment/data/",
	"The folder where the images are stored.");
DEFINE_string(statistics_path, "lineSegmentResults.csv",
	"The folder where the results are saved in .csv format.");
DEFINE_int32(number_of_test_images, 321,
	"The number of test scenes in the folder.");
DEFINE_bool(draw_results, true,
	"A flag determining if the results should be drawn and shown.");
DEFINE_double(confidence, 0.9999,
	"The confidence of the multi-model fitting.");
DEFINE_double(threshold, 1.5,
	"The inlier-outlier threshold.");
DEFINE_int32(maximum_iterations, 100,
	"The maximum round of multi-model fitting.");
DEFINE_int32(starting_hypothesis_number, 5,
	"The number of hypotheses proposed before the optimization starts.");
DEFINE_int32(added_hypothesis_number, 5,
	"The number of hypotheses proposed in each round.");
DEFINE_double(model_to_model_distance, 0.3,
	"The maximum accepted similarity in the clustering. A values in-between [0, 1].");
DEFINE_int32(minimum_point_number, 3,
	"The minimum number of points required to keep a model.");
DEFINE_int32(core_number, 1,
	"The number of cores used for processing the dataset.");
DEFINE_int32(repetitions, 5,
	"The number of repetitions.");
/*DEFINE_int32(sampler, 0,
	"The used sampler. (0) Uniform, (1) P-NAPSAC");
DEFINE_int32(loss, 0,
	"The used loss function. (0) MAGSAC++, (1) Tukey-bisquare, (2) Huber, (3) Redescending Huber, (4) MSAC.");
DEFINE_int32(clustering, 0,
	"The used clustering algorithm. (0) DBScan, (1) Mean-Shift, (2) Median-Shift.");*/

// Applying Progressive-X
typedef progx::ProgressiveXPrime<
	clustering::density::MeanShiftClustering<
	progx::ModelData,
	clustering::distances::TanimotoDistance<progx::ModelData>>,
	clustering::distances::TanimotoDistance<progx::ModelData>,
	clustering::losses::MAGSACLoss<double, progx::utils::Default2DLineEstimator, 2>,
	progx::utils::Default2DLineEstimator,
	gcransac::sampler::UniformSampler> ProgXPrime;

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

std::mutex writing_mutex;

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (FLAGS_core_number > 1 && FLAGS_draw_results)
	{
		fprintf(stderr, "We do not recommend to set the --draw-results flag true while the --core-number is not 1\n");
		return 0;
	}

	printf("The used parameters are\n\tInlier-outlier threshold %f\n\tMinimum point number = %d\n\tMaximum iteration number = %d\n\tConfidence = %f\n\tModel-to-model distance = %f\nStarting number of models = %d\n\tAdded models = %d\n",
		FLAGS_threshold, FLAGS_minimum_point_number, FLAGS_maximum_iterations, FLAGS_confidence, FLAGS_model_to_model_distance, FLAGS_starting_hypothesis_number, FLAGS_added_hypothesis_number);

#pragma omp parallel for num_threads(FLAGS_core_number)
	for (size_t imageIdx = 1; imageIdx <= FLAGS_number_of_test_images; ++imageIdx)
	{
		const std::string data_path =
			FLAGS_data_path + "point2d_" + std::to_string(imageIdx) + ".txt";
		const std::string ground_truth_path =
			FLAGS_data_path + "poly_" + std::to_string(imageIdx) + ".txt";

		for (size_t repetition = 0; repetition < FLAGS_repetitions; ++repetition)
			testMulti2DLineFitting(
				data_path,
				ground_truth_path,
				FLAGS_confidence,
				FLAGS_threshold,
				FLAGS_maximum_iterations,
				FLAGS_starting_hypothesis_number,
				FLAGS_added_hypothesis_number,
				FLAGS_model_to_model_distance,
				FLAGS_minimum_point_number,
				FLAGS_draw_results);
	}

	return 0;
}

void testMulti2DLineFitting(
	const std::string &data_path_,
	const std::string &ground_truth_path_,
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t starting_hypothesis_number_,
	const size_t added_hypothesis_number_,
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_) // A flag to determine if the results should be visualized
{
	static std::mutex savingMutex;

	// Load the point coordinates
	cv::Mat points;
	gcransac::utils::loadPointsFromFile<2, 1, false>(
		points,
		data_path_.c_str());

	// Loading the ground thruth polygon's coordinates
	cv::Mat groundTruthPolygon;
	gcransac::utils::loadPointsFromFile<2, 1, false>(
		groundTruthPolygon,
		ground_truth_path_.c_str());

	// The main sampler is used inside the local optimization
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::sampler::ProgressiveNapsacSampler<2> sampler(&points, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		progx::utils::Default2DLineEstimator::sampleSize(), // The size of a minimal sample
		{ static_cast<double>(256), // The width of the source image
			static_cast<double>(256) }); // The height of the source image
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("P-NAPSAC initialization time = %f secs.\n", elapsed_seconds.count());

	gcransac::sampler::UniformSampler uniform_sampler(&points);

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
		uniform_sampler,
		models,
		modelData);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsed_seconds = end - start; // The elapsed time in seconds
	double time = elapsed_seconds.count();

	// Calculate the end points of the line segments
	std::vector<Eigen::Vector4d> lineSegments(models.size());
	for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
	{
		const double
			& a = models[modelIdx].descriptor(0),
			& b = models[modelIdx].descriptor(1),
			& c = models[modelIdx].descriptor(2);

		Eigen::Vector2d lineTangent;
		lineTangent << b, -a;

		Eigen::Vector2d pointOnLine;
		pointOnLine << 0, -c / b;

		const auto& inliers = modelData[modelIdx].inliers;

		if (inliers.size() < minimum_point_number_)
			continue;

		auto& lineSegment = lineSegments[modelIdx];
		double minParameter = std::numeric_limits<double>::max(),
			maxParameter = std::numeric_limits<double>::lowest(),
			parameter;
		Eigen::Vector2d point;

		// Projecting the point to the line
		Eigen::Matrix3d coefficients;
		coefficients << 2, 0, a,
			0, 2, b,
			a, b, 0;
		const Eigen::Matrix3d &coefficientsTransposed =
			coefficients;
		const Eigen::Matrix3d covariance =
			coefficientsTransposed * coefficients;
		Eigen::Vector3d inhomogeneousPart;
		inhomogeneousPart(2) = -c;

		for (const auto& inlierIdx : inliers)
		{
			inhomogeneousPart(0) = 2 * points.at<double>(inlierIdx, 0);
			inhomogeneousPart(1) = 2 * points.at<double>(inlierIdx, 1);

			point = covariance.llt().solve(coefficientsTransposed * inhomogeneousPart).head<2>();
			parameter = (point(0) - pointOnLine(0)) / lineTangent(0);

			if (parameter < minParameter)
			{
				minParameter = parameter;
				lineSegment(0) = point(1);
				lineSegment(1) = point(0);
			}

			if (parameter > maxParameter)
			{
				maxParameter = parameter;
				lineSegment(2) = point(1);
				lineSegment(3) = point(0);
			}
		}
	}

	if (models.size() > 0)
	{
		// Calculate the error from the ground truth
		Eigen::VectorXd distances(groundTruthPolygon.rows);
		for (size_t pointIdx = 0; pointIdx < groundTruthPolygon.rows; ++pointIdx)
			distances(pointIdx) = std::numeric_limits<double>::max();

		for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
		{
			const auto& lineSegment = lineSegments[modelIdx];

			Eigen::Vector3d v1, v2;
			v1 << lineSegment.head<2>(), 1;
			v2 << lineSegment.tail<2>(), 1;

			double d = 0;

			for (size_t segmentIdx = 0; segmentIdx < groundTruthPolygon.rows; ++segmentIdx)
			{
				Eigen::Vector3d gtSegment;
				gtSegment(0) = groundTruthPolygon.at<double>(segmentIdx, 0);
				gtSegment(1) = groundTruthPolygon.at<double>(segmentIdx, 1);
				gtSegment(2) = 1;

				Eigen::Vector3d
					a = v1 - v2,
					b = gtSegment - v2,
					c = gtSegment - v1;

				double D1 = abs(a.cross(b)(2)) / sqrt(a(0) * a(0) + a(1) * a(1));
				double D2 = sqrt(c(0) * c(0) + c(1) * c(1));
				double D3 = sqrt(b(0) * b(0) + b(1) * b(1));

				bool insegment = a.dot(b) * (-a).dot(c) >= 0;
				if (insegment)
					d = D1;
				else
					d = MIN(D2, D3);

				distances(segmentIdx) = 
					MIN(distances(segmentIdx), d);
			}
		}

		const double meanError = distances.mean();
		printf("The mean error is %f px.\n", meanError);
		printf("The processing time is %f secs.\n", time);

		savingMutex.lock();
		std::ofstream file(FLAGS_statistics_path, std::fstream::app);
		file <<
			inlier_outlier_threshold_ << ";" <<
			maximum_tanimoto_similarity_ << ";" <<
			maximum_iterations << ";" <<
			minimum_point_number_ << ";" <<
			starting_hypothesis_number_ << ";" <<
			added_hypothesis_number_ << ";" <<
			confidence_ << ";" <<
			meanError << ";" <<
			time << ";" <<
			models.size() << "\n";
		savingMutex.unlock();
	}
	else
	{
		savingMutex.lock();
		std::ofstream file(FLAGS_statistics_path, std::fstream::app);
		file <<
			inlier_outlier_threshold_ << ";" <<
			maximum_tanimoto_similarity_ << ";" <<
			maximum_iterations << ";" <<
			minimum_point_number_ << ";" <<
			starting_hypothesis_number_ << ";" <<
			added_hypothesis_number_ << ";" <<
			confidence_ << ";" <<
			std::numeric_limits<double>::max() << ";" <<
			time << ";" <<
			models.size() << "\n";
		savingMutex.unlock();
	}

	if (visualize_results_)
	{
		cv::Mat image = cv::Mat::zeros(256, 256, CV_8UC3);

		for (size_t pointIdx = 0; pointIdx < points.rows; ++pointIdx)
			cv::circle(image,
				(cv::Point2d)points.row(pointIdx),
				1,
				cv::Scalar(255, 255, 255),
				-1);

		for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
		{
			cv::Scalar color(rand() / (double)RAND_MAX * 255.0,
				rand() / (double)RAND_MAX * 255.0,
				rand() / (double)RAND_MAX * 255.0);

			for (const auto& inlierIdx : modelData[modelIdx].inliers)
				cv::circle(image,
					(cv::Point2d)points.row(inlierIdx),
					1,
					color,
					-1);

			const double
				& a = models[modelIdx].descriptor(0),
				& b = models[modelIdx].descriptor(1),
				& c = models[modelIdx].descriptor(2);

			cv::line(
				image,
				cv::Point(0, -c / b),
				cv::Point(255, (-255 * a - c) / b),
				cv::Scalar(255, 0, 0),
				1);

			// Draw the line segments
			cv::line(
				image,
				cv::Point(lineSegments[modelIdx](1), lineSegments[modelIdx](0)),
				cv::Point(lineSegments[modelIdx](3), lineSegments[modelIdx](2)),
				cv::Scalar(0, 255, 0),
				1);
		}

		cv::resize(image, image, cv::Size(image.cols * 2, image.rows * 2));
		cv::imshow("Image", image);
		cv::waitKey(0);
	}
}