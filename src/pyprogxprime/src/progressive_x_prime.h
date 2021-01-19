namespace progx
{
	struct MultiModelSettings
	{
		// The settings of the proposal engine
		gcransac::utils::Settings proposal_engine_settings;

		size_t minimum_number_of_inliers,
			max_proposal_number_without_change,
			cell_number_in_neighborhood_graph,
			maximum_model_number;

		double maximum_tanimoto_similarity,
			confidence, // Required confidence in the result
			one_minus_confidence, // 1 - confidence
			inlier_outlier_threshold, // The inlier-outlier threshold
			spatial_coherence_weight; // The weight of the spatial coherence term

		void setConfidence(const double& confidence_)
		{
			confidence = confidence_;
			one_minus_confidence = 1.0 - confidence;
		}

		MultiModelSettings() :
			maximum_tanimoto_similarity(0.5),
			minimum_number_of_inliers(20),
			cell_number_in_neighborhood_graph(8),
			max_proposal_number_without_change(10),
			spatial_coherence_weight(0.14),
			inlier_outlier_threshold(2.0),
			confidence(0.95),
			one_minus_confidence(0.05),
			maximum_model_number(std::numeric_limits<size_t>::max())
		{
			proposal_engine_settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph;
			proposal_engine_settings.max_iteration_number = 5000;
			proposal_engine_settings.max_local_optimization_number = 50;
			proposal_engine_settings.threshold = inlier_outlier_threshold;
			proposal_engine_settings.confidence = confidence;
			proposal_engine_settings.spatial_coherence_weight = 0.975;
		}
	};

	struct IterationStatistics
	{
		double time_of_proposal_engine,
			time_of_model_validation,
			time_of_optimization,
			time_of_compound_model_update;
		size_t number_of_instances;
	};

	struct MultiModelStatistics
	{
		double processing_time,
			total_time_of_proposal_engine,
			total_time_of_model_validation,
			total_time_of_optimization,
			total_time_of_compound_model_calculation;
		std::vector<std::vector<size_t>> inliers_of_each_model;
		std::vector<size_t> labeling;
		std::vector<IterationStatistics> iteration_statistics;

		void addIterationStatistics(IterationStatistics iteration_statistics_)
		{
			iteration_statistics.emplace_back(iteration_statistics_);

			total_time_of_proposal_engine += iteration_statistics_.time_of_proposal_engine;
			total_time_of_model_validation += iteration_statistics_.time_of_model_validation;
			total_time_of_optimization += iteration_statistics_.time_of_optimization;
			total_time_of_compound_model_calculation += iteration_statistics_.time_of_compound_model_update;
		}
	};

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
		class ProgressiveX
	{


	};
}