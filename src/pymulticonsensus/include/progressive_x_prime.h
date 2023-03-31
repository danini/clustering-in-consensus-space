#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp> 

#include "model.h"
#include "settings.h"
#include "statistics.h"
#include "modified_scoring_function.h"
//#include "connected_component_sampler.h"

namespace progx
{
	struct MultiModelSettings
	{
		size_t minimumInlierNumber,
			maximumModelNumber,
			startingHypothesisNumber,
			addedHypothesisNumber,
			maximumAttempts,
			maximumRANSACIterations,
			maximumIterations;

		double
			inlierPassingThreshold,
			modelDistanceThreshold,
			generationConfidence,
			confidence, // Required confidence in the result
			oneMinusConfidence, // 1 - confidence
			inlierOutlierThreshold; // The inlier-outlier threshold

		void setConfidence(const double& confidence_)
		{
			confidence = confidence_;
			oneMinusConfidence = 1.0 - confidence;
		}

		MultiModelSettings() :
			addedHypothesisNumber(10),
			inlierPassingThreshold(0.5),
			startingHypothesisNumber(10),
			maximumRANSACIterations(10000),
			maximumAttempts(100),
			modelDistanceThreshold(0.8),
			generationConfidence(0.99),
			maximumIterations(100),
			maximumModelNumber(std::numeric_limits<size_t>::max()),
			minimumInlierNumber(20),
			inlierOutlierThreshold(3.0),
			confidence(0.99)
		{
			oneMinusConfidence = 1.0 - confidence;
		}

		void print()
		{
			printf("Parameters:");
			printf("\n\tInlier-outlier threshold = %f", inlierOutlierThreshold);
			printf("\n\tModel-to-model distance threshold = %f", modelDistanceThreshold);
			printf("\n\tMaximum iteration number = %d", static_cast<int>(maximumIterations));
			printf("\n\tStarting number of hypotheses = %d", static_cast<int>(startingHypothesisNumber));
			printf("\n\tAdded number of hypotheses = %d", static_cast<int>(addedHypothesisNumber));
			printf("\n\tConfidence = %f", confidence);
			printf("\n\tMinimum inlier number = %d\n", static_cast<int>(minimumInlierNumber));
				
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

	struct ModelData
	{
		std::vector<double> consensusVector;
		std::vector<size_t> inliers;

		size_t size() const
		{
			return consensusVector.size();
		}

		size_t inlierNumber() const
		{
			return inliers.size();
		}
	};

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, // The clustering algorithm used in the consensus space
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler,
		class _ScoringFunction = progx::MSACScoringFunction<_ModelEstimator>> // The sampler used in the main RANSAC loop of GC-RANSAC
		class MultiConsensusFitting
	{
	public:
		cv::Mat image1, image2;

		MultiConsensusFitting() :
			scoringFunction(std::make_unique<_ScoringFunction>()),
			clusteringObject(std::make_unique<_Clustering>()),
			lossObject(std::make_unique<_RobustLoss>()),
			estimatorObject(std::make_unique<_ModelEstimator>())
		{

		}

		void run(
			const cv::Mat& points_,
			_Sampler& sampler_,
			std::vector<gcransac::Model>& models_,
			std::vector<ModelData>& modelData_) const;

		MultiModelSettings& getMutableSettings()
		{
			return settings;
		}

		_ScoringFunction& getMutableScoring()
		{
			return *scoringFunction;
		}

		const MultiModelSettings& getSettings() const
		{
			return settings;
		}

		_ModelEstimator& getMutableEstimator()
		{
			return *estimatorObject;
		}

		void iterativelyReweightedLSQ(
			const cv::Mat& points_,
			const _ModelEstimator& estimator_,
			gcransac::Model& model_,
			std::vector<size_t>& inliers_) const;

		void displayClustering(
			const cv::Mat& points_,
			const std::vector<ModelData>& hypothesisData_,
			const std::vector< std::vector<size_t> >& clusterIndices_) const;

	protected:
		MultiModelSettings settings;
		std::unique_ptr<_ModelEstimator> estimatorObject;
		const std::unique_ptr<_Clustering> clusteringObject;
		const std::unique_ptr<_RobustLoss> lossObject;
		const std::unique_ptr<_ScoringFunction> scoringFunction; // The scoring function used to measure the quality of a model

		template <size_t _SampleSize>
		bool sample(
			const std::vector<size_t>& pool_,
			_Sampler& sampler_,
			size_t* sample_) const;

		template <size_t _SampleSize>
		inline bool sample(
			_Sampler& sampler_,
			std::vector<size_t>& sample_) const;

		void extractInliersFromClusters(
			const cv::Mat& points_,
			const std::vector<ModelData>& hypothesisData_,
			const std::vector< std::vector<size_t> >& clusterIndices_,
			std::vector<std::pair<std::vector<size_t>, size_t>>& clusterInliers_) const;

		void refitAndReplace(
			std::vector<ModelData>& hypothesisData_,
			std::vector<gcransac::Model>& hypotheses_,
			const _ModelEstimator& estimator_,
			const cv::Mat& points_,
			const std::vector< std::vector<size_t> >& clusterIndices_,
			const std::vector<std::pair<std::vector<size_t>, size_t>>& clusterInliers_) const;

		bool generateHypotheses(
			const cv::Mat& points_,
			const size_t& nextHypothesesNumber_,
			std::vector<size_t>& pointPool_,
			const std::vector<double>& compoundModel_,
			_Sampler& sampler_,
			_ModelEstimator& estimator_,
			std::vector<ModelData>& hypothesisData_,
			std::vector<gcransac::Model>& hypothesesPool_,
			size_t& totalHypothesisNumber_) const;
	};

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, //
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _ScoringFunction>
		void MultiConsensusFitting<_Clustering, _ModelModelDistance, _RobustLoss, _ModelEstimator, _Sampler, _ScoringFunction>::run(
			const cv::Mat& points_,
			_Sampler& sampler_,
			std::vector<gcransac::Model>& models_,
			std::vector<ModelData>& modelData_) const
	{
		bool running = true;
		size_t nextHypothesesNumber = settings.startingHypothesisNumber;

		constexpr size_t kSampleNumber = _ModelEstimator::sampleSize();
		const size_t kPointNumber = points_.rows;

		std::vector<double> compoundModel(kPointNumber, 0);

		models_.reserve(10 * nextHypothesesNumber);
		modelData_.reserve(10 * nextHypothesesNumber);

		std::vector<size_t> pointPool(kPointNumber);
		std::iota(std::begin(pointPool), std::end(pointPool), 0);
		size_t coveredPoints,
			maxInlierNumber = kPointNumber,
			totalHypothesisNumber = 0;
		size_t iterationIdx = 0;

		// TODO: this has been replaced by by squared truncated
		scoringFunction->initialize(settings.inlierOutlierThreshold,
			kPointNumber); // Initializing the scoring function

		while (running)
		{
			++iterationIdx;

			if (iterationIdx >= settings.maximumRANSACIterations)
			{
				running = false;
				break;
			}

			// Add strong hypotheses to the pool
			const bool success = generateHypotheses(
				points_,
				nextHypothesesNumber,
				pointPool,
				compoundModel,
				sampler_,
				*estimatorObject,
				modelData_,
				models_,
				totalHypothesisNumber);
					
			if (!success)
				break;

			if (modelData_.size() < 1)
				continue;

			// Cluster the hypotheses in the consensus space
			std::vector< std::vector<size_t> > clusterIndices;
			size_t lastClusterSize = 0;
			bool changes = true;

			do
			{
				std::vector< std::vector<size_t> > tmpClusterIndices;

				clusteringObject->run(
					settings.modelDistanceThreshold,
					modelData_,
					tmpClusterIndices);

				// Select the inliers of the clusters
				std::vector<std::pair<std::vector<size_t>, size_t>> clusterInliers;
				extractInliersFromClusters(
					points_,
					modelData_,
					tmpClusterIndices,
					clusterInliers);

				// Refit the models to the found inliers and replace the cluster elements by the new model parameters
				refitAndReplace(
					modelData_,
					models_,
					*estimatorObject,
					points_,
					tmpClusterIndices,
					clusterInliers);

				if (lastClusterSize == tmpClusterIndices.size())
				{
					clusterIndices = tmpClusterIndices;
					break;
				}

				lastClusterSize = tmpClusterIndices.size();
			} while (changes);

			// Estimate the inlier number of an unseen model in the data
			std::fill(std::begin(compoundModel), std::end(compoundModel), 0.0);
			std::vector<bool> pointMask(kPointNumber, false);
			coveredPoints = 0;
			for (const auto& data : modelData_)
			{
				for (const auto& inlierIdx : data.inliers)
				{
					if (!pointMask[inlierIdx])
						++coveredPoints;
					pointMask[inlierIdx] = true;
					compoundModel[inlierIdx] =
						MAX(compoundModel[inlierIdx], data.consensusVector[inlierIdx]);
				}
			}

			const size_t tmpMaxInlierNumber = (kPointNumber - coveredPoints) *
				std::pow(1.0 - std::pow(1.0 - settings.confidence, 1.0 / totalHypothesisNumber), 1.0 / _ModelEstimator::sampleSize());
			maxInlierNumber = MIN(maxInlierNumber, tmpMaxInlierNumber);

			if (maxInlierNumber < settings.minimumInlierNumber)
				running = false;

			if (iterationIdx > settings.maximumIterations)
				running = false;

			// Increase the hypothesis number
			double theoreticalHypothesesNumber =
				std::log(1.0 - settings.generationConfidence) / std::log(1.0 - std::pow((double)coveredPoints / kPointNumber, _ModelEstimator::sampleSize()));

			nextHypothesesNumber = MIN(settings.addedHypothesisNumber, totalHypothesisNumber - theoreticalHypothesesNumber); 
		}

		for (size_t modelIdx = 0; modelIdx < models_.size(); ++modelIdx)
		{
			size_t inlierNums = 0;
			std::vector<gcransac::Model> refitModel;

			if (modelData_[modelIdx].inliers.size() < kSampleNumber)
				continue;

			// Estimate the model parameters using the current sample
			if (!estimatorObject->estimateModelNonminimal(
				points_,  // All points
				&(modelData_[modelIdx].inliers[0]), // The current sample
				modelData_[modelIdx].inliers.size(),
				&refitModel)) // The estimated model parameters
				continue;

			models_[modelIdx] = refitModel[0];
		}
	}

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, //
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _ScoringFunction>
		bool MultiConsensusFitting<_Clustering, _ModelModelDistance, _RobustLoss, _ModelEstimator, _Sampler, _ScoringFunction>::generateHypotheses(
			const cv::Mat& points_,
			const size_t& nextHypothesesNumber_,
			std::vector<size_t>& pointPool_,
			const std::vector<double>& compoundModel_,
			_Sampler &sampler_,
			_ModelEstimator &estimator_,
			std::vector<ModelData>& hypothesisData_,
			std::vector<gcransac::Model>& hypothesesPool_,
			size_t& totalHypothesisNumber_) const 
	{
		constexpr size_t kSampleNumber = _ModelEstimator::sampleSize();
		const size_t kPointNumber = points_.rows;
		const static progx::Score emptyScore;
		std::vector<size_t> currentExtendedSample;
		std::unique_ptr<size_t[]> currentSample(new size_t[kSampleNumber]); // Minimal sample for model fitting
		int currentHypothesisNumber;
		std::vector<size_t> tmpInliers;
		tmpInliers.reserve(kPointNumber);
		std::vector<gcransac::Model> tmpHypotheses;
		tmpHypotheses.reserve(_ModelEstimator::maximumMinimalSolutions());
		size_t attempts = 0;

		for (size_t hypothesesIdx = 0; hypothesesIdx < nextHypothesesNumber_; ++hypothesesIdx)
		{
			if (attempts > settings.maximumAttempts)
				return false;

			// Increase the total hypothesis number
			++totalHypothesisNumber_;

			// If the sampling is not successful, try again.
			/*if constexpr (std::is_same<gcransac::sampler::ConnectedComponentSampler, _Sampler>())
			{
				if (!sample<kSampleNumber>(
					sampler_,
					currentExtendedSample)) // The current sample
				{
					++attempts;
					--hypothesesIdx;
					continue;
				}
			}
			else*/
			{
				if (!sample<kSampleNumber>(
					pointPool_,
					sampler_, // The current pool from which the points are chosen
					currentSample.get())) // The current sample
				{
					++attempts;
					--hypothesesIdx;
					continue;
				}
			}

			// Check if the selected sample is valid before estimating the model
			// parameters which usually takes more time. 
			/*if constexpr (!std::is_same<gcransac::sampler::ConnectedComponentSampler, _Sampler>())
				if (!estimator_.isValidSample(points_, // All points
					currentSample.get())) // The current sample
				{
					++attempts;
					--hypothesesIdx;
					continue;
				}*/

			currentHypothesisNumber = hypothesesPool_.size();
			tmpHypotheses.clear();

			// Estimate the model parameters using the current sample
			/*if constexpr (std::is_same<gcransac::sampler::ConnectedComponentSampler, _Sampler>())
			{
				if (currentExtendedSample.size() == kSampleNumber)
				{
					if (!estimator_.estimateModel(points_,  // All points
						&(currentExtendedSample[0]), // The current sample
						&tmpHypotheses)) // The estimated model parameters
					{
						++attempts;
						--hypothesesIdx;
						continue;
					}
				}
				else
				{
					if (!estimator_.estimateModelNonminimal(points_,  // All points
						&(currentExtendedSample[0]), // The current sample
						currentExtendedSample.size(), // The current sample
						&tmpHypotheses)) // The estimated model parameters
					{
						++attempts;
						--hypothesesIdx;
						continue;
					}
				}
			}
			else*/
			{
				if (!estimator_.estimateModel(points_,  // All points
					currentSample.get(), // The current sample
					&tmpHypotheses)) // The estimated model parameters
				{
					++attempts;
					--hypothesesIdx;
					continue;
				}
			}

			//for (int innerHypothesisIdx = hypothesesPool_.size() - 1; innerHypothesisIdx >= currentHypothesisNumber; --innerHypothesisIdx)
			for (int innerHypothesisIdx = 0; innerHypothesisIdx < tmpHypotheses.size(); ++innerHypothesisIdx)
			{
				auto& model = tmpHypotheses[innerHypothesisIdx];

				// Calculate the consensus set of the current hypothesis
				progx::Score score = scoringFunction->getScore(
					points_, // All points
					model, // The current model parameters
					estimator_, // The estimator 
					settings.inlierOutlierThreshold, // The current threshold
					tmpInliers, // The current inlier set
					emptyScore, // The score of the current so-far-the-best model
					true); // Flag to decide if the inliers are needed

				size_t unique_inliers = score.inlier_number;
				for (const auto& inlierIdx : tmpInliers)
					if (compoundModel_[inlierIdx] > std::numeric_limits<double>::epsilon())
						--unique_inliers;

				if (unique_inliers < settings.minimumInlierNumber)
				{
					++attempts;
					continue;
				}

				hypothesesPool_.emplace_back(model);
				hypothesisData_.resize(hypothesisData_.size() + 1);
				auto& currentData = hypothesisData_.back();
				currentData.inliers.swap(tmpInliers);
				tmpInliers.clear();
				tmpInliers.reserve(kPointNumber);
				attempts = 0;

				// Initialize the consensus vector
				currentData.consensusVector.resize(kPointNumber, 0);
				for (const auto& inlierIdx : currentData.inliers)
				{
					const double residual = estimator_.residual(
						points_.row(inlierIdx),
						model);

					currentData.consensusVector[inlierIdx] = 
						lossObject->get(residual,
							settings.inlierOutlierThreshold);
				}
			}
		}

		return true;
	}

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, //
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _ScoringFunction>
	void MultiConsensusFitting<_Clustering, _ModelModelDistance, _RobustLoss, _ModelEstimator, _Sampler, _ScoringFunction>::refitAndReplace(
		std::vector<ModelData>& hypothesisData_,
		std::vector<gcransac::Model>& hypotheses_,
		const _ModelEstimator& estimator_,
		const cv::Mat& points_,
		const std::vector< std::vector<size_t> >& clusterIndices_,
		const std::vector<std::pair<std::vector<size_t>, size_t>>& clusterInliers_) const
	{
		const size_t& kPointNumber = points_.rows;
		const size_t& kClusterNumber = clusterIndices_.size();
		std::vector<bool> usedCluster(hypotheses_.size(), false);
		std::vector<ModelData> newHypothesisData;
		std::vector<gcransac::Model> newHypotheses;
		newHypotheses.reserve(kClusterNumber);
		newHypothesisData.reserve(kClusterNumber);

		// Iterate through the clusters
		for (size_t clusterIdx = 0; clusterIdx < kClusterNumber; ++clusterIdx)
		{
			const std::vector<size_t>& currentInliers = clusterInliers_[clusterIdx].first;

			if (currentInliers.size() < estimator_.nonMinimalSampleSize())
				continue;

			// Estimate the model parameters using the current sample
			if (!estimator_.estimateModelNonminimal(
				points_,  // All points
				&currentInliers[0], // The current sample
				currentInliers.size(),
				&newHypotheses)) // The estimated model parameters
			{
				newHypothesisData.emplace_back(hypothesisData_[clusterIdx]);
				continue;
			}

			newHypothesisData.resize(newHypothesisData.size() + 1);
			auto& currentData = newHypothesisData.back();

			// Apply iteratively re-weighted least-squares fitting to improve the model parameters and find their inliers
			iterativelyReweightedLSQ(
				points_, // All points
				estimator_, // The estimator 
				newHypotheses.back(), // The current model parameters
				currentData.inliers); // The current inlier set		

			// Initialize the consensus vector
			currentData.consensusVector.resize(kPointNumber, 0);
			for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
			{
				const double residual = estimator_.residual(
					points_.row(pointIdx),
					newHypotheses.back());

				currentData.consensusVector[pointIdx] =
					lossObject->get(residual,
						settings.inlierOutlierThreshold);
			}

			for (const auto& idx : clusterIndices_[clusterIdx])
				usedCluster[idx] = true;
		}

		hypothesisData_ = newHypothesisData;
		hypotheses_ = newHypotheses;
	}

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, //
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _ScoringFunction>
	void MultiConsensusFitting<_Clustering, _ModelModelDistance, _RobustLoss, _ModelEstimator, _Sampler, _ScoringFunction>::iterativelyReweightedLSQ(
		const cv::Mat& points_,
		const _ModelEstimator& estimator_,
		gcransac::Model& model_,
		std::vector<size_t>& inliers_) const
	{
		progx::Score bestScore;
		std::vector<size_t> tmpInliers;
		std::vector<gcransac::Model> tmpModel(1);
		std::vector<double> weights(points_.rows, 0.0);
		tmpModel[0] = model_;
		tmpInliers.reserve(points_.rows);

		while (1)
		{
			std::fill(std::begin(weights), std::end(weights), 0.0);

			progx::Score score = scoringFunction->getScore(
				points_, // All points
				tmpModel[0], // The current model parameters
				estimator_, // The estimator 
				settings.inlierOutlierThreshold, // The current threshold
				tmpInliers, // The current inlier set
				bestScore, // The score of the current so-far-the-best model
				true); // Flag to decide if the inliers are needed

			if (tmpInliers.size() != 0 && 
				bestScore < score)
			{
				tmpInliers.swap(inliers_);
				tmpInliers.clear();
				bestScore = score;
				model_ = tmpModel[0];
			}
			else
				break;

			// Calculate the weights
			weights.clear();
			lossObject->get(score.residuals,
				settings.inlierOutlierThreshold,
				weights);

			// Estimate the model parameters using the current sample
			tmpModel.clear();
			if (!estimator_.estimateModelNonminimal(
				points_,  // All points
				&inliers_[0], // The current sample
				inliers_.size(),
				&tmpModel, // The estimated model parameters
				&weights[0]))
				break;
		}
	}

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, //
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _ScoringFunction>
	void MultiConsensusFitting<_Clustering, _ModelModelDistance, _RobustLoss, _ModelEstimator, _Sampler, _ScoringFunction>::extractInliersFromClusters(
		const cv::Mat& points_,
		const std::vector<ModelData>& hypothesisData_,
		const std::vector< std::vector<size_t> >& clusterIndices_,
		std::vector<std::pair<std::vector<size_t>, size_t>> &clusterInliers_) const
	{
		const size_t& kPointNumber = points_.rows;
		const size_t& kClusterNumber = clusterIndices_.size();

		std::vector<size_t> tmpPoints(kPointNumber, 0);
		std::vector<bool> pointsAdded(kPointNumber, false);
		clusterInliers_.resize(kClusterNumber);

		// Iterate through the clusters
		for (size_t clusterIdx = 0; clusterIdx < kClusterNumber; ++clusterIdx)
		{
			const auto& cluster = clusterIndices_[clusterIdx];
			clusterInliers_[clusterIdx].first.reserve(kPointNumber);

			// An inlier is kept only if it appears this often
			// If settings.inlierPassingThreshold = 0.5, this is the median.
			const size_t minimalInlierFrequency =
				settings.inlierPassingThreshold * cluster.size();

			// Iterate through the hypotheses assigned to the current cluster
			int largestHypothesisIdx = -1;
			size_t largestHypothesisSize = 0;
			for (const auto& hypothesisIdx : cluster)
			{
				const size_t &hypothesisSize = hypothesisData_[hypothesisIdx].inliers.size();

				if (largestHypothesisSize < hypothesisSize)
				{
					largestHypothesisIdx = hypothesisIdx;
					largestHypothesisSize = hypothesisSize;
				}

				// Iterate through the inliers of the current hypothesis
				/*for (const auto& inlierIdx : hypothesisData_[hypothesisIdx].inliers)
				{
					if (++tmpPoints[inlierIdx] >= minimalInlierFrequency &&
						!pointsAdded[inlierIdx])
					{
						clusterInliers_[clusterIdx].emplace_back(inlierIdx);
						pointsAdded[inlierIdx] = true;
					}
				}*/
			}

			if (largestHypothesisIdx > -1)
			{
				clusterInliers_[clusterIdx].second = largestHypothesisIdx;
				for (const auto& inlierIdx : hypothesisData_[largestHypothesisIdx].inliers)
				{
					clusterInliers_[clusterIdx].first.emplace_back(inlierIdx);
					pointsAdded[inlierIdx] = true;
				}
			}

			// Reset the vector
			std::fill(std::begin(tmpPoints), std::end(tmpPoints), 0);
			std::fill(std::begin(pointsAdded), std::end(pointsAdded), 0);
		}

	}

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, //
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _ScoringFunction>
	void MultiConsensusFitting<_Clustering, _ModelModelDistance, _RobustLoss, _ModelEstimator, _Sampler, _ScoringFunction>::displayClustering(
		const cv::Mat& points_,
		const std::vector<ModelData>& hypothesisData_,
		const std::vector< std::vector<size_t> >& clusterIndices_) const
	{
		cv::Mat tmpImage1 = image1.clone();
		cv::Mat tmpImage2 = image2.clone();

		for (const auto& hypothesis : hypothesisData_)
		{
			cv::Scalar color(255.0 * (double)rand() / RAND_MAX, 
				255.0 * (double)rand() / RAND_MAX,
				255.0 * (double)rand() / RAND_MAX);

			/*for (const auto& inlierIdx : hypothesis.inliers)
			{
				cv::circle(tmpImage1,
					cv::Point(points_.at<double>(inlierIdx, 0), points_.at<double>(inlierIdx, 1)),
					3,
					color,
					-1);

				cv::circle(tmpImage2,
					cv::Point(points_.at<double>(inlierIdx, 2), points_.at<double>(inlierIdx, 3)),
					3,
					color,
					-1);
			}*/
		}

		/*for (const auto& cluster : clusterIndices_)
		{
			// An inlier is kept only if it appears this often
			// If settings.inlierPassingThreshold = 0.5, this is the median.
			const size_t minimalInlierFrequency =
				settings.inlierPassingThreshold * cluster.size();

			cv::Scalar color(255.0 * (double)rand() / RAND_MAX, 255.0 * (double)rand() / RAND_MAX, 255.0 * (double)rand() / RAND_MAX);
			std::vector<size_t> tmpPoints(points_.rows, 0);

			for (const auto& hypothesisIdx : cluster)
			{
				for (const auto& inlierIdx : hypothesisData_[hypothesisIdx].inliers)
				{
					if (++tmpPoints[inlierIdx] == minimalInlierFrequency)
					{
						cv::circle(tmpImage1,
							cv::Point(points_.at<double>(inlierIdx, 0), points_.at<double>(inlierIdx, 1)),
							3,
							color,
							-1);

						cv::circle(tmpImage2,
							cv::Point(points_.at<double>(inlierIdx, 2), points_.at<double>(inlierIdx, 3)),
							3,
							color,
							-1);
					}
				}
			}
		}*/
			
		cv::namedWindow("Image 1", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
		cv::namedWindow("Image 2", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
		cv::resizeWindow("Image 1", cv::Size(1024, 1024.0 / tmpImage1.cols * tmpImage1.rows));
		cv::resizeWindow("Image 2", cv::Size(1024, 1024.0 / tmpImage2.cols * tmpImage2.rows));
		cv::imshow("Image 1", tmpImage1);
		cv::imshow("Image 2", tmpImage2); 
	}

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, //
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _ScoringFunction>
		template <size_t _SampleSize>
	inline bool MultiConsensusFitting<_Clustering, _ModelModelDistance, _RobustLoss, _ModelEstimator, _Sampler, _ScoringFunction>::sample(
		const std::vector<size_t>& pool_,
		_Sampler& sampler_,
		size_t* sample_) const
	{
		return sampler_.sample(pool_, // The pool of indices
			sample_, // The selected sample
			_SampleSize); // The number of points to be selected
	}

	template<
		class _Clustering, // The clustering algorithm used in the consensus space
		class _ModelModelDistance, //
		class _RobustLoss,
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _Sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _ScoringFunction>
	template <size_t _SampleSize>
	inline bool MultiConsensusFitting<_Clustering, _ModelModelDistance, _RobustLoss, _ModelEstimator, _Sampler, _ScoringFunction>::sample(
		_Sampler& sampler_,
		std::vector<size_t> &sample_) const
	{
		return sampler_.sample(
			sample_, // The selected sample
			_SampleSize); // The number of points to be selected
	}
}

//#include "progressive_x_prime.cpp"