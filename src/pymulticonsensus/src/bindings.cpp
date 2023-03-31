#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "pymulticonsensus_python.h"

namespace py = pybind11;

py::tuple find6DPoses(
	py::array_t<double>  x1y1_,
	py::array_t<double>  x2y2z2_,
	py::array_t<double>  K_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number)
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=3");
	}
	if (NUM_TENTS < 3) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=3");
	}

	py::buffer_info buf1a = x2y2z2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 3) {
		throw std::invalid_argument("x2y2z2 should be an array with dims [n,3], n>=3");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2z2 should be the same size");
	}

	py::buffer_info buf1K = K_.request();
	size_t DIMK1 = buf1K.shape[0];
	size_t DIMK2 = buf1K.shape[1];

	if (DIMK1 != 3 || DIMK2 != 3) {
		throw std::invalid_argument("K should be an array with dims [3,3]");
	}

	double* ptr1 = (double*)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double* ptr1a = (double*)buf1a.ptr;
	std::vector<double> x2y2z2;
	x2y2z2.assign(ptr1a, ptr1a + buf1a.size);

	double* ptr1K = (double*)buf1K.ptr;
	std::vector<double> K;
	K.assign(ptr1K, ptr1K + buf1K.size);

	std::vector<double> poses;

	int num_models = 0;
	/*int num_models = find6DPoses_(
		x1y1,
		x2y2z2,
		K,
		labeling,
		poses,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number);*/

	py::array_t<double> poses_ = py::array_t<double>({ static_cast<size_t>(num_models) * 3, 4 });
	py::buffer_info buf2 = poses_.request();
	double* ptr2 = (double*)buf2.ptr;
	for (size_t i = 0; i < 12 * num_models; i++)
		ptr2[i] = poses[i];
	return poses_;
}

py::tuple findHomographies(
	py::array_t<double>  corrs_,
	size_t w1, size_t h1,
	size_t w2, size_t h2,
	double threshold,
	double confidence,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int starting_hypothesis_number,
	int added_hypothesis_number,
	int max_iters,
	int minimum_point_number,
	int sampler_id)
{
	py::buffer_info buf1 = corrs_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("corrs should be an array with dims [n,4], n>=4");
	}
	if (NUM_TENTS < 4) {
		throw std::invalid_argument("corrs should be an array with dims [n,4], n>=4");
	}

	double* ptr1 = (double*)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	std::vector<double> homographies;

	int num_models = findHomographies_(
		corrs,
		homographies,
		w1, h1,
		w2, h2,
		threshold,
		confidence,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		starting_hypothesis_number,
		added_hypothesis_number,
		max_iters,
		minimum_point_number,
		sampler_id);

	py::array_t<double> homographies_ = py::array_t<double>({ static_cast<size_t>(num_models) * 3, 3 });
	py::buffer_info buf2 = homographies_.request();
	double* ptr2 = (double*)buf2.ptr;
	for (size_t i = 0; i < 9 * num_models; i++)
		ptr2[i] = homographies[i];
	return homographies_;
}

py::tuple findTwoViewMotions(
	py::array_t<double>  corrs_,
	size_t w1, size_t h1,
	size_t w2, size_t h2,
	double threshold,
	double confidence,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int starting_hypothesis_number,
	int added_hypothesis_number,
	int max_iters,
	int minimum_point_number,
	int sampler_id)
{
	py::buffer_info buf1 = corrs_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("corrs should be an array with dims [n,4], n>=7");
	}
	if (NUM_TENTS < 7) {
		throw std::invalid_argument("corrs should be an array with dims [n,4], n>=7");
	}

	double* ptr1 = (double*)buf1.ptr;
	std::vector<double> corrs;
	corrs.assign(ptr1, ptr1 + buf1.size);

	std::vector<double> motions;

	int num_models = findTwoViewMotions_(
		corrs,
		motions,
		w1, h1,
		w2, h2,
		threshold,
		confidence,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		starting_hypothesis_number,
		added_hypothesis_number,
		max_iters,
		minimum_point_number,
		sampler_id);

	py::array_t<double> motions_ = py::array_t<double>({ static_cast<size_t>(num_models) * 3, 3 });
	py::buffer_info buf2 = motions_.request();
	double* ptr2 = (double*)buf2.ptr;
	for (size_t i = 0; i < 9 * num_models; i++)
		ptr2[i] = motions[i];
	return motions_;
}

py::tuple findPlanes(
	py::array_t<double>  points,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int sampler_id,
	double scoring_exponent,
	bool do_logging) 
{		
	py::buffer_info buf1 = points.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 3) {
		throw std::invalid_argument("points should be an array with dims [n,3], n>=3");
	}
	if (NUM_TENTS < 3) {
		throw std::invalid_argument("points should be an array with dims [n,3], n>=3");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> points_;
	points_.assign(ptr1, ptr1 + buf1.size);

	std::vector<double> planes;
	
	int num_models = 0;
	/*int num_models = findPlanes_(
		points_,
		labeling,
		planes,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number,
		sampler_id,
		scoring_exponent,
		do_logging);*/
	
	py::array_t<double> planes_ = py::array_t<double>({ static_cast<size_t>(num_models), 3 });
	py::buffer_info buf2 = planes_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 3 * num_models; i++)
		ptr2[i] = planes[i];
	return planes_;
}

py::tuple findVanishingPoints(
	py::array_t<double>  lines_,
	size_t w, 
	size_t h,
	double threshold,
	double confidence,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int starting_hypothesis_number,
	int added_hypothesis_number,
	int max_iters,
	int minimum_point_number,
	int sampler_id)
{
	py::buffer_info buf1 = lines_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("corrs should be an array with dims [n,3], n>=2");
	}
	if (NUM_TENTS < 2) {
		throw std::invalid_argument("corrs should be an array with dims [n,3], n>=2");
	}

	double* ptr1 = (double*)buf1.ptr;
	std::vector<double> lines;
	lines.assign(ptr1, ptr1 + buf1.size);

	std::vector<double> vanishing_points;

	int num_models = findVanishingPoints_(
		lines,
		vanishing_points,
		w, h,
		threshold,
		confidence,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		starting_hypothesis_number,
		added_hypothesis_number,
		max_iters,
		minimum_point_number,
		sampler_id);

	py::array_t<double> vanishing_points_ = py::array_t<double>({ static_cast<size_t>(num_models), 3 });
	py::buffer_info buf2 = vanishing_points_.request();
	double* ptr2 = (double*)buf2.ptr;
	for (size_t i = 0; i < 3 * num_models; i++)
		ptr2[i] = vanishing_points[i];
	return vanishing_points_;
}

py::array_t<double> getSoftLabeling(
	py::array_t<double>  models,
	py::array_t<double>  points,
	int model_type,
	double inlier_outlier_threshold) 
{
	py::buffer_info buf = points.request();
	size_t NUM_TENTS = buf.shape[0];
	size_t DIM = buf.shape[1];
	
	py::buffer_info model_buf = models.request();
	size_t NUM_MODELS = model_buf.shape[0];
	size_t DIM_MODELS = model_buf.shape[1];

	if (NUM_MODELS < 1) 
		throw std::invalid_argument("Models should be an array with dims [n,m], n>=1.");

	// Homography
	if (model_type == 0)
	{
		if (DIM != 4) 
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=4, for homography estimation.");
		if (NUM_TENTS < 4) 
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=4, for homography estimation.");
		if (DIM_MODELS != 9) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=9, where the homography elements are in a row-major order.");
	// Two-view motion
	} else if (model_type == 1)
	{
		if (DIM != 4) {
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=7, for two-view motion estimation.");
		}
		if (NUM_TENTS < 7) {
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=7, for two-view motion estimation.");
		}
		if (DIM_MODELS != 9) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=9, where the fundamental matrix elements are in a row-major order.");
	// Vanishing point estimation
	} else if (model_type == 2)
	{
		if (DIM != 4) {
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=2, for vanishing point estimation.");
		}
		if (NUM_TENTS < 2) {
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=2, for vanishing point estimation.");
		}
		if (DIM_MODELS != 3) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=3");
	// 6D rigid pose estimaton
	} else if (model_type == 3)
	{
		if (DIM != 5) {
			throw std::invalid_argument("Points should be an array with dims [n,5], n>=3, for 6D pose estimation from 2D-3D correspondences.");
		}
		if (NUM_TENTS < 3) {
			throw std::invalid_argument("Points should be an array with dims [n,5], n>=3, for 6D pose estimation from 2D-3D correspondences.");
		}
		if (DIM_MODELS != 12) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=12, where the pose (i.e., [R | t]) elements are in a row-major order.");
	// Plane estimation
	} else if (model_type == 4)
	{
		if (DIM != 3) {
			throw std::invalid_argument("Points should be an array with dims [n,3], n>=3, for 3D plane estimation.");
		}
		if (NUM_TENTS < 3) {
			throw std::invalid_argument("Points should be an array with dims [n,3], n>=3, for 3D plane estimation.");
		}
		if (DIM_MODELS != 4) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=4");
	}

	double *ptr = (double *)buf.ptr;
	std::vector<double> points_;
	points_.assign(ptr, ptr + buf.size);

	double *model_ptr = (double *)model_buf.ptr;
	std::vector<double> models_;
	models_.assign(model_ptr, model_ptr + model_buf.size);

	std::vector<double> labeling;

	getSoftLabeling_(
		points_,
		models_,
		model_type,
		inlier_outlier_threshold,
		labeling);
		
	py::array_t<double> labeling_ = py::array_t<double>({ NUM_TENTS, NUM_MODELS });
	py::buffer_info labeling_buf = labeling_.request();
	double *labeling_ptr = (double *)labeling_buf.ptr;
	for (size_t i = 0; i < NUM_TENTS * NUM_MODELS; i++)
		labeling_ptr[i] = labeling[i];

	return labeling_;
}

py::array_t<int> getLabeling(
	py::array_t<double>  models,
	py::array_t<double>  points,
	int model_type,
	double inlier_outlier_threshold,
	double neighborhood_size,
	double spatial_weight,
	double label_cost) 
{
	py::buffer_info buf = points.request();
	size_t NUM_TENTS = buf.shape[0];
	size_t DIM = buf.shape[1];
	
	py::buffer_info model_buf = models.request();
	size_t NUM_MODELS = model_buf.shape[0];
	size_t DIM_MODELS = model_buf.shape[1];

	if (NUM_MODELS < 1) 
		throw std::invalid_argument("Models should be an array with dims [n,m], n>=1.");

	// Homography
	if (model_type == 0)
	{
		if (DIM != 4) 
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=4, for homography estimation.");
		if (NUM_TENTS < 4) 
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=4, for homography estimation.");
		if (DIM_MODELS != 9) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=9, where the homography elements are in a row-major order.");
	// Two-view motion
	} else if (model_type == 1)
	{
		if (DIM != 4) {
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=7, for two-view motion estimation.");
		}
		if (NUM_TENTS < 7) {
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=7, for two-view motion estimation.");
		}
		if (DIM_MODELS != 9) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=9, where the fundamental matrix elements are in a row-major order.");
	// Vanishing point estimation
	} else if (model_type == 2)
	{
		if (DIM != 4) {
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=2, for vanishing point estimation.");
		}
		if (NUM_TENTS < 2) {
			throw std::invalid_argument("Points should be an array with dims [n,4], n>=2, for vanishing point estimation.");
		}
		if (DIM_MODELS != 3) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=3");
	// 6D rigid pose estimaton
	} else if (model_type == 3)
	{
		if (DIM != 5) {
			throw std::invalid_argument("Points should be an array with dims [n,5], n>=3, for 6D pose estimation from 2D-3D correspondences.");
		}
		if (NUM_TENTS < 3) {
			throw std::invalid_argument("Points should be an array with dims [n,5], n>=3, for 6D pose estimation from 2D-3D correspondences.");
		}
		if (DIM_MODELS != 12) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=12, where the pose (i.e., [R | t]) elements are in a row-major order.");
	// Plane estimation
	} else if (model_type == 4)
	{
		if (DIM != 3) {
			throw std::invalid_argument("Points should be an array with dims [n,3], n>=3, for 3D plane estimation.");
		}
		if (NUM_TENTS < 3) {
			throw std::invalid_argument("Points should be an array with dims [n,3], n>=3, for 3D plane estimation.");
		}
		if (DIM_MODELS != 4) 
			throw std::invalid_argument("Models should be an array with dims [n,m], n>=1, m=4");
	}

	double *ptr = (double *)buf.ptr;
	std::vector<double> points_;
	points_.assign(ptr, ptr + buf.size);

	double *model_ptr = (double *)model_buf.ptr;
	std::vector<double> models_;
	models_.assign(model_ptr, model_ptr + model_buf.size);

	std::vector<size_t> labeling;
	size_t max_label;

	getLabeling_(
		points_,
		models_,
		model_type,
		inlier_outlier_threshold,
		neighborhood_size,
		spatial_weight,
		label_cost,
		labeling,
		max_label);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info labeling_buf = labeling_.request();
	int *labeling_ptr = (int *)labeling_buf.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		// Replace the outlier label by -1
		if (labeling[i] == max_label)
			labeling_ptr[i] = -1;
		else
			labeling_ptr[i] = static_cast<int>(labeling[i]);

	return labeling_;
}

PYBIND11_PLUGIN(pymulticonsensus) {

	py::module m("pymulticonsensus", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pymulticonsensus
        .. autosummary::
           :toctree: _generate
           
           find6DPoses,
           findHomographies,
           findTwoViewMotions,
		   findPlanes,
		   findVanishingPoints,
		   getLabeling

    )doc");
	
	m.def("getLabeling", &getLabeling, R"doc(some doc)doc",
		py::arg("models"),
		py::arg("points"),
		py::arg("model_type"),
		py::arg("inlier_outlier_threshold") = 1.0,
		py::arg("neighborhood_size") = 20.0,
		py::arg("spatial_weight") = 0.0,
		py::arg("label_cost") = 0.1);
	
	m.def("getSoftLabeling", &getSoftLabeling, R"doc(some doc)doc",
		py::arg("models"),
		py::arg("points"),
		py::arg("model_type"),
		py::arg("inlier_outlier_threshold") = 1.0);

	m.def("findHomographies", &findHomographies, R"doc(some doc)doc",
		py::arg("corrs"),
		py::arg("w1"),
		py::arg("h1"),
		py::arg("w2"),
		py::arg("h2"),
		py::arg("threshold") = 3.0,
		py::arg("confidence") = 0.90,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("starting_hypothesis_number") = 20,
		py::arg("added_hypothesis_number") = 50,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 2 * 4,
		py::arg("sampler_id") = 0);

	m.def("findTwoViewMotions", &findTwoViewMotions, R"doc(some doc)doc",
		py::arg("corrs"),
		py::arg("w1"),
		py::arg("h1"),
		py::arg("w2"),
		py::arg("h2"),
		py::arg("threshold") = 3.0,
		py::arg("confidence") = 0.90,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("starting_hypothesis_number") = 20,
		py::arg("added_hypothesis_number") = 50,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 2 * 7,
		py::arg("sampler_id") = 0);

	m.def("find6DPoses", &find6DPoses, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2z2"),
		py::arg("K"),
		py::arg("threshold") = 4.0,
		py::arg("conf") = 0.90,
		py::arg("spatial_coherence_weight") = 0.1,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("max_iters") = 400,
		py::arg("minimum_point_number") = 2 * 3);

	m.def("findPlanes", &findPlanes, R"doc(some doc)doc",
		py::arg("points"),
		py::arg("threshold") = 4.0,
		py::arg("conf") = 0.5,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("neighborhood_ball_radius") = 200.0,
		py::arg("maximum_tanimoto_similarity") = 0.4,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 10,
		py::arg("sampler_id") = 3,
		py::arg("scoring_exponent") = 2,
		py::arg("do_logging") = false);

	m.def("findVanishingPoints", &findVanishingPoints, R"doc(some doc)doc",
		py::arg("lines"),
		py::arg("w"),
		py::arg("h"),
		py::arg("threshold") = 3.0,
		py::arg("confidence") = 0.90,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("starting_hypothesis_number") = 20,
		py::arg("added_hypothesis_number") = 50,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 2 * 7,
		py::arg("sampler_id") = 0);
	return m.ptr();
}