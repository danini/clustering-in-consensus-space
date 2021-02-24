#include "openMVG.hpp"

#include <openMVG/sfm/pipelines/global/GlobalSfM_rotation_averaging.hpp>
#include <openMVG/sfm/pipelines/global/GlobalSfM_translation_averaging.hpp>
#include <openMVG/multiview/rotation_averaging.hpp>
#include <openMVG/multiview/translation_averaging_solver.hpp>

using namespace openMVG;

namespace pose_averaging {

    
  bool Translation_averaging(
    const std::vector<Sophus::SE3d>& poses_,
    const std::vector<bool>& vec_inliers,
    const Mat3& globalR,
    Vec3& C,
    openMVG::sfm::ETranslationAveragingMethod eTranslationAveragingMethod = openMVG::sfm::ETranslationAveragingMethod::TRANSLATION_AVERAGING_L2_DISTANCE_CHORDAL);

  bool poseAveraging_openMVG(
    const std::vector<Sophus::SE3d>& poses_,
    Sophus::SE3d& estimatedPose_,
    eTranslationAveragingMethod translationAvergaingMethod)
  {
    // Rotation averaging
    Eigen::Matrix3d rotationMatrix;
    std::vector<bool> vec_inliers;
    bool b_rotation_averaging = rotationAveraging_openMVG(poses_, rotationMatrix, vec_inliers);

    if (b_rotation_averaging) {
      // set rotation
      estimatedPose_.setRotationMatrix(rotationMatrix);

      switch (translationAvergaingMethod)
      {
      case pose_averaging::eTranslationAveragingMethod::Naive:
        // NOTE: this is just a really naive "translation averaging"
        return translationAveraging_naive(poses_, vec_inliers, estimatedPose_.translation());
        break;
      case pose_averaging::eTranslationAveragingMethod::OpenMVG_L2:
        return Translation_averaging(poses_, vec_inliers, rotationMatrix, estimatedPose_.translation());
        break;
      default:
        break;
      }

      return true; // SUCCESS
    }
    return false; // FAILURE
  }

  bool rotationAveraging_openMVG(const std::vector<Sophus::SE3d>& poses_, Eigen::Matrix3d& rotationMatrix, std::vector<bool>& vec_inliers)
  {
    using namespace openMVG;
    using namespace openMVG::sfm;
    using namespace openMVG::rotation_averaging;
    using namespace openMVG::rotation_averaging::l1;

    RelativeRotations vec_relatives_R;
    // TODO
    for (const auto& relative_pose : poses_) {
      // Add the relative rotation to the relative 'rotation' pose graph
      vec_relatives_R.emplace_back(
        0, 1,
        relative_pose.rotationMatrix(),
        1.f);
    }

    std::vector<Mat3> vec_globalR(2);


    //- Solve the global rotation estimation problem:
    const size_t nMainViewID = 0; //arbitrary choice

    bool b_rotation_averaging = rotation_averaging::l1::GlobalRotationsRobust(
      vec_relatives_R, vec_globalR, nMainViewID, 0.0f, &vec_inliers);

    std::cout << "\ninliers of L1 rotation averaging: " << std::endl;
    std::copy(vec_inliers.begin(), vec_inliers.end(), std::ostream_iterator<bool>(std::cout, " "));
    std::cout << std::endl;

    if (b_rotation_averaging) {
      rotationMatrix = vec_globalR.at(1);

      return true; // SUCCESS
    }
    return false; // FAILURE
  }

  bool translationAveraging_naive(const std::vector<Sophus::SE3d>& poses_, const std::vector<bool>& vec_inliers, Eigen::Vector3d& t)
  {
    int num_inliers = 0;
    t.setZero();
    for (size_t i = 0; i < vec_inliers.size(); ++i) {
      if (vec_inliers[i]) {
        ++num_inliers;
        t += poses_.at(i).translation().normalized();
      }
    }
    t /= num_inliers;
    return num_inliers > 0;
  }

  bool Translation_averaging(
    const std::vector<Sophus::SE3d>& poses_,
    const std::vector<bool>& vec_inliers,
    const Mat3& globalR,
    Vec3& C,
    openMVG::sfm::ETranslationAveragingMethod eTranslationAveragingMethod)
  {
    using namespace  openMVG::sfm;
    Pair_Set pairs = { {0,1} };


    const std::set<IndexT> index = { 0,1 };

    const size_t iNview = index.size();

    //-- Update initial estimates from [minId,maxId] to range [0->Ncam]}

    //openMVG::system::Timer timerLP_translation;

    switch (eTranslationAveragingMethod)
    {
    case TRANSLATION_AVERAGING_L1:
    {
      throw std::runtime_error("not implemented");
      //  double gamma = -1.0;
      //  std::vector<double> vec_solution;
      //  {
      //    vec_solution.resize(iNview * 3 + vec_relative_motion_cpy.size() + 1);
      //    using namespace openMVG::linearProgramming;
      //    OSI_CLP_SolverWrapper solverLP(vec_solution.size());

      //    lInfinityCV::Tifromtij_ConstraintBuilder cstBuilder(vec_relative_motion_cpy);

      //    LP_Constraints_Sparse constraint;
      //    //-- Setup constraint and solver
      //    cstBuilder.Build(constraint);
      //    solverLP.setup(constraint);
      //    //--
      //    // Solving
      //    const bool bFeasible = solverLP.solve();
      //    std::cout << " \n Feasibility " << bFeasible << std::endl;
      //    //--
      //    if (bFeasible) {
      //      solverLP.getSolution(vec_solution);
      //      gamma = vec_solution[vec_solution.size() - 1];
      //    }
      //    else {
      //      std::cerr << "Compute global translations: failed" << std::endl;
      //      return false;
      //    }
      //  }

      //  const double timeLP_translation = timerLP_translation.elapsed();
      //  //-- Export triplet statistics:
      //  {

      //    std::ostringstream os;
      //    os << "Translation fusion statistics.";
      //    os.str("");
      //    os << "-------------------------------" << "\n"
      //      << "-- #relative estimates: " << vec_relative_motion_cpy.size()
      //      << " converge with gamma: " << gamma << ".\n"
      //      << " timing (s): " << timeLP_translation << ".\n"
      //      << "-------------------------------" << "\n";
      //    std::cout << os.str() << std::endl;
      //  }

      //  std::cout << "Found solution:\n";
      //  std::copy(vec_solution.begin(), vec_solution.end(), std::ostream_iterator<double>(std::cout, " "));

      //  std::vector<double> vec_camTranslation(iNview * 3, 0);
      //  std::copy(&vec_solution[0], &vec_solution[iNview * 3], &vec_camTranslation[0]);

      //  std::vector<double> vec_camRelLambdas(&vec_solution[iNview * 3], &vec_solution[iNview * 3 + vec_relative_motion_cpy.size()]);
      //  std::cout << "\ncam position: " << std::endl;
      //  std::copy(vec_camTranslation.begin(), vec_camTranslation.end(), std::ostream_iterator<double>(std::cout, " "));
      //  std::cout << "\ncam Lambdas: " << std::endl;
      //  std::copy(vec_camRelLambdas.begin(), vec_camRelLambdas.end(), std::ostream_iterator<double>(std::cout, " "));
      //  std::cout << std::endl;

      //  // Update the view poses according the found camera centers
      //  for (size_t i = 0; i < iNview; ++i)
      //  {
      //    const Vec3 t(vec_camTranslation[i * 3], vec_camTranslation[i * 3 + 1], vec_camTranslation[i * 3 + 2]);
      //    const IndexT pose_id = reindex_backward[i];
      //    const Mat3& Ri = map_globalR.at(pose_id);
      //    sfm_data.poses[pose_id] = Pose3(Ri, -Ri.transpose() * t);
      //  }
      //}
      break;

    case TRANSLATION_AVERAGING_SOFTL1:
      throw std::runtime_error("not implemented");
      //{
      //  std::vector<Vec3> vec_translations;
      //  if (!solve_translations_problem_softl1(
      //    vec_relative_motion_cpy, vec_translations))
      //  {
      //    std::cerr << "Compute global translations: failed" << std::endl;
      //    return false;
      //  }

      //  // A valid solution was found:
      //  // - Update the view poses according the found camera translations
      //  for (size_t i = 0; i < iNview; ++i)
      //  {
      //    const Vec3& t = vec_translations[i];
      //    const IndexT pose_id = reindex_backward[i];
      //    const Mat3& Ri = map_globalR.at(pose_id);
      //    sfm_data.poses[pose_id] = Pose3(Ri, -Ri.transpose() * t);
      //  }
      //}
      break;

    case TRANSLATION_AVERAGING_L2_DISTANCE_CHORDAL:
    {
      std::vector<int> vec_edges;
      vec_edges.reserve(poses_.size() * 2);
      std::vector<double> vec_poses;
      vec_poses.reserve(2);
      std::vector<double> vec_weights;
      vec_weights.reserve(poses_.size());

      //for (const openMVG::RelativeInfo_Vec& iter : vec_relative_motion_cpy)
      for (int i = 0; i < poses_.size(); ++i)
      {
        //for (const relativeInfo& rel : iter)
        if (vec_inliers.at(i))
        {
          vec_edges.push_back(0);
          vec_edges.push_back(1);

          const Vec3 direction = poses_.at(i).translation().normalized();

          vec_poses.push_back(direction(0));
          vec_poses.push_back(direction(1));
          vec_poses.push_back(direction(2));

          vec_weights.push_back(1.0);
        }
      }

      const double function_tolerance = 1e-7, parameter_tolerance = 1e-8;
      const int max_iterations = 500;

      const double loss_width = 0.0; // No loss in order to compare with TRANSLATION_AVERAGING_L1

      std::vector<double> X(iNview * 3, 0.0);
      if (!solve_translations_problem_l2_chordal(
        &vec_edges[0],
        &vec_poses[0],
        &vec_weights[0],
        vec_edges.size() / 2,
        loss_width,
        &X[0],
        function_tolerance,
        parameter_tolerance,
        max_iterations)) {
        std::cerr << "Compute global translations: failed" << std::endl;
        return false;
      }

      // Update camera center for each view
      for (size_t i = 0; i < iNview; ++i)
      {
        const Vec3 C_(X[i * 3], X[i * 3 + 1], X[i * 3 + 2]);
        C = C_;
        //std::cout << "C: " << C.transpose() << std::endl;
      }
    }
    break;
    default:
    {
      std::cerr << "Unknown translation averaging method" << std::endl;
      return false;
    }
    }
    }
    return true;

  }

  bool translationAveraging_openMVG(const std::vector<Sophus::SE3d>& poses_, const std::vector<bool>& vec_inliers, Eigen::Vector3d& t)
  {
    // TODO: this is wild hacking
    return Translation_averaging(
      poses_,
      vec_inliers,
      Mat3::Identity(),
      t,
      openMVG::sfm::ETranslationAveragingMethod::TRANSLATION_AVERAGING_L2_DISTANCE_CHORDAL);
  }
  
}