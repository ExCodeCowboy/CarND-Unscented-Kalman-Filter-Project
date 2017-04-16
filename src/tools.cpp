#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  unsigned long eSize = estimations.size();
  unsigned long gSize = ground_truth.size();
  if (eSize == 0 || gSize != eSize) {
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
  }

  //TODO linearize
  for (unsigned int i = 0; i < eSize; i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array().pow(2);
    rmse += residual;
  }

  rmse = rmse / eSize;
  rmse = rmse.array().sqrt();

  return rmse;
}
