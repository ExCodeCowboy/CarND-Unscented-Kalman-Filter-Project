#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  n_x_ = 5;
  n_aug_ = 7;
  n_sig_aug_ = 2 * n_aug_ + 1;

  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(0.0);

  // initialize predicted x sigma
  Xsig_pred_ = MatrixXd(n_x_, n_sig_aug_);
  Xsig_pred_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 2, 0, 0, 0,0,
      0, 4, 0, 0,0,
      0, 0, 1, 0,0,
      0, 0, 0, 0.5,0,
      0, 0, 0, 0,0.5;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.0175;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.1;

  // Setup lambda
  lambda_aug_ = 3 - n_aug_;


  weights_ = VectorXd(n_sig_aug_);
  weights_.fill(0.0);
  //set weights
  double w1 = lambda_aug_ / (lambda_aug_ + n_aug_);
  double wn = 0.5 / (lambda_aug_ + n_aug_);
  weights_(0) = w1;
  for (int i = 1; i < n_sig_aug_; i++) {
    weights_(i) = wn;
  }
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

double AngleFix(const double& x) {
  double w = fmod(x, 2.*M_PI);
  while (w> M_PI) w-=2.*M_PI;
  while (w<-M_PI) w+=2.*M_PI;
  return w;
}

// define a custom template unary functor
struct AngleFixOp {
  AngleFixOp(){}
  const double operator()(const double& x) const {
    return AngleFix(x);
  }
};

double Calculate_NIS(VectorXd z, VectorXd z_pred, MatrixXd S, int angle_dimension) {
  VectorXd z_diff = z - z_pred;
  z_diff(angle_dimension) = AngleFix(z_diff(angle_dimension));
  return z_diff.transpose() * S.inverse() * z_diff;
}

double Calculate_lidar_NIS(VectorXd z, VectorXd z_pred, MatrixXd S) {
  VectorXd z_diff = z - z_pred;
  return z_diff.transpose() * S.inverse() * z_diff;
}


UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //Skip initialization 0 measurements.
  if (meas_package.raw_measurements_(0) == 0) {
    return;
  }

  if (is_initialized_ == false) {
    if(meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR &&
        use_radar_ == true) {
      VectorXd readings = meas_package.raw_measurements_;
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      float v = meas_package.raw_measurements_(2);
      float px = ro * cos(phi);
      float py = ro * sin(phi);
      //use observed velocity and observation angle to seed state.
      x_ << px, py, v, 0, 0;
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
    } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR &&
               use_laser_ == true){
      /**
       * Use the lidar positional data to initialize.
       */
      double px = meas_package.raw_measurements_(0);
      double py = meas_package.raw_measurements_(1);
      x_ << px, py, 0, 0, 0;
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
    }
    return;
  }

  if (time_us_ != meas_package.timestamp_) {
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    std::cout << "time D: " << meas_package.timestamp_ << " " << delta_t << " " << meas_package.sensor_type_
              << std::endl;
    while (delta_t > 1) {
      Prediction(0.5);
      delta_t -= 0.5;
    }
    Prediction(delta_t);
  }

  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR &&
      use_radar_ == true) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER &&
      use_laser_ == true) {
    UpdateLidar(meas_package);
  }

  if (isnanf(x_(0))) {
    throw;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  int n_aug = 7;
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);
  P_aug.fill(0.0);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, n_sig_aug_);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  //calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  //calculate the square root of lambda
  double root_lambda = sqrt(lambda_aug_ + n_aug);

  //build wide x matrix
  MatrixXd wx = x_aug.replicate(1,n_aug);
  MatrixXd sig_plus = wx + (root_lambda * A);
  MatrixXd sig_minus = wx - (root_lambda * A);

  Xsig_aug.block(0,1, n_aug, n_aug) = sig_plus;
  Xsig_aug.block(0,n_aug+1, n_aug, n_aug) = sig_minus;
  Xsig_aug.col(0) = x_aug;

  std::cout << "Xsig_aug: " << endl << Xsig_aug << endl;

  for (int i = 0; i < 2 * n_aug + 1; i++) {
    VectorXd point = Xsig_aug.col(i);

    //Current State
    double px = point(0);
    double py = point(1);
    double v = point(2);
    double yaw = point(3);
    double yaw_rate = point(4);
    double v_noise = point(5);
    double yaw_rate_noise = point(6);

    yaw_rate = min(yaw_rate, 20.0);
    yaw_rate = max(yaw_rate, -20.0);
    v = min(v, 90.0);
    v = max(v, -90.0);

    //New State
    double n_px, n_py, n_v, n_yaw, n_yaw_rate;
    if (yaw_rate != 0) {
      //Non-zero
      n_px = ((v/yaw_rate) * (sin(yaw + (yaw_rate * delta_t)) - sin(yaw)))
             + ((1/2.0)*(delta_t*delta_t)*cos(yaw)*v_noise);
      n_py = ((v/yaw_rate) * (-cos(yaw + (yaw_rate * delta_t)) + cos(yaw)))
             + ((1/2.0)*(delta_t*delta_t)*sin(yaw)*v_noise);
      n_v = delta_t * v_noise;
      n_yaw = (yaw_rate * delta_t) + ((1/2.0) * (delta_t*delta_t) * yaw_rate_noise);
      n_yaw_rate = delta_t * yaw_rate_noise;
    } else {
      //Zero
      n_px = (v * cos(yaw) * delta_t)
             + ((1/2.0)*(delta_t*delta_t)*cos(yaw)*v_noise);
      n_py = (v * sin(yaw) * delta_t)
             + ((1/2.0)*(delta_t*delta_t)*sin(yaw)*v_noise);
      n_v = delta_t * v_noise;
      n_yaw = ((1/2.0) * (delta_t*delta_t) * yaw_rate_noise);
      n_yaw_rate = delta_t * yaw_rate_noise;
    }

    Xsig_pred_.col(i) << px + n_px, py + n_py, v + n_v, yaw + n_yaw, yaw_rate + n_yaw_rate;
  }
  Xsig_pred_.row(3) = Xsig_pred_.row(3).unaryExpr(AngleFixOp());

  //predicted state mean
  x_ = Xsig_pred_ * weights_;
  x_(3) = AngleFix(x_(3));
  std::cout << "predicted x: " << endl << x_ << std::endl << std::endl;

  //predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
    //angle normalization
    //std::cout << "diff x: " << endl << x_diff << std::endl << std::endl;
    x_diff(3) = AngleFix(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  double px = meas_package.raw_measurements_(0);
  double py = meas_package.raw_measurements_(1);
  VectorXd z = VectorXd(n_z);
  z << px, py;
  std::cout << "measured: " << std::endl << z <<std::endl<<std::endl;
  //measurement covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;

  MatrixXd Zsig = Xsig_pred_.block(0,0,n_z,n_sig_aug_);

  //calculate mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  MatrixXd z_pred_diff = Zsig - z_pred.replicate(1, n_sig_aug_);
  MatrixXd w_z_pred_diff = z_pred_diff.array() * weights_.transpose().replicate(2,1).array();
  MatrixXd S = (w_z_pred_diff * z_pred_diff.transpose()) + R;

  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_aug_; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = AngleFix(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //Calculate NIS
  NIS_laser_ = Calculate_lidar_NIS(z, z_pred, S);

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  x_(3) = AngleFix(x_(3));
  P_ = P_ - K*S*K.transpose();

  std::cout << "calculated: x" << std::endl << x_ <<std::endl<<std::endl;
  std::cout << "calculated: P" << std::endl << P_ <<std::endl<<std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //measurement covariance matrix S
  int n_z = 3;

  MatrixXd Zsig = MatrixXd(n_z, n_sig_aug_);
  Zsig.fill(0.0);
  //transform sigma points into measurement space
  for (int i = 0; i<n_sig_aug_; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);
    //phi
    if (p_x == 0 && p_y == 0) {
      Zsig(2,i) = 0;
    } else{
      Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
    }
  }
  cout<< "Xsig_pred_: " << endl << Xsig_pred_ << endl <<endl;
  cout<< "Zsig: " << endl << Zsig << endl <<endl;

  //calculate mean predicted measurement
  VectorXd z_pred = Zsig * weights_;
  cout<<"pred_z: "<<endl<<z_pred<<endl<<endl;

  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R <<    std_radr_*std_radr_, 0, 0,
      0, std_radphi_*std_radphi_, 0,
      0, 0,std_radrd_*std_radrd_;

  //calculate measurement covariance matrix S
  MatrixXd z_pred_diff = Zsig - z_pred.replicate(1, n_sig_aug_);
  z_pred(1) = AngleFix(z_pred(1));
  MatrixXd w_z_pred_diff = z_pred_diff.array() * weights_.transpose().replicate(n_z,1).array();
  MatrixXd S = (w_z_pred_diff * z_pred_diff.transpose()) + R;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
//calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_aug_; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    z_diff(1) = AngleFix(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = AngleFix(x_diff(3));

    Tc = Tc + (weights_(i) * x_diff) * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z = meas_package.raw_measurements_;
  cout<<"z: "<<endl<<z<<endl<<endl;
  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = AngleFix(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  x_(3) = AngleFix(x_(3));
  P_ = P_ - K*S*K.transpose();
  std::cout << "calculated: x" << std::endl << x_ <<std::endl<<std::endl;
  std::cout << "calculated: P" << std::endl << P_ <<std::endl<<std::endl;

  NIS_radar_ = Calculate_NIS(z, z_pred, S, 1);
}
