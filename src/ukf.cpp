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
  use_laser_ = true;

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
  P_ << 1, 0, 0, 0,0,
      0, 1, 0, 0,0,
      0, 0, 1, 0,0,
      0, 0, 0, 0.5,0,
      0, 0, 0, 0,0.5;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ =1.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.2;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.2;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.002;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Setup lambda
  lambda_aug_ = 3 - n_aug_;

  // Setup weights for augmented sigma points
  weights_ = VectorXd(n_sig_aug_);
  weights_.fill(0.0);
  double w1 = lambda_aug_ / (lambda_aug_ + n_aug_);
  double wn = 0.5 / (lambda_aug_ + n_aug_);
  weights_(0) = w1;
  for (int i = 1; i < n_sig_aug_; i++) {
    weights_(i) = wn;
  }

  //Setup the
  laser_R_ = MatrixXd(2, 2);
  laser_R_ << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;
}

//Function to apply for angle normalization.
double AngleFix(const double& x) {
  double w = fmod(x, 2.*M_PI);
  while (w> M_PI) w-=2.*M_PI;
  while (w<-M_PI) w+=2.*M_PI;
  return w;
}

// define a custom template unary functor for applying to matrix rows.
struct AngleFixOp {
  AngleFixOp(){}
  const double operator()(const double& x) const {
    return AngleFix(x);
  }
};

//NIS Calculations
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

  //Initialize
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
    } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER &&
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

  //Update delta T
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  //Walk out the prediction in steps to improve stability
  while (delta_t > 0.05) {
    Prediction(0.05);
    delta_t -= 0.05;
  }
  Prediction(delta_t);

  //Process measurements
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR &&
      use_radar_ == true) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER &&
      use_laser_ == true) {
    UpdateLidar(meas_package);
  }

  //Exit early if calculations get unstable. I was running into issues earlier.
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
  int n_aug = 7;
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug);
  x_aug.fill(0.0);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);
  P_aug.fill(0.0);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, n_sig_aug_);
  Xsig_aug.fill(0);

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

  //Build Augmented Sigma points.
  Xsig_aug.block(0,1, n_aug, n_aug) = sig_plus;
  Xsig_aug.block(0,n_aug+1, n_aug, n_aug) = sig_minus;
  Xsig_aug.col(0) = x_aug;

  //Calculate predicted locations of sigma points.
  for (int i = 0; i < n_sig_aug_; i++) {
    VectorXd point = Xsig_aug.col(i);

    //Current State
    double px = point(0);
    double py = point(1);
    double v = point(2);
    double yaw = point(3);
    double yaw_rate = point(4);
    double v_noise = point(5);
    double yaw_rate_noise = point(6);

    //Put bounds on yaw and velocity rates.
    yaw_rate = min(yaw_rate, 3.0);
    yaw_rate = max(yaw_rate, -3.0);
    v = min(v, 75.0);
    v = max(v, -75.0);

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
  //Normalize angle.
  Xsig_pred_.row(3) = Xsig_pred_.row(3).unaryExpr(AngleFixOp());

  //predicted state mean
  x_ = Xsig_pred_ * weights_;
  x_(3) = AngleFix(x_(3));

  //build state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_aug_; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
    //angle normalization
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

  //pull in current measurement
  double px = meas_package.raw_measurements_(0);
  double py = meas_package.raw_measurements_(1);
  VectorXd z = VectorXd(n_z);
  z << px, py;

  //Slice out just the x,y positions
  MatrixXd Zsig = Xsig_pred_.block(0,0,n_z,n_sig_aug_);

  //calculate mean predicted position
  VectorXd z_pred = Zsig * weights_;

  //Calculated the difference between predicted mean and predicted positions
  MatrixXd z_pred_diff = Zsig - z_pred.replicate(1, n_sig_aug_);
  //Create a weighted copy
  MatrixXd w_z_pred_diff = z_pred_diff.array() * weights_.transpose().replicate(2,1).array();

  //calculate measurement covariance matrix S
  MatrixXd S = (w_z_pred_diff * z_pred_diff.transpose()) + laser_R_;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_aug_; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
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
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;

  //build Sigma points in Z space.
  MatrixXd Zsig = BuildZsig(Xsig_pred_, n_sig_aug_);

  //calculate mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  MatrixXd S = BuildRadarS(Zsig,
                           weights_,
                           z_pred,
                            n_sig_aug_,
                           std_radr_,
                           std_radphi_,
                           std_radrd_);


  //calculate cross correlation matrix
  MatrixXd Tc = BuildRadarTc(n_sig_aug_, Zsig, z_pred, n_x_, Xsig_pred_, x_, weights_);

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z = meas_package.raw_measurements_;

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = AngleFix(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  x_(3) = AngleFix(x_(3));
  P_ = P_ - K*S*K.transpose();

  NIS_radar_ = Calculate_NIS(z, z_pred, S, 1);
}

MatrixXd UKF::BuildRadarTc(int n_sig,
                           MatrixXd &Zsig,
                           VectorXd &z_pred,
                           int n_x,
                           MatrixXd &Xsig,
                           VectorXd &x,
                           VectorXd &weights) const {
  int n_z = 3;
  MatrixXd Tc = MatrixXd(n_x, n_z);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - Zsig.col(0);
    //angle normalization
    z_diff(1) = AngleFix(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig.col(i) - Xsig.col(0);
    //angle normalization
    x_diff(3) = AngleFix(x_diff(3));

    Tc = Tc + (weights(i) * x_diff) * z_diff.transpose();
  }
  return Tc;
}

MatrixXd UKF::BuildRadarS(MatrixXd &Zsig,
                          VectorXd &weights,
                          VectorXd &z_pred,
                          int n_sig,
                          double std_radr,
                          double std_radphi,
                          double std_radrd) const {
  int n_z = 3;
  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R << std_radr * std_radr, 0, 0,
      0, std_radphi * std_radphi, 0,
      0, 0, std_radrd * std_radrd;

  MatrixXd z_pred_diff = Zsig - z_pred.replicate(1, n_sig);
  z_pred_diff.row(1) = z_pred_diff.row(1).unaryExpr(AngleFixOp());
  MatrixXd w_z_pred_diff = z_pred_diff.array() * weights.transpose().replicate(n_z, 1).array();

  MatrixXd S = (w_z_pred_diff * z_pred_diff.transpose()) + R;
  return S;
}

MatrixXd UKF::BuildZsig(MatrixXd& Xsig_pred, int n_sig) const {
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, n_sig);
  Zsig.fill(0.0);
  //transform sigma points into measurement space
  for (int i = 0; i < n_sig; i++) {
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);
    double v  = Xsig_pred(2, i);
    double yaw = Xsig_pred(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = AngleFix(atan2(p_y,p_x));
    //phi
    if (p_x == 0 && p_y == 0) {
      Zsig(2,i) = 0;
    } else{
      Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
    }
  }
  return Zsig;
}
