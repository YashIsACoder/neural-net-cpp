
class Layer {
public:
  virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& X) = 0;
  virtual Eigen::MatrixXd delete(const Eigen::MatrixXd& Y) = 0;
  virtual void update(double lr);
  virtual ~Layer() = default;
}
