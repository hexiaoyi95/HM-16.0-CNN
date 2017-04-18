
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


class CNN {
public:
  CNN(const string& model_file,
      const string& trained_file
     );

   float* ProcessImg(const cv::Mat& img , bool uv, int num_poc);
    float* ProcessImg2Blobs(const cv::Mat &img, const cv::Mat &images, bool uv, int num_poc);
private:
  //void SetMean(const string& mean_file);

  // std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels, int num_blob);
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};
 


































