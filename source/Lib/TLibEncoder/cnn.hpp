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

   float* ProcessImg(float* img);

private:
  //void SetMean(const string& mean_file);

  // std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};
 
CNN::CNN(const string& model_file,
         const string& trained_file){
  #ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  #else 
  Caffe::set_mode(Caffe::GPU);
  #endif

  /*Load the network*/
  net_.reset(new Net<float>(model_file,TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_geometry_=cv::Size(input_layer->width(),input_layer->height());
  num_channels_=input_layer->channels();
}

void CNN::WrapInputLayer(std::vector<cv::Mat>* input_channels){
   Blob<float>* input_layer = net_->input_blobs()[0];

   int width = input_layer->width();
   int height = input_layer->height();
   float* input_data = input_layer->mutable_cpu_data();
   for(int i = 0; i < input_layer->channels();++i){
    cv::Mat channel(height,width,CV_32FC1,input_data);
    input_channels->push_back(channel);
    input_data += width * height;
   }
}

 float* CNN::ProcessImg(float* img){
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1,num_channels_,input_geometry_.height,input_geometry_.width);

  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  cv::Mat img_float(input_geometry_.height,input_geometry_.width,CV_32FC1,img);
  cv::split(img_float,input_channels);
  //std::cout<<input_channels.size()<<std::endl;
  CHECK(reinterpret_cast<float*>((&input_channels)->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

  net_->Forward();

  Blob<float>* output_layer = net_->output_blobs()[0];
  const string string_size = output_layer->shape_string();
  std::cout<<string_size<<std::endl;
  float* begin = output_layer->mutable_cpu_data();
   
  return begin;
}


































