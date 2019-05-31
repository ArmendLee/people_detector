#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>


#include <chrono>
#include <iostream>
#include <memory>


#include "/home/lhd/Documents/vaml-master/caffe-detectors/include/caffedetector.h"
#include "/home/lhd/Documents/vaml-master/caffe-detectors/include/caffedetectorfactory.h"

#include <detector_msg/bounding_box.h>
#include <detector_msg/detection_result.h>
 
class ImageConverter
{
public:
    ImageConverter() : it_(nh_)
    {
        // Subscrive to input video feed and publish output video feed
        image_sub_ = it_.subscribe("/sensor/camera/front/image", 1,&ImageConverter::imageCb, this);
        
        image_pub_ = it_.advertise("/output_video", 1);
        detector_factory_ptr = cz::CaffeDetectorFactory::Instance();
        // create detector
        detector_ptr = detector_factory_ptr->createDetector(cz::CaffeDetectorFactory::DetectorMode::MOBILENET);
        std::string net_pt = "/home/lhd/Documents/vaml-master/models/mbssd_voc/mbssd_voc_deploy.prototxt";
        std::string net_weights = "/home/lhd/Documents/vaml-master/models/mbssd_voc/mbssd_voc.caffemodel";
        std::string classes_name = "/home/lhd/Documents/vaml-master/models/mbssd_voc/voc_labels.txt";
        detector_ptr->init(net_pt, net_weights, classes_name);
        detector_ptr->setComputeMode("cpu", 0);
    }
 
    ~ImageConverter() {}
 
void imageCb(const sensor_msgs::ImageConstPtr& msg){
        const auto t1 = std::chrono::high_resolution_clock::now();
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }    
        // Draw an example circle on the video stream
        // if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
        //     cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

        cv::Mat img = cv_ptr->image;

	    std::cout<<"read image victory"<<std::endl;
        
        std::vector<cz::Detection> dets;
        detector_ptr->detect(img, dets, 0.3);
        detector_msg::detection_result result;
        result.header = msg->header;
        // result.header.stamp = ros::Time::now();
        // result.header.frame_id = "map";

        detector_msg::bounding_box target;
        for(int i = 0; i < dets.size(); ++i){
            cv::Rect rc = dets[i].getRect();
            target.score = dets[i].getScore();
            target.l_x = rc.x;
            target.l_y = rc.y;
            target.r_x = rc.x + rc.width;
            target.r_y = rc.x + rc.height;
            result.boxs.push_back(target);
        }
        pub.publish(result);
        const auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Used time:" << (t2-t1).count()*1e-6 << "ms" << std::endl;
        detector_ptr->drawBox(img, dets);
        cv::namedWindow("img",0);
        cv::imshow("img", img);
        cv::waitKey(3);    
        // Output modified video stream
        image_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg());
    }
private:
    ros::NodeHandle nh_;
    ros::Publisher pub = nh_.advertise<detector_msg::detection_result>("/detection_result", 1);
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    cz::CaffeDetectorFactory::SharedPtr detector_factory_ptr;
    cz::CaffeDetector::SharedPtr detector_ptr;
    std::string net_pt;
    std::string net_weights;
    std::string classes_name;
};
 
int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}