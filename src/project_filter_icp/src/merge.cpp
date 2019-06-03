#include <ros/ros.h>  
#include "std_msgs/String.h"
#include <boost/thread.hpp>
#include <mutex>
#include <sstream>
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sensor_msgs/PointCloud2.h>

#include <detector_msg/bounding_box.h>
#include <detector_msg/detection_result.h>



#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
    using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;//using可以在派生类中使用基类的protected成员
public:
    MyPointRepresentation ()
    {
        nr_dimensions_ = 4;
    }

    virtual void copyToFloatArray (const PointNormalT &p, float * out) const//const修饰类的成员函数，则该成员函数不能修改类中任何非const成员函数
    {
        // < x, y, z, curvature >
        out[0] = p.x;
        out[1] = p.y;
        out[2] = p.z;
        out[3] = p.curvature;//这个函数是干嘛用的？
    }
};


  











class SubscribeAndPublish  
{  
public:  
    SubscribeAndPublish()  
    {  
        count = 0;

        //Topic you want to publish  
        pthread_rwlock_init(&right_lock, NULL);   //初始化读写锁
        pthread_rwlock_init(&left_lock, NULL);   //初始化读写锁

        _detection_sub = new message_filters::Subscriber<detector_msg::detection_result>(nh, "/detection_result", 1);
        _left_cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/sensor/lidar16/left/pointcloud", 1);
        _right_cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/sensor/lidar16/right/pointcloud", 1);


        _sync = new message_filters::Synchronizer<SyncPolicy>(
        SyncPolicy(10), *_detection_sub, *_left_cloud_sub, *_right_cloud_sub);
        _sync->registerCallback(boost::bind(&SubscribeAndPublish::combineCallback, this, _1, _2, _3));


        Eigen::Vector3f rvec(-0.0251048, 0.064995, -0.513204);// No.6  -0.0459966, -0.128626, 0.0558593
        Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();
        double rot_angle = rvec.norm();
	      if (std::fabs(rot_angle) > 1e-6) {
	      	Eigen::Vector3f axis = rvec.normalized();
          rot = Eigen::AngleAxis<float>(rot_angle, axis);
	      }
          
        Eigen::Vector3f tvec(-0.63735, -1.14282, -0.0275067);//No.6  -0.066045, -1.23302, 0.0377227
        _pairTransform.block<3,3>(0,0) = rot;
        _pairTransform.block<3,1>(0,3) = tvec;
    }  


    void combineCallback(const detector_msg::detection_resultConstPtr &dt_result,
                        const sensor_msgs::PointCloud2ConstPtr &left, 
                        const sensor_msgs::PointCloud2ConstPtr &right){ 
        dt_result->boxs[0].l_x;
        pcl::fromROSMsg (*right, *cloud_right);
        pcl::fromROSMsg (*left, *cloud_left);
        cloud_raw_merge->clear();
        *cloud_raw_merge += *cloud_left;
    

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());    
        pcl::transformPointCloud (*cloud_right, *transformed_cloud, _pairTransform);
        *cloud_raw_merge += *transformed_cloud;
    
        const auto project_t1 = std::chrono::high_resolution_clock::now();
        project_pc_image(image_origin,cloud_raw_merge,dt_result);
        const auto project_t2 = std::chrono::high_resolution_clock::now();
        ROS_INFO("Project a cloud cost %f ms",(project_t2-project_t1).count()*1e-6);

        const auto t1 = std::chrono::high_resolution_clock::now();
        if(cloud_former->size()>0){
            pairAlign(cloud_raw_merge, cloud_former, cloud_all, _sequence_transform, false);
            // std::cout<<_sequence_transform<<std::endl;
        }
        const auto t2 = std::chrono::high_resolution_clock::now();
        ROS_INFO("Total ICP cost %f ms",(t2-t1).count()*1e-6);

        cloud_former->clear();
        pcl::copyPointCloud(*cloud_raw_merge,*cloud_former);


        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(*cloud_all, ros_cloud);//topic pulicted
        ros_cloud.header = left->header;
        ros_cloud.header.frame_id = "camera_init";
        _pub.publish(ros_cloud);
    }



    bool in_bounding_box(cv::Point2f &pt,const float &l_x,const float &l_y,const float &r_x,const float &r_y){
        if( pt.x < l_x || pt.y < l_y || pt.x > r_x || pt.y > r_y){
            return false;
            }
        return true;
    }
    
    void project_pc_image(cv::Mat & img,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                            const detector_msg::detection_resultConstPtr &dt_result){
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_withoutNAN(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud_withoutNAN, indices);
        std::vector<cv::Point3f> pts_3d;
        for (size_t i = 0; i < cloud_withoutNAN->size(); ++i)
        {
            pcl::PointXYZ point_3d = cloud_withoutNAN->points[i];
            if (!(point_3d.x < 0 && point_3d.x > -3 && abs(point_3d.y) < 1.3)
                && point_3d.x < 50 && point_3d.x > -50 && point_3d.y < 50 && point_3d.y > -50)//截取周围环境点云
            {
                pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
            }
        }
        cloud->clear();

        std::vector<cv::Point2f> pts_2d;
        cv::projectPoints(pts_3d, _leftlidar_to_camera_rvec, _leftlidar_to_camera_tvec, _camera_intrinsic_coeff, _camera_distortion_coeff, pts_2d);
        cv::Mat img_show = cv::Mat::zeros(img.size(),CV_8UC3);
        int image_rows = image_origin.rows;
        int image_cols = image_origin.cols;
        int count=0;
        bool outside;
        for (size_t i = 0; i < pts_2d.size(); ++i)
        {
            outside = true;
            cv::Point2f point_2d = pts_2d[i];

            for(int j=0; j < dt_result->boxs.size(); ++j){
                if(in_bounding_box(point_2d,dt_result->boxs[j].l_x,dt_result->boxs[j].l_y,
                                    dt_result->boxs[j].r_x,dt_result->boxs[j].r_y))
                    {
                        outside = false;
                        cv::circle(img_show, point_2d, 5, cv::Scalar(255, 0, 0), -1);

                        continue;
                    }}
            if(outside){
                cloud->push_back(pcl::PointXYZ(pts_3d[i].x,pts_3d[i].y,pts_3d[i].z));
            }
        }
        // cv::namedWindow("img_projection",0);
        // cv::imshow("img_projection", img_show);
        // cv::waitKey(3); 
    }


    void pairAlign (const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_filter(new pcl::PointCloud<pcl::PointXYZ>),cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
        const auto cp_t1 = std::chrono::high_resolution_clock::now();

        *cloud_to_filter = *cloud_src;
        
        const auto cp_t2 = std::chrono::high_resolution_clock::now();
        ROS_INFO("Copy a cloud frame cost %f ms",(cp_t2-cp_t1).count()*1e-6);
        
        //半径滤波器
        const auto rd_filter_t1 = std::chrono::high_resolution_clock::now();
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> rd_filter;
	    // build the filter
	    rd_filter.setInputCloud(cloud_to_filter);
	    rd_filter.setRadiusSearch(0.1);
	    rd_filter.setMinNeighborsInRadius(3);
	    // apply filter
	    rd_filter.filter(*cloud_tmp);
        const auto rd_filter_t2 = std::chrono::high_resolution_clock::now();
        ROS_INFO("Radius filter a cloud cost %f ms",(rd_filter_t2-rd_filter_t1).count()*1e-6);

        //体素滤波器
        const auto filter_t1 = std::chrono::high_resolution_clock::now();
        pcl::VoxelGrid<pcl::PointXYZ> sor;//滤波处理对象
        sor.setInputCloud(cloud_tmp);
        sor.setLeafSize(0.1f, 0.1f, 0.2f);//设置滤波器处理时采用的体素大小的参数
        sor.filter(*cloud_to_filter);
        const auto filter_t2 = std::chrono::high_resolution_clock::now();
        ROS_INFO("Voxel filter a cloud cost %f ms",(filter_t2-filter_t1).count()*1e-6);

        const auto icp_t1 = std::chrono::high_resolution_clock::now();
        pcl::IterativeClosestPoint<PointT, PointT> icp;
	    icp.setMaximumIterations(50);
	    icp.setInputSource(cloud_to_filter);
	    icp.setInputTarget(cloud_tgt);
	    icp.align(*cloud_to_filter);   
    
	    if (icp.hasConverged()){	    	
	    	final_transform = icp.getFinalTransformation().cast<float>();	
	    }
	    else{
	    	ROS_ERROR("\nICP has not converged.\n");
	    }
        const auto icp_t2 = std::chrono::high_resolution_clock::now();
        ROS_INFO("Pair a cloud cost %f ms",(icp_t2-icp_t1).count()*1e-6);

        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud (*cloud_to_filter, *tmp, final_transform);
        output->clear();
        *output += *cloud_former;
        *output += *tmp;    
    }

private:  
    ros::NodeHandle nh;   
    ros::Publisher _pub = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 1);;
    message_filters::Subscriber<detector_msg::detection_result> *_detection_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2>  *_left_cloud_sub, *_right_cloud_sub;
    
    // 消息同步
    typedef message_filters::sync_policies::ApproximateTime<detector_msg::detection_result, 
                sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> *_sync;
    

    std_msgs::String output;
    pthread_rwlock_t right_lock;
    pthread_rwlock_t left_lock;
    int count;


    typedef boost::shared_lock<boost::shared_mutex> read_lock;
    typedef boost::unique_lock<boost::shared_mutex> write_lock;


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all {new pcl::PointCloud<pcl::PointXYZ>};//所有的cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left {new pcl::PointCloud<pcl::PointXYZ>};//转换为pcl格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_right {new pcl::PointCloud<pcl::PointXYZ>};//每两个相加的结果
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw_merge {new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_former {new pcl::PointCloud<pcl::PointXYZ>};//加工后的pcl格式
    sensor_msgs::PointCloud2 cloud_out;//转换为ros格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_last{new pcl::PointCloud<pcl::PointXYZ>};
    
    //right lidar to left lidar
    Eigen::Matrix4f _pairTransform = Eigen::Matrix4f::Identity (); //左右lidar 相对位姿

    Eigen::Matrix4f _sequence_transform = Eigen::Matrix4f::Identity ();

    //lidar_camera
    cv::Mat _camera_intrinsic_coeff = (cv::Mat_<double>(3,3,CV_64F) << 1673.764472, 0.0, 615.269162, 0.0, 1671.726940, 486.603777 , 0.0, 0.0, 1.0);
    

    cv::Mat _camera_distortion_coeff = (cv::Mat_<double>(1,4,CV_64F) << -0.093271, 0.295162, -0.002398, 0.000073); 
    cv::Mat _leftlidar_to_camera_tvec = (cv::Mat_<double>(3,1,CV_64F) << -0.7, -0.4, 0.3);
    cv::Mat _leftlidar_to_camera_rvec = (cv::Mat_<double>(3,1,CV_64F) << 0.861723, -1.48802, 1.63929);

    cv::Mat image_origin = cv::imread("/home/lhd/Documents/vaml-master/data/src.jpeg");
    // 
};//End of class SubscribeAndPublish  

int main(int argc, char **argv)  
{  
  //Initiate ROS  
  ros::init(argc, argv, "subscribe_and_publish");  

  //Create an object of class SubscribeAndPublish that will take care of everything  
  SubscribeAndPublish test;  
  ros::spin();

  return 0;  
}  

