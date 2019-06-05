#include <ros/ros.h>  
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <mutex>
#include <sstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>

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
#include <nav_msgs/Odometry.h>

#include <detector_msg/bounding_box.h>
#include <detector_msg/detection_result.h>

#include <ros_msgs/Odometry.h>
#include "algorithm/feature3d/pcl_utils.h"
#include "tergeo/common/eigen_utils.h"
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

class SubscribeAndPublish  
{  
public:  
    SubscribeAndPublish()  
    {  
        _count = 0;       

        _detection_sub = new message_filters::Subscriber<detector_msg::detection_result>(nh, "/detection_result", 1);
        _left_cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/sensor/lidar16/left/pointcloud", 1);
        _right_cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/sensor/lidar16/right/pointcloud", 1);
        _pose_sub = new message_filters::Subscriber<ros_msgs::Odometry>(nh, "/tergeo/localization/pose", 1);

        _sync = new message_filters::Synchronizer<SyncPolicy>(
        SyncPolicy(10), *_detection_sub, *_left_cloud_sub, *_right_cloud_sub, *_pose_sub);
        _sync->registerCallback(boost::bind(&SubscribeAndPublish::combineCallback, this, _1, _2, _3,_4)); 
        initLidarExtrinsics(); 
        initImutoLidar(); 
    }  

    void initLidarExtrinsics(){
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

    void initImutoLidar(){
        Eigen::Vector3f rvec(3.06471694496992, -0.828900259298348, 0.226311990377033);// No.6  -0.0459966, -0.128626, 0.0558593
        Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();
        double rot_angle = rvec.norm();
	      if (std::fabs(rot_angle) > 1e-6) {
	      	Eigen::Vector3f axis = rvec.normalized();
          rot = Eigen::AngleAxis<float>(rot_angle, axis);
	      }
          
        Eigen::Matrix4f tmp = Eigen::Matrix4f::Identity (); //左右lidar 相对位姿

        Eigen::Vector3f tvec(0.662062909899473, -0.612549212533238, 0);//No.6  -0.066045, -1.23302, 0.0377227
        tmp.block<3,3>(0,0) = rot;
        tmp.block<3,1>(0,3) = tvec;
        
        

        _imu_to_lidar=tmp.inverse();
    }

    void combineCallback(const detector_msg::detection_resultConstPtr &dt_result,
                        const sensor_msgs::PointCloud2ConstPtr &left, 
                        const sensor_msgs::PointCloud2ConstPtr &right,const ros_msgs::OdometryConstPtr &pose){ 
        
        dt_result->boxs[0].l_x;

        update_pose(pose);

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

        pcl::PointCloud<pcl::PointXYZ>::Ptr without_ground_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
        const auto rm_ground_t1 = std::chrono::high_resolution_clock::now();
        remove_ground(cloud_raw_merge,without_ground_cloud);
        const auto rm_ground_t2 = std::chrono::high_resolution_clock::now();
        ROS_INFO("remove ground cost %f ms",(rm_ground_t2-rm_ground_t1).count()*1e-6);

        const auto t1 = std::chrono::high_resolution_clock::now();
        if(cloud_former->size()>0){
            pairAlign(without_ground_cloud, cloud_former, cloud_pair, _sequence_transform, false);
            pcl::copyPointCloud(*without_ground_cloud,*cloud_former);
        }
        else{
            cloud_former->clear();
            pcl::copyPointCloud(*without_ground_cloud,*cloud_former);
        }

        const auto t2 = std::chrono::high_resolution_clock::now();
        ROS_INFO("Total ICP cost %f ms",(t2-t1).count()*1e-6);
        
        // pcl::io::savePLYFileBinary ("tmp.ply", *cloud_pair);
        


        sensor_msgs::PointCloud2 ros_cloud;
        ROS_INFO("Cloud all size is : %d",cloud_former->size());
        // ROS_INFO("cloud_former size is %d",cloud_former->size());

        pcl::toROSMsg(*cloud_all, ros_cloud);//topic pulished
        ros_cloud.header = left->header;
        ros_cloud.header.frame_id = "rslidar";
        _pub.publish(ros_cloud);
        pubish_odom(_former_to_init);
    }

    void remove_ground(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src,pcl::PointCloud<pcl::PointXYZ>::Ptr& withoutGround){
        Eigen::Matrix<double, 4, 1> plane_coeff;
        std::vector<int> inliers;
        tergeo::algorithm::PclUtils::extractPlane(cloud_src,2,plane_coeff,inliers);
        tergeo::algorithm::PclUtils::extractIndices(cloud_src,withoutGround,inliers,true);
    }

    void update_pose(const ros_msgs::OdometryConstPtr &pose){
        Eigen::Matrix4f tmp;
        tmp<< pose->R[0], pose->R[1], pose->R[2], 0,
                 pose->R[3], pose->R[4], pose->R[5], 0,
                 pose->R[6], pose->R[7], pose->R[8], 0,
                 0,0,0,1;        
        if(cloud_former->size()>0){
            tmp(0,3) = pose->x - _former_point[0];
            tmp(1,3) = pose->y - _former_point[1];
            tmp(2,3) = pose->z - _former_point[2];            
            _sequence_transform =  _former_status * tmp.inverse();
            std::cout<<"_sequence_transform     :"<<_sequence_transform<<endl;
        }
        _former_point[0] = pose->x;
        _former_point[1] = pose->y;
        _former_point[2] = pose->z;
        _former_status = tmp;
        _former_status(0,3) = 0;
        _former_status(1,3) = 0;
        _former_status(2,3) = 0; 
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
                && point_3d.x < 40 && point_3d.x > -40 && point_3d.y < 40 && point_3d.y > -40)//截取周围环境点云
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
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& output, Eigen::Matrix4f &final_transform, bool downsample = false)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_filter(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
        const auto cp_t1 = std::chrono::high_resolution_clock::now();

        pcl::transformPointCloud (*cloud_src, *cloud_to_filter, _sequence_transform);
        // pcl::copyPointCloud(*cloud_src, *cloud_to_filter);
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


        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setMaximumIterations(1000);
        icp.setMaxCorrespondenceDistance(0.1);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-2);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree<pcl::PointXYZ>());
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_dst(new pcl::search::KdTree<pcl::PointXYZ>());
        icp.setInputSource(cloud_to_filter);
        icp.setSearchMethodSource(tree_src);
        icp.setInputTarget(cloud_tgt);
        icp.setSearchMethodTarget(tree_dst);
        pcl::PointCloud<pcl::PointXYZ>::Ptr align(new pcl::PointCloud<pcl::PointXYZ>());
        icp.align(*align);
        final_transform = icp.getFinalTransformation(); 
        std::cout<<final_transform<<std::endl;

        // pcl::IterativeClosestPoint<PointT, PointT> icp;
	    // icp.setMaximumIterations(1000);
	    // icp.setInputSource(cloud_to_filter);
	    // icp.setInputTarget(cloud_tgt);
	    // icp.align(*cloud_to_filter);   

        // final_transform = _former_to_init * final_transform;
        // const auto icp_t2 = std::chrono::high_resolution_clock::now();
        // // ROS_INFO("Pair a cloud cost %f ms",(icp_t2-icp_t1).count()*1e-6);
        // _after_pair->clear();
        // pcl::transformPointCloud (*cloud_to_filter, *_after_pair, final_transform);
        // _former_to_init = final_transform;
        // output->clear();
        // *output += *cloud_former;
        // *output += *_after_pair;
        // *cloud_all += *_after_pair;
        final_transform = final_transform;// _sequence_transform
        pcl::transformPointCloud (*cloud_to_filter, *_after_pair, final_transform);


        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_from_init(new pcl::PointCloud<pcl::PointXYZ>);

        Eigen::Matrix4f init_to_now =_former_to_init *  (_sequence_transform * final_transform).inverse();
        _former_to_init = init_to_now;
        pcl::transformPointCloud (*cloud_src, *cloud_from_init, init_to_now);

        std::cout<<final_transform<<std::endl;

        // _after_pair->clear();
        output->clear();

        // pcl::copyPointCloud(*align,*_after_pair);
        *output += *_after_pair;
        *output += *cloud_tgt;
        *cloud_all += *cloud_from_init;


    }

void pubish_odom(const Eigen::Matrix4f status_matrix){
    Eigen::Matrix3f tmp = status_matrix.block(0,0,3,3);
    Eigen::Matrix<float, 3, 1> RPY = tergeo::common::eigen::RotationToRPY(tmp);
    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(RPY(0),RPY(1),RPY(2));
   _odomAftMapped.header.stamp = ros::Time::now();
   _odomAftMapped.header.frame_id = "rslidar";

   _odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
   _odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
   _odomAftMapped.pose.pose.orientation.z = geoQuat.x;
   _odomAftMapped.pose.pose.orientation.w = geoQuat.w;
   _odomAftMapped.pose.pose.position.x = status_matrix(0,3);
   _odomAftMapped.pose.pose.position.y = status_matrix(1,3);
   _odomAftMapped.pose.pose.position.z = status_matrix(2,3);
   _odomAftMapped.twist.twist.angular.x = RPY(0);
   _odomAftMapped.twist.twist.angular.y = RPY(1);
   _odomAftMapped.twist.twist.angular.z = RPY(2);
   _odomAftMapped.twist.twist.linear.x = status_matrix(0,3);;
   _odomAftMapped.twist.twist.linear.y = status_matrix(1,3);
   _odomAftMapped.twist.twist.linear.z = status_matrix(2,3);
   _pubOdomAftMapped.publish(_odomAftMapped);
}


private:  
    ros::NodeHandle nh;   
    ros::Publisher _pub = nh.advertise<sensor_msgs::PointCloud2>("/after_merge", 1);
    ros::Publisher _pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/laser_Odom", 1);

    message_filters::Subscriber<detector_msg::detection_result> *_detection_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2>  *_left_cloud_sub, *_right_cloud_sub;
    message_filters::Subscriber<ros_msgs::Odometry> *_pose_sub;

    // 消息同步
    typedef message_filters::sync_policies::ApproximateTime<detector_msg::detection_result, 
                sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, ros_msgs::Odometry> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> *_sync;
    nav_msgs::Odometry _odomAftMapped;

    int _count;


    typedef boost::shared_lock<boost::shared_mutex> read_lock;
    typedef boost::unique_lock<boost::shared_mutex> write_lock;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pair {new pcl::PointCloud<pcl::PointXYZ>};//前后两帧匹配的cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all {new pcl::PointCloud<pcl::PointXYZ>};//所有的cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left {new pcl::PointCloud<pcl::PointXYZ>};//转换为pcl格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_right {new pcl::PointCloud<pcl::PointXYZ>};//每两个相加的结果
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw_merge {new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_former {new pcl::PointCloud<pcl::PointXYZ>};//加工后的pcl格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr _after_pair{new pcl::PointCloud<pcl::PointXYZ>};

    sensor_msgs::PointCloud2 cloud_out;//转换为ros格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_last{new pcl::PointCloud<pcl::PointXYZ>};
    
    //right lidar to left lidar
    Eigen::Matrix4f _pairTransform = Eigen::Matrix4f::Identity (); //左右lidar 相对位姿
    Eigen::Matrix4f _imu_to_lidar = Eigen::Matrix4f::Identity (); //左右lidar 相对位姿

    Eigen::Matrix4f _former_to_init = Eigen::Matrix4f::Identity (); //左右lidar 相对位姿

    float _former_point[3];



    Eigen::Matrix4f _sequence_transform = Eigen::Matrix4f::Identity ();
    Eigen::Matrix4f _former_status = Eigen::Matrix4f::Identity();//the former pose from /tergeo/Odometry/pose
    //lidar_camera
    cv::Mat _camera_intrinsic_coeff = (cv::Mat_<double>(3,3,CV_64F) << 1673.764472, 0.0, 615.269162, 0.0, 1671.726940, 486.603777 , 0.0, 0.0, 1.0);
    

    cv::Mat _camera_distortion_coeff = (cv::Mat_<double>(1,4,CV_64F) << -0.093271, 0.295162, -0.002398, 0.000073); 
    cv::Mat _leftlidar_to_camera_tvec = (cv::Mat_<double>(3,1,CV_64F) << -0.7, -0.4, 0.3);
    cv::Mat _leftlidar_to_camera_rvec = (cv::Mat_<double>(3,1,CV_64F) << 0.861723, -1.48802, 1.63929);

    cv::Mat image_origin = cv::imread("/home/lhd/Documents/vaml-master/data/src.jpeg");

    int _frame;
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

