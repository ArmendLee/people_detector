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
#include "algorithm/feature3d/cloud_pair_solver.h"

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

        // ROS_WARN("RPY between lidar is:");
        // std::cout<<tergeo::common::eigen::RotationToRPY(rot)<<std::endl;
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
                        const sensor_msgs::PointCloud2ConstPtr &right,const ros_msgs::OdometryConstPtr &pose)
                        { 
        update_pose(pose);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_show(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::fromROSMsg (*right, *cloud_right);
        pcl::fromROSMsg (*left, *cloud_left);

        std::vector<int> indices;
        cloud_raw_merge->clear();
        *cloud_raw_merge += *cloud_left;        

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());    
        pcl::transformPointCloud (*cloud_right, *transformed_cloud, _pairTransform);
        *cloud_raw_merge += *transformed_cloud;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_withoutNAN(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::removeNaNFromPointCloud(*cloud_raw_merge, *cloud_withoutNAN, indices);
        project_pc_image(image_origin,cloud_withoutNAN,dt_result);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_ground(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ground_rst(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected_ground(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_rectified(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Vector4d plane_coeff;
        extractGround(cloud_withoutNAN, cloud_without_ground, cloud_ground_rst, plane_coeff);
        Eigen::Matrix4d trans = calProjectPlaneMat(plane_coeff);
        pcl::transformPointCloud(*cloud_without_ground, *cloud_ground_rectified, trans);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_as_road(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::transformPointCloud(*cloud_without_ground, *cloud_to_show, trans);


        tergeo::algorithm::PclUtils::passThrough(cloud_ground_rectified, cloud_projected_ground, "z", -1, 1.5);
        tergeo::algorithm::PclUtils::passThrough(cloud_projected_ground, cloud_as_road, "x", -15, 10);


        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_pair(new pcl::PointCloud<pcl::PointXYZ>);
        filter(cloud_as_road,cloud_to_pair); 
        projectToGround(cloud_to_pair);

        const auto t1 = std::chrono::high_resolution_clock::now();
        if(cloud_former->size()>0){
            pairAlign(cloud_to_pair, cloud_former);
            display_pair_result(cloud_to_show,cloud_former_toshow);
            pcl::copyPointCloud(*cloud_to_pair,*cloud_former);
            pcl::copyPointCloud(*cloud_to_show,*cloud_former_toshow);

        }
        else{
            cloud_former->clear();
            pcl::copyPointCloud(*cloud_to_pair,*cloud_former);
            pcl::copyPointCloud(*cloud_to_show,*cloud_former_toshow);
        }   

        sensor_msgs::PointCloud2 ros_cloud;

        pcl::toROSMsg(*cloud_without_ground, ros_cloud);//topic pulished
        ros_cloud.header = left->header;
        ros_cloud.header.frame_id = "rslidar";
        _pub.publish(ros_cloud);
        pubish_odom(_former_to_init);
        publish_pc2(cloud_all);
    }


    void extractGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_without_ground,
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& ground_rst, Eigen::Vector4d& plane_coeff) {
        std::vector<int> inliers;
        tergeo::algorithm::PclUtils::extractPlane(cloud_in, 1.1, plane_coeff, inliers);
        tergeo::algorithm::PclUtils::extractIndices(cloud_in, cloud_without_ground, inliers, true);
        tergeo::algorithm::PclUtils::convertCloud(cloud_in, ground_rst, tergeo::algorithm::Color::White());
        for (int i = 0; i < inliers.size(); ++i) {
            ground_rst->points[inliers[i]].r = 255;
            ground_rst->points[inliers[i]].g = 0;
            ground_rst->points[inliers[i]].b = 0;
        }
    }

    Eigen::Matrix4d calProjectPlaneMat(const Eigen::Vector4d& plane_coeff) {
        Eigen::Vector3d t_vec(0,0,plane_coeff(3));
        Eigen::Vector3d z_vec = plane_coeff.head(3);
        Eigen::Vector3d y_vec(0,1,0);
        Eigen::Vector3d x_vec = y_vec.cross(z_vec);
        y_vec = z_vec.cross(x_vec);
        Eigen::Matrix4d trans_mat = Eigen::Matrix4d::Identity();
        trans_mat.block(0, 3, 3, 1) = t_vec;
        trans_mat.block(0, 0, 1, 3) = x_vec.transpose();
        trans_mat.block(1, 0, 1, 3) = y_vec.transpose();
        trans_mat.block(2, 0, 1, 3) = z_vec.transpose();

        Eigen::Matrix3d rot= Eigen::Matrix3d::Identity();;
        Eigen::Vector3d axis(0,0,1);

        rot = Eigen::AngleAxis<double>(2.2, axis) * trans_mat.block<3,3>(0,0);
        trans_mat.block<3,3>(0,0) = rot;
        return trans_mat;
    }

    void projectToGround(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        for (int i = 0; i < cloud->size(); ++i) {
            cloud->points[i].z = 0;
        }
    }

    void remove_ground(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src,pcl::PointCloud<pcl::PointXYZ>::Ptr& withoutGround){
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pointcloud (new pcl::PointCloud<pcl::PointXYZ> ());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_pointcloud_toshow (new pcl::PointCloud<pcl::PointXYZRGB> ());

        
        Eigen::Matrix<double, 4, 1> plane_coeff;
        std::vector<int> inliers;

        tergeo::algorithm::PclUtils::extractPlane(cloud_src,2,plane_coeff,inliers);
        tergeo::algorithm::PclUtils::extractIndices(cloud_src,withoutGround,inliers,true);
        // tergeo::algorithm::PclUtils::extractIndices(cloud_src,ground_pointcloud,inliers,false);

        // pcl::io::savePLYFile("/home/lhd/Documents/vaml-master/caffe-detectors/people_detector/without_ground.ply", *withoutGround);


        // int g_size = ground_pointcloud->points.size();
        // int not_ground_size = withoutGround->points.size();
        // for (int i = 0;i <g_size; ++i)
        // {
        //         pcl::PointXYZRGB p;
        //         p.x = ground_pointcloud->points[i].x;
        //         p.y = ground_pointcloud->points[i].y;
        //         p.z = ground_pointcloud->points[i].z;
        //         p.b = 255;
        //         p.g = 0;
        //         p.r = 0;
        //         p.a = 1;
        //         ground_pointcloud_toshow->points.push_back(p);
        // }
        // for (int i = 0;i <not_ground_size; ++i)
        // {
        //         pcl::PointXYZRGB p;
        //         p.x = withoutGround->points[i].x;
        //         p.y = withoutGround->points[i].y;
        //         p.z = withoutGround->points[i].z; 
        //         p.b = 0;
        //         p.g = 0;
        //         p.r = 255;
        //         p.a = 1;
        //         ROS_WARN("RGB field is %f.",p.rgb);

        //         ground_pointcloud_toshow->points.push_back(p);
        // }
        // ground_pointcloud_toshow->width = g_size + not_ground_size;
        // ground_pointcloud_toshow->height = 1;
        // // ground_pointcloud_toshow->points.resize()
        // pcl::io::savePLYFile("/home/lhd/Documents/vaml-master/caffe-detectors/people_detector/ground_tmp.ply", *ground_pointcloud_toshow);
        
        // sensor_msgs::PointCloud2 ground_cloud;
        // // ROS_INFO("Cloud all size is : %d",cloud_former->size());
        // // ROS_INFO("cloud_former size is %d",cloud_former->size());
        // // pcl::toPCLPointCloud2(*ground_pointcloud_toshow, ground_cloud);
        // pcl::toROSMsg(*ground_pointcloud_toshow, ground_cloud);//topic pulished
        // // ros_cloud.header = left->header;
        // ground_cloud.header.frame_id = "rslidar";
        // _pubTmpPoincCloud.publish(ground_cloud);
    }

    void publish_pc2(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src){
        sensor_msgs::PointCloud2 ground_cloud;
        pcl::toROSMsg(*cloud_src, ground_cloud);
        ground_cloud.header.frame_id = "rslidar";
        _pubTmpPoincCloud.publish(ground_cloud);
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
            // std::cout<<"_sequence_transform     :"<<_sequence_transform<<endl;
        }
        _former_point[0] = pose->x;
        _former_point[1] = pose->y;
        _former_point[2] = pose->z;
        _former_status = tmp;
        _former_status(0,3) = 0;
        _former_status(1,3) = 0;
        _former_status(2,3) = 0;
        // Eigen::Matrix4f tmp;
        // tmp<< pose->R[0], pose->R[1], pose->R[2], pose->x,
        //          pose->R[3], pose->R[4], pose->R[5], pose->y,
        //          pose->R[6], pose->R[7], pose->R[8], pose->z,
        //          0,0,0,1;        
        // if(cloud_former->size()>0){                      
        //     _sequence_transform =  _former_status * tmp.inverse();
        //     std::cout<<"_sequence_transform     :"<<_sequence_transform<<endl;
        // }
        // _former_status = tmp;
    }

    bool in_bounding_box(cv::Point2f &pt,const float &l_x,const float &l_y,const float &r_x,const float &r_y){
        if( pt.x < l_x || pt.y < l_y || pt.x > r_x || pt.y > r_y){
            return false;
            }
        return true;
    }
    
    void project_pc_image(cv::Mat & img,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                            const detector_msg::detection_resultConstPtr &dt_result){       
        std::vector<cv::Point3f> pts_3d;
        for (size_t i = 0; i < cloud->size(); ++i)
        {
            pcl::PointXYZ point_3d = cloud->points[i];
            if (!(point_3d.x < 0 && point_3d.x > -3 && abs(point_3d.y) < 1.3)
                && point_3d.x < 30 && point_3d.x > -30 && point_3d.y < 30 && point_3d.y > -30)//截取周围环境点云
            {
                pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
            }
        }
        cloud->clear();

        std::vector<cv::Point2f> pts_2d;
        cv::projectPoints(pts_3d, _leftlidar_to_camera_rvec, _leftlidar_to_camera_tvec, _camera_intrinsic_coeff, _camera_distortion_coeff, pts_2d);
        cv::Mat img_show = cv::Mat::zeros(img.size(),CV_8UC3);//img;
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

    void filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_filter(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::transformPointCloud (*cloud_in, *cloud_to_filter, _sequence_transform);

        pcl::RadiusOutlierRemoval<pcl::PointXYZ> rd_filter;
	    rd_filter.setInputCloud(cloud_to_filter);
	    rd_filter.setRadiusSearch(0.3);
	    rd_filter.setMinNeighborsInRadius(2);
	    rd_filter.filter(*cloud_tmp);

        pcl::VoxelGrid<pcl::PointXYZ> sor;//滤波处理对象
        sor.setInputCloud(cloud_tmp);
        sor.setLeafSize(0.1f, 0.1f, 0.2f);//设置滤波器处理时采用的体素大小的参数
        sor.filter(*cloud_out);
    }


    void pairAlign (const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, 
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt
                    )
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_pair(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud (*cloud_src, *cloud_to_pair, _sequence_transform);
        Eigen::Matrix4d trans_rst;
        tergeo::algorithm::CloudPairSolver cloud_solver(cloud_to_pair, cloud_tgt);
        cloud_solver.gicp(trans_rst, 0.2, 0.001, 1e-6, 300);
        _final_transform = trans_rst.cast<float>();
        _to_former = _final_transform * _sequence_transform;
        _init_to_now = _former_to_init * _to_former;
        _former_to_init = _init_to_now;
    }
    void display_pair_result(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, 
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_former_show){
        _after_pair->clear();
        pcl::transformPointCloud (*cloud_src, *_after_pair, _to_former);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_from_init(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud (*cloud_src, *cloud_from_init, _init_to_now);
        *cloud_all += *cloud_from_init;
        *_after_pair += *cloud_former_show;
    }

void pubish_odom(const Eigen::Matrix4f status_matrix){

    // ros_msgs::PointXYZV pose = refer_trj_in_area.points[i];
    // nav_msgs::Odometry odom;
    // odom.header.stamp = ros::Time();
    // odom.header.frame_id = "map";

    // //set the position
    // odom.pose.pose.position.x = pose.x;
    // odom.pose.pose.position.y = pose.y;
    // odom.pose.pose.position.z = 0.0;
    // odom.pose.pose.orientation.x = pose.qx;
    // odom.pose.pose.orientation.y = pose.qy;
    // odom.pose.pose.orientation.z = pose.qz;
    // odom.pose.pose.orientation.w = pose.qw;

    // //set the velocity
    // odom.child_frame_id = "rslidar";
    // odom.twist.twist.linear.x = 0;
    // odom.twist.twist.linear.y = 0;
    // odom.twist.twist.angular.z = 0;
    // publisher.publish(odom);



    Eigen::Matrix3f tmp = status_matrix.block<3,3>(0,0);
    Eigen::Vector3f tmp_position = status_matrix.block<3,1>(0,3);

    Eigen::Matrix<float, 3, 1> RPY = tergeo::common::eigen::RotationToRPY(tmp);
    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(RPY(0),-RPY(1),-RPY(2));
   _odomAftMapped.header.stamp = ros::Time::now();
   _odomAftMapped.header.frame_id = "rslidar";

   _odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
   _odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
   _odomAftMapped.pose.pose.orientation.z = geoQuat.x;
   _odomAftMapped.pose.pose.orientation.w = geoQuat.w;
   _odomAftMapped.pose.pose.position.x = tmp_position(0);
   _odomAftMapped.pose.pose.position.y = tmp_position(1);
   _odomAftMapped.pose.pose.position.z = tmp_position(2);
   _odomAftMapped.twist.twist.angular.x = RPY(0);
   _odomAftMapped.twist.twist.angular.y = RPY(1);
   _odomAftMapped.twist.twist.angular.z = RPY(2);
   _odomAftMapped.twist.twist.linear.x = tmp_position(0);;
   _odomAftMapped.twist.twist.linear.y = tmp_position(1);
   _odomAftMapped.twist.twist.linear.z = tmp_position(2);
   _pubOdomAftMapped.publish(_odomAftMapped);
}


private:  
    ros::NodeHandle nh;   
    ros::Publisher _pub = nh.advertise<sensor_msgs::PointCloud2>("/after_merge", 1);
    ros::Publisher _pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/laser_Odom_my", 1);
    ros::Publisher _pubTmpPoincCloud = nh.advertise<sensor_msgs::PointCloud2>("/middleCloud", 1);


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
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_former_toshow {new pcl::PointCloud<pcl::PointXYZ>};//加工后的pcl格式

    pcl::PointCloud<pcl::PointXYZ>::Ptr _after_pair{new pcl::PointCloud<pcl::PointXYZ>};

    sensor_msgs::PointCloud2 cloud_out;//转换为ros格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_last{new pcl::PointCloud<pcl::PointXYZ>};
    
    //right lidar to left lidar
    Eigen::Matrix4f _pairTransform = Eigen::Matrix4f::Identity (); //左右lidar 相对位姿
    Eigen::Matrix4f _imu_to_lidar = Eigen::Matrix4f::Identity (); //左右lidar 相对位姿

    Eigen::Matrix4f _former_to_init = Eigen::Matrix4f::Identity (); //左右lidar 相对位姿
    Eigen::Matrix4f _final_transform = Eigen::Matrix4f::Identity (); 
    Eigen::Matrix4f _init_to_now = Eigen::Matrix4f::Identity (); 
    Eigen::Matrix4f _to_former = Eigen::Matrix4f::Identity (); 

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

