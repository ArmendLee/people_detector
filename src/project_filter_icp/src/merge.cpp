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
        // std::cout<<pairTransform<<std::endl;


    }  


    void combineCallback(const detector_msg::detection_resultConstPtr &dt_result,
                        const sensor_msgs::PointCloud2ConstPtr &left, 
                        const sensor_msgs::PointCloud2ConstPtr &right){ 
        
        // std::cout<<dt_result->boxs.size()<<std::endl;
        dt_result->boxs[0].l_x;
        pcl::fromROSMsg (*right, *cloud_right);
        // ROS_INFO("get right cloud message");
        pcl::fromROSMsg (*left, *cloud_left);
        // ROS_INFO("get left cloud message");
        cloud_raw_merge->clear();
        *cloud_raw_merge += *cloud_left;
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());    
        pcl::transformPointCloud (*cloud_right, *transformed_cloud, _pairTransform);
        *cloud_raw_merge += *transformed_cloud;
        // std::cout<<cloud_raw_merge->size()<<std::endl;

        project_pc_image(image_origin,cloud_raw_merge,dt_result);
        
        const auto t1 = std::chrono::high_resolution_clock::now();
        if(cloud_former->size()>0){
            pairAlign(cloud_raw_merge, cloud_former, cloud_all, _sequence_transform, false);
            std::cout<<_sequence_transform<<std::endl;
        }
        const auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Used time:" << (t2-t1).count()*1e-6 << "ms" << std::endl;
        cloud_former->clear();
        pcl::copyPointCloud(*cloud_raw_merge,*cloud_former);


        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(*cloud_raw_merge, ros_cloud);
        ros_cloud.header = left->header;
        ros_cloud.header.frame_id = "camera_init";
        _pub.publish(ros_cloud);
    }

    void data_clean_remove_movable(){
        
    }


    bool in_bounding_box(cv::Point2f &pt,const float &l_x,const float &l_y,const float &r_x,const float &r_y){
        if( pt.x < l_x || pt.y < l_y || pt.x > r_x || pt.y > r_y){
            return false;
            }
        // std::cout<<pt<<'\t'<<l_x<<'\t'<<l_y<<'\t'<<r_x<<'\t'<<r_y<<'\n';
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
            // if (point_3d.x > 2  && point_3d.y > 2 && point_3d.z > -2)
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
        std::cout<<pts_2d.size()<<std::endl;
        int count=0;
        bool outside;
        std::cout<<dt_result->boxs.size()<<std::endl;
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
                // cv::circle(img_show, point_2d, 5, cv::Scalar(255, 0, 0), -1);
            }
        }


            // if (point_2d.x < 0 || point_2d.x > image_cols || point_2d.y < 0 || point_2d.y > image_rows){
            //     continue;
            // }
            // else{
            //     cv::circle(img_show, point_2d, 5, cv::Scalar(255, 0, 0), -1);          
            //     ++count;
            // }

            // if (point_2d.x > 0 && point_2d.x < image_cols && point_2d.y > 0 && point_2d.y < image_rows){
            //     cv::circle(img_show, point_2d, 5, cv::Scalar(255, 0, 0), -1);
            //     ++count;
            // } 
            // else{
            //     continue;
            // }  
        cv::namedWindow("img_projection",0);
        cv::imshow("img_projection", img_show);
        cv::waitKey(3); 
    }


    void pairAlign (const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
    {
        pcl::IterativeClosestPoint<PointT, PointT> icp;
	    icp.setMaximumIterations(20);
	    icp.setInputSource(cloud_tgt);
	    icp.setInputTarget(cloud_src);
	    icp.align(*cloud_tgt);
	    //icp.setMaximumIterations(1);  // We set this variable to 1 for the next time we will call .align () function
	    //std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;
    
	    if (icp.hasConverged())
	    {
	    	// std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
	    	// std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
	    	final_transform = icp.getFinalTransformation().cast<float>();
	    	// print4x4Matrix(transformation_matrix);
	    }
	    else
	    {
	    	ROS_ERROR("\nICP has not converged.\n");
	    	// system("pause");
	    	// return (-1);
	    }
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp;
        pcl::transformPointCloud (*cloud_tgt, *tmp, final_transform);
        *output += *tmp;
        
        
        // PointCloud::Ptr src (new PointCloud);//倒贴的
        // PointCloud::Ptr tgt (new PointCloud);//被贴的
        // pcl::VoxelGrid<PointT> grid;//下采样
        // if (downsample){
        //     grid.setLeafSize (0.05, 0.05, 0.05);
        //     grid.setInputCloud (cloud_src);
        //     grid.filter (*src);
        //     grid.setInputCloud (cloud_tgt);
        //     grid.filter (*tgt);
        // }else{
        //     src = cloud_src;
        //     tgt = cloud_tgt;
        // }

        // PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);//倒贴的变成带法线的
        // PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);
        // pcl::NormalEstimation<PointT, PointNormalT> norm_est;//法线估算对象
        // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());//kdtree对象，Kd_tree是一种数据结构便于管理点云以及搜索点云，法线估计对象会使用这种结构来找到最近邻点
        // norm_est.setSearchMethod (tree);//使用kdtree的方法搜索估计法线
        // norm_est.setKSearch (30);//设置最近邻的数量，或者用setRadiusSearch，对每个点设置一个半径，用来确定法线和曲率
        // norm_est.setInputCloud (src);
        // norm_est.compute (*points_with_normals_src);//计算得到带每个点的法向量
        // pcl::copyPointCloud (*src, *points_with_normals_src);//将点云复制过去，最后得到的点云，每个点都包含坐标和法向量
        // norm_est.setInputCloud (tgt);//对被倒贴的也这样操作,弄成带法线的
        // norm_est.compute (*points_with_normals_tgt);
        // pcl::copyPointCloud (*tgt, *points_with_normals_tgt);


        // MyPointRepresentation point_representation;

        // float alpha[4] = {1.0, 1.0, 1.0, 1.0};//这个是干嘛的？加权曲率维度，以和坐标xyz保持平衡，没看懂
        // point_representation.setRescaleValues (alpha);


        // pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;//创建非线性ICP对象
        // reg.setTransformationEpsilon (1e-6);//这个值一般设为1e-6或者更小。意义是什么？
        // reg.setMaxCorrespondenceDistance (1.0);//设置对应点之间的最大距离（0.1m）,在配准过程中，忽略大于该阈值的点
        // reg.setMaximumIterations (40);//迭代次数，几十上百都可能出现

        // reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));//括号内类比于：int a？？？ make_shared<T>:C创建一个shared_ptr共享指针
        // reg.setInputSource (points_with_normals_src);//非线性icp主角：带法线的点云
        // reg.setInputTarget (points_with_normals_tgt);

        // Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;//声明3个变量，其中Ti=Eigen::Matrix4f::Identity ()创建一个单位矩阵
        // PointCloudWithNormals::Ptr reg_result = points_with_normals_src;//存放icp结果

        // for (int i = 0; i < 30; ++i)//这个for循环是和距离阈值有关的，不断的减小距离阈值30次，相当于不断地离目标点云越来越近
        // {
        //    // PCL_INFO ("Iteration Nr. %d.\n", i);

        //     points_with_normals_src = reg_result;//将上次的拼接结果作为本次拼接的倒贴者。因为在for循环里，不是每次都定义，所以直接用reg_result这个变量

        //     reg.setInputSource (points_with_normals_src);
        //     reg.align (*reg_result);//拼接

        //     Ti = reg.getFinalTransformation()*Ti;//将本次最终的转换矩阵累积到之前的转换矩阵

        //     if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())//如果上次和本次的转换矩阵差值的元素和小于我们设置的值（就是相比上一次动不了多少了）
        //         reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.01);//就将距离阈值设的小一些，把一些远的点去掉，再继续（for循环中）匹配
        //     prev = reg.getLastIncrementalTransformation ();//更新上一次的变换


        // }

        // targetToSource = Ti.inverse();//使倒贴者和被倒贴者互换的矩阵

        // pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);//被倒贴变倒贴了

        // *output += *cloud_src;//贴过去以后和人加一块，变成一家人。拼接完成。
        // final_transform = targetToSource;
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

