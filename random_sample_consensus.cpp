#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <boost/thread/thread.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace std;

void extract_torso(IplImage* src, IplImage* torso_image, CvPoint* right_bm, CvPoint* left_tp, int* widthstep);
void bounding_box( IplImage* src, CvPoint* right_bm, CvPoint* left_tp);
void extract_torso_pc( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr torso_total_cloud,
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, CvPoint right_bm, CvPoint left_tp, int widthstep); 
void frontal_plane( pcl::ModelCoefficients::Ptr coeff, pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud);
void fs_intersection( pcl::PointCloud<pcl::PointXYZ>::Ptr intersection, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, IplImage* torso_image);
void sagittal_plane( pcl::ModelCoefficients::Ptr sagittal_coeff, pcl::ModelCoefficients::Ptr frontal_coeff, pcl::PointCloud<pcl::PointXYZ>::Ptr intersection); 
void coordinate_transform( pcl::ModelCoefficients::Ptr frontal_coeff, pcl::ModelCoefficients::Ptr sagittal_coeff,
		pcl::PointCloud<pcl::PointXYZ>::Ptr torso_total_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);
void adjust_coordinate( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);
void finding_mean_xy( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, double xmean, double ymean);
void trajectory_line1( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean);
void trajectory_line2( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean);
void trajectory_circle1( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean);
void trajectory_circle2( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean);
void trajectory_shoulder( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean);

boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// -----Open 3D viewer and add point cloud-----
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	
	return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer>
rgbVis (pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud)
{
	// -----Open 3D viewer and add point cloud-----
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud); 
	viewer->setBackgroundColor (0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, rgb, "sample cloud");
	//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
//	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	
	return (viewer);
}
	
int
main(int argc, char** argv)
{
	
	// Main loop
	// k: number of images
	for( int k = 1; k < 2; k++)
	{
		// read point cloud, image and label files
		stringstream ss;		ss<<k;
		string image_filename = "./experimental_data/" + ss.str() + "/image.png";
		string label_filename = "./experimental_data/" + ss.str() + "/label.png";
		string pointcloud_filename = "./experimental_data/" + ss.str() + "/pointcloud.pcd";
		
		// image & label
		IplImage* image = cvLoadImage(image_filename.c_str());
		IplImage* label = cvLoadImage(label_filename.c_str());
		IplImage* torso_image = cvCreateImage( cvSize(image->width, image->height), 8, 1);

		// point cloud
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::io::loadPCDFile (pointcloud_filename, *cloud);

		// Create the filtering object
		pcl::PassThrough<pcl::PointXYZRGBA> pass;
		pass.setInputCloud (cloud);
	    pass.setFilterFieldName ("z");
		pass.setFilterLimits (0.0, 3.0);
		pass.filter (*cloud);

		pass.setInputCloud (cloud);
		pass.setFilterFieldName ("y");
		pass.setFilterLimits (-0.8, 1.0);
		pass.filter (*cloud);

		// enlarge the scale
	/*
		for( int i = 0; i < cloud->size(); i++)
		{
			cloud->points[i].x *= 3;
			cloud->points[i].y *= 3;
			cloud->points[i].z *= 3;
		}
*/
		// extract central part in bounding box and prepare to estimate frontal plane
		CvPoint left_tp, right_bm;
		int widthstep;
		pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr torso_total_cloud (new pcl::PointCloud<pcl::PointXYZ>);

        // find the torso parts
		extract_torso( label, torso_image, &right_bm, &left_tp, &widthstep);
		extract_torso_pc( torso_cloud, torso_total_cloud, cloud, right_bm, left_tp, widthstep);	
		
		// frontal plane estimation 
		pcl::ModelCoefficients::Ptr frontal_coeff (new pcl::ModelCoefficients);
		frontal_plane( frontal_coeff, torso_cloud);

		// extract the point across left and right chest, and prepare to estimate the central line
		pcl::PointCloud<pcl::PointXYZ>::Ptr intersection (new pcl::PointCloud<pcl::PointXYZ>);
		fs_intersection( intersection, cloud, torso_image);

		// sagittal plane estimation
		pcl::ModelCoefficients::Ptr sagittal_coeff (new pcl::ModelCoefficients);
		sagittal_plane( sagittal_coeff, frontal_coeff, intersection);

		// coordinate transform
		coordinate_transform( frontal_coeff, sagittal_coeff, torso_total_cloud, cloud);		

		// refine the coordinate by determining the pricinpal planes
		adjust_coordinate( torso_total_cloud, cloud);

		// find the max and min, x and y values
		double xmean, ymean;
		finding_mean_xy( torso_total_cloud, xmean, ymean); 

		// specified massage trajectory generator
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr trajectory (new pcl::PointCloud<pcl::PointXYZRGBA>);
		trajectory_line1( torso_total_cloud, trajectory, xmean, ymean);
		trajectory_line2( torso_total_cloud, trajectory, xmean, ymean);
		trajectory_circle1( torso_total_cloud, trajectory, xmean, ymean);
		trajectory_circle2( torso_total_cloud, trajectory, xmean, ymean);
		trajectory_shoulder( torso_total_cloud, trajectory, xmean, ymean);

		
		// point cloud visualization
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
		viewer = rgbVis( cloud);
		viewer->addCoordinateSystem (1.0);
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(trajectory);
		viewer->addPointCloud<pcl::PointXYZRGBA>(trajectory, rgb, "trajectory");
//		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "trajectory");
//		viewer->addPlane( *coeff_sagittal, "sagittal");
//		viewer->addPlane( *coeff_f, "frontal");
//		viewer->addLine( *coeff_s, "line");
		while (!viewer->wasStopped ())
		{
			viewer->spinOnce (100);
			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		}
		
		// release memeories
		cvReleaseImage(&image);
		cvReleaseImage(&label);
		cvReleaseImage(&torso_image);
		
	
	}
	
	return 0;
}

void extract_torso(IplImage* src, IplImage* torso_image, CvPoint* right_bm, CvPoint* left_tp, int* widthstep)
{
	IplImage* dst = cvCreateImage( cvSize(src->width, src->height), 8, 1);
    cvZero( dst);	cvZero( torso_image);
    
	int color_table[2][3] = { {246,198,0}, {202,177,251}};
    for( int v = 0; v < src->height; v++)
		for( int u = 0; u < src->width; u++)
	    {
			unsigned char r, g, b;
			r = src->imageData[v*src->widthStep+3*u];
			g = src->imageData[v*src->widthStep+3*u+1];
			b = src->imageData[v*src->widthStep+3*u+2];
		  
			for( int k = 0; k < 2; k++)
				if( r == color_table[k][0] && g == color_table[k][1] && b == color_table[k][2])
				{
					dst->imageData[v*dst->widthStep+u] = 255;
					if( k)
						torso_image->imageData[v*torso_image->widthStep+u] = 1;
					else
						torso_image->imageData[v*torso_image->widthStep+u] = 2;
				}
		}
  
	// find the largest part 
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* contour;
	cvFindContours(dst, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		
	int max_num = contour->total;
	CvSeq* largest_contour = contour;

	while(contour->h_next)
	{
		contour = contour->h_next;
		if( contour->total > max_num)
		{
			max_num = contour->total;
			largest_contour = contour;
		}
	}

	IplImage* dst2 = cvCreateImage( cvSize( dst->width, dst->height), 8, 1);
	cvZero(dst2);
	CvPoint* Point = new CvPoint [largest_contour->total];
	CvSeqReader reader;
	cvStartReadSeq( largest_contour, &reader, 0);
	
	for( int i = 0; i < largest_contour->total; i++)
	{
		CV_READ_SEQ_ELEM( Point[i], reader);
		dst2->imageData[Point[i].y*dst2->widthStep+Point[i].x] = 255;
	}
	
	// fill the color in the contour
	cvFloodFill( dst2, cvPoint( Point[0].x+2, Point[0].y+2), cvScalarAll(255), cvRealScalar(0), cvRealScalar(0), NULL, 4);
  
	bounding_box( dst2, right_bm, left_tp);
	*widthstep = dst->widthStep;	

	cvReleaseImage(&dst);
	cvReleaseImage(&dst2);
	cvReleaseMemStorage(&storage);
	delete [] Point;
}

void bounding_box( IplImage* src, CvPoint* right_bm, CvPoint* left_tp)
{
	// find the bounding box
	*left_tp = cvPoint( src->width, src->height);
	*right_bm = cvPoint( 0, 0);
	for( int i = 0; i < src->height; i++)
		for( int j = 0; j < src->width; j++)		
		{
			if( src->imageData[i*src->widthStep+j])
			{
				if( left_tp->x > j)
					left_tp->x = j;
				if( left_tp->y > i)
					left_tp->y = i;
				if( right_bm->x < j)
					right_bm->x = j;
				if( right_bm->y < i)
					right_bm->y = i;
			}	  
		}
}

void extract_torso_pc( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr torso_total_cloud,
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, CvPoint right_bm, CvPoint left_tp, int widthstep) 
{
	torso_cloud->is_dense = 0;
	const int offset = 30;
	for( int i = left_tp.y; i < right_bm.y; i++)
		for( int j = left_tp.x; j < right_bm.x; j++)
		{
			pcl::PointXYZ pt;
			int index = i*widthstep+j;
			pt.x = cloud->points[index].x;
			pt.y = cloud->points[index].y;
			pt.z = cloud->points[index].z;
			
			if( i > left_tp.y + offset && i < right_bm.y - offset)
				if( j > left_tp.x + offset && j < right_bm.x -offset)
					torso_cloud->push_back(pt);
			torso_total_cloud->push_back(pt);
		}
}

void frontal_plane( pcl::ModelCoefficients::Ptr coeff, pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud)
{
	// Remove the nan from cloud
	vector<int> index_torso_cloud;
    pcl::removeNaNFromPointCloud( *torso_cloud, *torso_cloud, index_torso_cloud);

	// RANSAC estimation
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients (true);
	
	seg.setModelType( pcl::SACMODEL_PLANE);
	seg.setMethodType( pcl::SAC_RANSAC);
	seg.setDistanceThreshold( 0.01);
	
	seg.setInputCloud( torso_cloud->makeShared());
	seg.segment(*inliers, *coeff);
}



void fs_intersection( pcl::PointCloud<pcl::PointXYZ>::Ptr intersection, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, IplImage* torso_image)
{
	for( int v = 0; v < torso_image->height; v++)
		for( int u = 0; u < torso_image->width-1; u++)
		{
			unsigned char p1, p2;
			p1 = torso_image->imageData[v*torso_image->widthStep+u];
			p2 = torso_image->imageData[v*torso_image->widthStep+u+1];
			
			if( p1 == 2 && p2 == 1)
			{
				pcl::PointXYZ pt;
				int index = v*torso_image->widthStep+u;
				pt.x = cloud->points[index].x;
				pt.y = cloud->points[index].y;
				pt.z = cloud->points[index].z;

				intersection->push_back(pt);
			}
		}
}

void sagittal_plane( pcl::ModelCoefficients::Ptr sagittal_coeff, pcl::ModelCoefficients::Ptr frontal_coeff, pcl::PointCloud<pcl::PointXYZ>::Ptr intersection)
{
	// Remove the nan from cloud
	vector<int> index_cloud;
	pcl::removeNaNFromPointCloud( *intersection, *intersection, index_cloud);
   
	// project the points onto the frontal plane
	pcl::ProjectInliers<pcl::PointXYZ> proj;
	proj.setModelType (pcl::SACMODEL_PLANE);
	proj.setInputCloud (intersection);
	proj.setModelCoefficients (frontal_coeff);
	proj.filter (*intersection);

	// RANSAC estimation
	pcl::ModelCoefficients::Ptr line_coeff (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers_s (new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients (true);
	
	seg.setModelType( pcl::SACMODEL_LINE);
	seg.setMethodType( pcl::SAC_RANSAC);
	seg.setDistanceThreshold( 0.01);

	seg.setInputCloud( intersection->makeShared());
	seg.segment(*inliers_s, *line_coeff);
	// coefficient format of line: [point_on_line.x point_on_line.y point_on_line.z line_direction.x line_direction.y line_direction.z]

	// cross product
	Eigen::Vector3d v( frontal_coeff->values[0], frontal_coeff->values[1], frontal_coeff->values[2]);
	Eigen::Vector3d w( line_coeff->values[3], line_coeff->values[4],line_coeff->values[5]);
	Eigen::Vector3d cp = v.cross(w);

	sagittal_coeff->values.push_back(cp(0));
	sagittal_coeff->values.push_back(cp(1));
	sagittal_coeff->values.push_back(cp(2));
	sagittal_coeff->values.push_back( -cp(0)*line_coeff->values[0] - cp(1)*line_coeff->values[1] - cp(2)*line_coeff->values[2] );
}

void coordinate_transform( pcl::ModelCoefficients::Ptr frontal_coeff, pcl::ModelCoefficients::Ptr sagittal_coeff,
	pcl::PointCloud<pcl::PointXYZ>::Ptr torso_total_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
	// rotate alpha, beta, gamma angles to let two coordinate parallel
	double beta, gamma;
	Eigen::Matrix3d R, Rx, Ry;
	Eigen::Vector3d nt, nt1;
	Eigen::Vector3d nf( frontal_coeff->values[0], frontal_coeff->values[1], frontal_coeff->values[2]);
	Eigen::Vector3d ns( sagittal_coeff->values[0], sagittal_coeff->values[1], sagittal_coeff->values[2]);

	pcl::ModelCoefficients::Ptr coeff;
	coeff = sagittal_coeff;

	beta = acos( ns(0) / sqrt( ns(0)*ns(0)+ns(1)*ns(1)+ns(2)*ns(2)) );
	Ry << cos(beta), 0, sin(beta), 0, 1, 0, -sin(beta), 0, cos(beta);
	nt = ns.cross(nf);
	
	nt1 = Ry*nt;
	gamma = acos( -nt1(1) / sqrt(nt1(0)*nt1(0)+nt1(1)*nt1(1)+nt1(2)*nt1(2)));
	Rx<< 1, 0, 0, 0, cos(gamma), -sin(gamma), 0, sin(gamma), cos(gamma);

	Eigen::Vector3d pt, pt1;
	R = Rx*Ry; 
	for( int i = 0; i < cloud->points.size(); i++)
	{
		pt(0) = cloud->points[i].x;
		pt(1) = cloud->points[i].y;
		pt(2) = cloud->points[i].z;
		pt1 = R*pt;
		cloud->points[i].x = pt1(0);
		cloud->points[i].y = pt1(1);
		cloud->points[i].z = pt1(2);
	}

	for( int i = 0; i < torso_total_cloud->points.size(); i++)
	{
		pt(0) = torso_total_cloud->points[i].x;
		pt(1) = torso_total_cloud->points[i].y;
		pt(2) = torso_total_cloud->points[i].z;
		pt1 = R*pt;
		torso_total_cloud->points[i].x = pt1(0);
		torso_total_cloud->points[i].y = pt1(1);
		torso_total_cloud->points[i].z = pt1(2);
	}
}

void adjust_coordinate( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
	torso_cloud->is_dense = 0;
	// Remove the nan from cloud
	vector<int> index_cloud;
	pcl::removeNaNFromPointCloud( *torso_cloud, *torso_cloud, index_cloud);
	
	double cx = 0, cy = 0, cz = 0;
	int pnum = torso_cloud->size();
	// extract the points on y-z plane
	// calculate the centroid
	for( int i = 0; i < pnum; i++)
	{
		cx += torso_cloud->points[i].x;
		cy += torso_cloud->points[i].y;
		cz += torso_cloud->points[i].z;
 	}

	cx /= pnum;
	cy /= pnum;
	cz /= pnum;
	
	cout<<cx<<endl;

	double y, z;
	double a, b, c;
	// a, b, c
	for( int i = 0; i < pnum; i++)
	{
		y = torso_cloud->points[i].y - cy;
		z = torso_cloud->points[i].z - cz;	
		a += y*y;
		b += 2*y*z;
		c += z*z;
	}

	// determine the angle
	double ddot, th;
	ddot = atan( (a-c)/b);
	th = 0.5*atan( b/(a-c));
	
	cout<<"th: "<<th*360/3.14159<<endl;
	cout<<"ddot: "<<ddot<<endl;
	// ddot < 0, there is minimum moment of inertia
	if( ddot < 0)
	{
		if( th > 0)
			th = th - 3.14159/2;
		else
			th = th + 3.14159/2;
	}

	// rotate along x-axis
	Eigen::Matrix3d Rth;
	Rth<< cos(th), -sin(th), 0, sin(th), cos(th), 0, 0, 0, 1;
	Rth<< 1, 0, 0, 0, cos(th), -sin(th), 0, sin(th), cos(th);
	cout<<th*360/3.14159<<endl;

	Eigen::Vector3d pt, pt1;

	for( int i = 0; i < cloud->points.size(); i++)
	{
		pt(0) = cloud->points[i].x;
		pt(1) = cloud->points[i].y;
		pt(2) = cloud->points[i].z;
		pt1 = Rth*pt;
		cloud->points[i].x = pt1(0);
		cloud->points[i].y = pt1(1);
		cloud->points[i].z = pt1(2);
	}

	for( int i = 0; i < torso_cloud->points.size(); i++)
	{
		pt(0) = torso_cloud->points[i].x;
		pt(1) = torso_cloud->points[i].y;
		pt(2) = torso_cloud->points[i].z;
		pt1 = Rth*pt;
		torso_cloud->points[i].x = pt1(0);
		torso_cloud->points[i].y = pt1(1);
		torso_cloud->points[i].z = pt1(2);
	}

}

void finding_mean_xy( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, double xmean, double ymean)
{
	double xmax = -1e10, ymax = -1e10, xmin = 1e10, ymin = 1e10;
	for( int i = 0; i < torso_cloud->points.size(); i++)
	{
		double x, y;
		x = torso_cloud->points[i].x;
		y = torso_cloud->points[i].y;

		if( x > xmax)
		  xmax = x;
		if( y > ymax)
		  ymax = y;
		if( x < xmin)
		  xmin = x;
		if( y < ymin)
		  ymin = y;
	}

	xmean = (xmax + xmin) / 2.0;
	ymean = (ymax + ymin) / 2.0;

	cout<<"xmax:"<<xmax<<"  xmin:"<<xmin<<endl;
	cout<<"ymax:"<<ymax<<"  ymin:"<<ymin<<endl;
}

void trajectory_line1( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean)
{
	for( int i = 0; i < torso_cloud->points.size(); i++)
	{
		pcl::PointXYZRGBA pt;
		pt.x = torso_cloud->points[i].x;	
		pt.y = torso_cloud->points[i].y;	
		pt.z = torso_cloud->points[i].z;
			
		if( fabs (pt.x -(xmean+0.06)) < 0.003 || fabs (pt.x - (xmean-0.04)) < 0.003)
		{
			pt.r = 255;   pt.g = 0;   pt.b = 0;
			traj->push_back(pt);	  
		}
	}
}

void trajectory_line2( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean)
{
	double y1, y2, y3;
	y1 = ymean - 0.08;
	y2 = ymean - 0.16;
	y3 = ymean - 0.24;

	double th = 0.005;

	double x1, x2, x3, x4;
	x1 = xmean + 0.03;
	x2 = xmean + 0.12;
	x3 = xmean - 0.03;
	x4 = xmean - 0.12;

	for( int i = 0; i < torso_cloud->points.size(); i++)
	{
		pcl::PointXYZRGBA pt;
		pt.x = torso_cloud->points[i].x;	
		pt.y = torso_cloud->points[i].y;	
		pt.z = torso_cloud->points[i].z;

		if( fabs (pt.y - y1) < th &&  (pt.x > x1) && (pt.x < x2) )
		{
			pt.r = 0;	pt.g = 255;		pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs (pt.y - y1) < th &&  (pt.x < x3) && (pt.x > x4) )
		{
			pt.r = 0;	pt.g = 255;		pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs (pt.y - y2) < th &&  (pt.x > x1) && (pt.x < x2) )
		{
			pt.r = 0;	pt.g = 255;		pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs (pt.y - y2) < th &&  (pt.x < x3) && (pt.x > x4) )
		{
			pt.r = 0;	pt.g = 255;		pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs (pt.y - y3) < th &&  (pt.x > x1) && (pt.x < x2) )
		{
			pt.r = 0;	pt.g = 255;		pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs (pt.y - y3) < th &&  (pt.x < x3) && (pt.x > x4) )
		{
			pt.r = 0;	pt.g = 255;		pt.b = 0;
			traj->push_back(pt);
		}

	}
}

void trajectory_circle1( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean)
{
	double th = 0.0001;
	double r = 0.02;
	double h1, h2, k1, k2, k3;
	h1 = xmean + 0.08;
	h2 = xmean - 0.06;
	k1 = -0.15;
	k2 = -0.22;
	k3 = -0.29;

	for( int i = 0; i < torso_cloud->points.size(); i++)
	{
		pcl::PointXYZRGBA pt;
		pt.x = torso_cloud->points[i].x;	
		pt.y = torso_cloud->points[i].y;	
		pt.z = torso_cloud->points[i].z;

		// circle trajectory
		if( fabs( (pt.x-h1)*(pt.x-h1) + (pt.y-k1)*(pt.y-k1) - r*r) < th )
		{
			pt.r = 255;	pt.g = 255;	pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs( (pt.x-h1)*(pt.x-h1) + (pt.y-k2)*(pt.y-k2) - r*r) < th )
		{
			pt.r = 255;	pt.g = 255;	pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs( (pt.x-h1)*(pt.x-h1) + (pt.y-k3)*(pt.y-k3) - r*r) < th )
		{
			pt.r = 255;	pt.g = 255;	pt.b = 0;
			traj->push_back(pt);
		}
			
		if( fabs( (pt.x-h2)*(pt.x-h2) + (pt.y-k1)*(pt.y-k1) - r*r) < th )
		{
			pt.r = 255;	pt.g = 255;	pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs( (pt.x-h2)*(pt.x-h2) + (pt.y-k2)*(pt.y-k2) - r*r) < th )
		{
			pt.r = 255; pt.g = 255;	pt.b = 0;
			traj->push_back(pt);
		}

		if( fabs( (pt.x-h2)*(pt.x-h2) + (pt.y-k3)*(pt.y-k3) - r*r) < th )
		{
			pt.r = 255;	pt.g = 255;	pt.b = 0;
			traj->push_back(pt);
		}
	}
}

void trajectory_shoulder( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean)
{
	IplImage* edge = cvCreateImage( cvSize(100, 200), 8, 1);
	cvZero(edge);
	vector<int> indices;
	int u, v;
	for( int i = 0; i < torso_cloud->points.size(); i++)
	{
		pcl::PointXYZRGBA pt;
		pt.x = torso_cloud->points[i].x;	
		pt.y = torso_cloud->points[i].y;	
		pt.z = torso_cloud->points[i].z;

		u = 50  + (int)(pt.x*200);
		v = 200 + (int)(pt.y*200);
		if( u >= 0 && u < 100 && v >= 0 && v < 200)
			edge->imageData[v*edge->widthStep+u] = 255;
	}
	
	IplImage* shoulder = cvCreateImage( cvSize(100, 200), 8, 1);
	cvZero(shoulder);
	for( int i = 0; i < 100; i++)
	{
		for( int j = 0; j < 198; j++)
			if( edge->imageData[j*edge->widthStep+i] != 0)
			{
				shoulder->imageData[j*edge->widthStep+i] = 255;
				shoulder->imageData[(j+1)*edge->widthStep+i] = 255;
				shoulder->imageData[(j+2)*edge->widthStep+i] = 255;
				break;
			}
	
	
	}

	pcl::PointXYZRGBA pt;
	pt.z = 2.3;
	double x1, x2;
	x1 = xmean + 0.09;
	x2 = xmean - 0.07;
	for( int i = 0; i < 200; i++)
	{
		for( int j = 0; j < 100; j++)
		{
			if( shoulder->imageData[i*edge->widthStep+j] != 0)
			{
				pt.x = (j-50)/200.0;
				pt.y = (i-200)/200.0;
				pt.r = 200;
				pt.g = 100;
				pt.b = 100;
				
				if( pt.x > x1 || pt.x < x2)
					traj->push_back(pt);
			}
		}
	}
	
	cvReleaseImage( &edge);
	cvReleaseImage( &shoulder);
}

void trajectory_circle2( pcl::PointCloud<pcl::PointXYZ>::Ptr torso_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr traj, double xmean, double ymean)
{
	double th = 0.0005;
	double r = 0.06;
	double h1, h2, k1, k2, k3;
	h1 = xmean + 0.10;
	h2 = xmean - 0.08;
	k1 = -0.35;

	for( int i = 0; i < torso_cloud->points.size(); i++)
	{
		pcl::PointXYZRGBA pt;
		pt.x = torso_cloud->points[i].x;	
		pt.y = torso_cloud->points[i].y;	
		pt.z = torso_cloud->points[i].z;

		// circle trajectory
		if( fabs( (pt.x-h1)*(pt.x-h1) + (pt.y-k1)*(pt.y-k1) - r*r) < th )
		{
			pt.r = 55;	pt.g = 255;	pt.b = 200;
			traj->push_back(pt);
		}

		if( fabs( (pt.x-h2)*(pt.x-h2) + (pt.y-k1)*(pt.y-k1) - r*r) < th )
		{
			pt.r = 55;	pt.g = 255;	pt.b = 200;
			traj->push_back(pt);
		}
	}
}
