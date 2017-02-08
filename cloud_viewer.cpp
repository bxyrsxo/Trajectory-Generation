#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/filters/filter.h>
#include "cv.h"
#include "highgui.h"
#include <string>
#include <sstream>
#include <fstream>

using namespace std;

int user_data;
    
void 
viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (0.0, 0.0, 0.0);
    //pcl::PointXYZ o;
    //o.x = 1.0;
    //o.y = 0;
    //o.z = 0;
    //viewer.addSphere (o, 0.25, "sphere", 0);
    //std::cout << "i only run once" << std::endl;
    
}
    
void 
viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
{
    static unsigned count = 0;
    std::stringstream ss;
    ss << "Once per viewer loop: " << count++;
    viewer.removeShape ("text", 0);
    viewer.addText (ss.str(), 200, 300, "text", 0);
    
    //FIXME: possible race condition here:
    user_data++;
}
    
int
main ()
{
	const int data_set_num = 10;
	int angles[data_set_num] = {0,30,60,120,150,180,210,240,300,330};
 
	
//	for( int i = 0; i < data_set_num; i++)
	for( int i = 0; i < 10; i++)
	{
		// read point cloud, image and label files
		stringstream ss;
		ss<<angles[i];
		string image_filename = "./experimental_data/" + ss.str() + "/image.png";
		string label_filename = "./experimental_data/" + ss.str() + "/label.png";
		string pointcloud_filename = "./experimental_data/" + ss.str() + "/pointcloud.pcd";
		
		// image & label
		IplImage* image = cvLoadImage(image_filename.c_str());
		IplImage* label = cvLoadImage(label_filename.c_str());

		// point cloud
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::io::loadPCDFile (pointcloud_filename, *cloud);

		// find the head parts
		IplImage* head = cvCreateImage( cvSize(image->width, image->height), 8, 1);
		cvZero( head);				
		int head_color_table[4][3] = { {52,183,58}, {54,155,75},{249,61,187},{143,57,11}};
		for( int v = 0; v < label->height; v++)
			for( int u = 0; u < label->width; u++)
			{
				unsigned char r, g, b;
				r = label->imageData[v*label->widthStep+3*u];
				g = label->imageData[v*label->widthStep+3*u+1];
				b = label->imageData[v*label->widthStep+3*u+2];
				
				for( int k = 0; k < 4; k++)
					if( r == head_color_table[k][0] && g == head_color_table[k][1] && b == head_color_table[k][2])
						head->imageData[v*head->widthStep+u] = 255;
			}	

		// pick up the point cloud by head
		pcl::PointCloud<pcl::PointXYZ>::Ptr head_cloud (new pcl::PointCloud<pcl::PointXYZ>);
		head_cloud->is_dense = 0;
		int yue_histogram[16] = {0.0};
		const double NaN = -1*sqrt(-1);

		for( int v = 0; v < head->height; v++)
			for( int u = 0; u < head->width; u++)
				if( head->imageData[v*head->widthStep+u])
				{
					pcl::PointXYZ pt;
					pcl::RGB rgb;
					int index = v*head->width+u;
					pt.x = cloud->points[index].x;
					pt.y = cloud->points[index].y;
					pt.z = cloud->points[index].z;
					
					head_cloud->push_back(pt);

					rgb.r = cloud->points[index].r;
					rgb.g = cloud->points[index].g;
					rgb.b = cloud->points[index].b;

					if( pt.x == NaN)
					  cout<<"nan"<<endl;

					int yue = 0.299*rgb.r + 0.587*rgb.g + 0.114*rgb.b;
					yue_histogram[(int)(yue/16)]++;
				}	 

		// remove nan from head_cloud
		vector<int> index_head_cloud;
		pcl::removeNaNFromPointCloud( *head_cloud, *head_cloud, index_head_cloud);

		float histogram[16];
		// normalize yue histogram
		for( int k = 0; k < 16; k++)
			histogram[k] = yue_histogram[k] / (float)head_cloud->points.size();

		// Normal Estimation
	  	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
		ne.setInputCloud (head_cloud);
		ne.setSearchMethod (tree);
		ne.setRadiusSearch (0.03);				// Use all neighbors in a sphere of radius 3cm
		ne.compute (*normals);

		// save head cloud for debug
		pcl::io::savePCDFileASCII( "head.pcd", *head_cloud);

		// VFH desriptor 
		pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
		vfh.setInputCloud (head_cloud);
		vfh.setInputNormals (normals);
		vfh.setSearchMethod (tree);
		pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
		vfh.compute (*vfhs);
/*		
	    // show the histogram content	
		for( int j = 0; j < 308; j++)
			cout<<j<<"  "<<vfhs->points[0].histogram[j]<<endl;

		// Histogram plot
		pcl::visualization::PCLHistogramVisualizer hist;
		hist.setBackgroundColor( 1.0, 1.0, 1.0);
		hist.addFeatureHistogram( *vfhs, 308);
		hist.spin();
*/

		// write into files
		string output_filename = ss.str() + ".txt";
		fstream fout(output_filename.c_str(), ios::out);
		for( int j = 0; j < 308; j++)
			fout<<vfhs->points[0].histogram[j]<<endl;
		for( int j = 0; j < 16; j++)
			fout<<histogram[j]<<endl;
		


		fout.close();


		// image display 
		/*
		cvShowImage("image", image);
		cvShowImage("head", head);
		cvWaitKey();
		*/
		// release memories
		cvReleaseImage(&image);
		cvReleaseImage(&label);
		cvReleaseImage(&head);
	}





/*

	
	// Normal Estimation
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud (cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	ne.setSearchMethod (tree);
	// Use all neighbors in a sphere of radius 3cm
	ne.setRadiusSearch (0.03);
	ne.compute (*normals);

	// VFH
	pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud (cloud);
	vfh.setInputNormals (normals);
	vfh.setSearchMethod (tree);
	pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
	vfh.compute (*vfhs);

	// Histogram plot
	hist.setBackgroundColor( 1.0, 1.0, 1.0);
	hist.addFeatureHistogram( *vfhs, 308);
	hist.spin();
*/

    	return 0;
}
