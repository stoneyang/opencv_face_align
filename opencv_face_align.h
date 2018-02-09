#ifndef OPENCV_FACE_ALIGN_H
#define OPENCV_FACE_ALIGN_H

#define PI 3.14159265

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

inline void center_crop(const Mat& src, Mat& dst, int crop_size = 128)
{
	int crop_x = (src.cols - crop_size) / 2;
	int crop_y = (src.rows - crop_size) / 2;
	Rect ROI(crop_x, crop_y, crop_size, crop_size);
	Mat croppedRef(src, ROI);
	Mat cropped;
	croppedRef.copyTo(cropped);
	dst = cropped;
}

inline void rotation(const Mat& src, Mat& dst, double rot_angle)
{
	Point2f rot_center(src.cols / 2.0, src.rows / 2.0);
	Mat rot_matrix = getRotationMatrix2D(rot_center, rot_angle, 1.0);
	// determine bounding rectangle
	Rect bbox = RotatedRect(rot_center, src.size(), rot_angle).boundingRect();
	// adjust transformation matrix
	rot_matrix.at<double>(0, 2) += bbox.width / 2.0 - rot_center.x;
	rot_matrix.at<double>(1, 2) += bbox.height / 2.0 - rot_center.y;
	warpAffine(src, dst, rot_matrix, bbox.size());
}

inline void rescale_transform(const int x, const int y, const double rot_angle, const Mat& src, const Mat& rot, float& x_out, float& y_out)
{
	double rot_angle_minus = -rot_angle / 180 * PI;
	double x0 = x - (double)src.cols / 2.;
	double y0 = y - (double)src.rows / 2.;
	x_out = x0 * cos(rot_angle_minus) - y0 * sin(rot_angle_minus) + (double)rot.cols / 2.;
	y_out = x0 * sin(rot_angle_minus) + y0 * cos(rot_angle_minus) + (double)rot.rows / 2.;
}

inline void crop_transform(const Mat& src, Mat& dst, const int& crop_x, const int& crop_y, int crop_size = 128)
{
	Rect ROI(crop_x, crop_y, crop_size, crop_size);
	Mat croppedRef(src, ROI);
	if (crop_size < 128) {
		std::cout << "crop size is smaller than 128" << std::endl;
	}
	Mat cropped;
	croppedRef.copyTo(cropped);
	dst = cropped;
}

#endif OPENCV_FACE_ALIGN_H // endif OPENCV_FACE_ALIGN
