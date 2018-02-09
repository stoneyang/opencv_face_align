#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <map>
#include <math.h>
#include <sys/stat.h>

#include "opencv_face_align.h"

inline bool exists(const std::string& fn)
{
	struct stat buffer;
	return (stat (fn.c_str(), &buffer) == 0);
}

inline void string2pt(const std::string& pt_anno_pair, int& x_coord, int& y_coord)
{
	std::istringstream iss(pt_anno_pair);
	iss >> x_coord >> y_coord;
	//std::cout << x_coord << ", " << y_coord << std::endl;
}

inline int guard(int prob, const int lower, const int upper)
{
	if (lower <= prob && upper >= prob) {
		return prob;
	}
	else if (lower > prob) {
		return prob = lower;
	}
	else if (upper < prob) {
		return prob = upper;
	}
}

int main(int argc, char** argv)
{
	if (argc != 6) {
		std::cerr << "Incorrect command line input! " << std::endl;
		std::cerr << "Usage: opencv_face_align facial_patch 5pt_anno rot_patch rescale_patch crop_patch" << std::endl;
		return EXIT_FAILURE;
	}

	// command line parser
	std::string facial_patch_fn = argv[1];
	std::string anno_fn         = argv[2];
	std::string rot_fn          = argv[3];
	std::string rescale_fn      = argv[4];
	std::string crop_fn         = argv[5];

	// standard values
	const int ec_mc_y_const   = 48;
	const int ec_y_const      = 48;
	const int crop_size_const = 128;
	const double max_rot_angle = 30;
	const double min_rot_angle = -30;
	const double max_diff_angle = 20;
	const double min_diff_angle = -20;

	// facial_patch
	Mat src = imread(facial_patch_fn, CV_LOAD_IMAGE_UNCHANGED);
	Mat dst;
	Mat rot;
	Mat rescale;
	Mat crop;

	// 5pt_anno
	if (!exists(anno_fn)) {
		std::cout << "pt anno file: " << anno_fn << " does not exist! " << std::endl;
		std::cout << "Center crop will be applied! " << std::endl;
		center_crop(src, dst);
		imwrite(crop_fn, dst);
	}
	else {
		std::ifstream pt_anno_stream(anno_fn);
		std::string cur_line;
		std::vector<std::string> pt_anno;
		while (std::getline(pt_anno_stream, cur_line)) {
			std::istringstream iss(cur_line);
			pt_anno.push_back(cur_line);
		}

		int pt_anno_coords[10];
		for (int i = 0; i < 5; ++i) {
			std::string cur_pt_anno = pt_anno.at(i + 1);
			string2pt(cur_pt_anno, pt_anno_coords[2 * i], pt_anno_coords[2 * i + 1]);
		}

		// img rotation
		/* to wrap as a function*/
		double rot_angle_tan = (double)(pt_anno_coords[3] - pt_anno_coords[1]) / (double)(pt_anno_coords[2] - pt_anno_coords[0]);
		double rot_angle = atan(rot_angle_tan) / PI * 180;
		std::cout << "tangent of rotation angle: " << rot_angle_tan << std::endl;
		std::cout << "rotation angle: " << rot_angle << std::endl;
		/* to wrap as a function*/

		double m_angle_tan = (double)(pt_anno_coords[9] - pt_anno_coords[7]) / (double)(pt_anno_coords[8] - pt_anno_coords[6]);
		double m_angle = atan(m_angle_tan) / PI * 180;
		double diff_angle = rot_angle - m_angle;
		if (rot_angle >= max_rot_angle || rot_angle <= min_rot_angle) {
			std::cout << "rot_angle is " << rot_angle << " does not lie in the accepted range (" << min_rot_angle << ", " << max_rot_angle << "). " << std::endl;
			return EXIT_FAILURE;
		} else if (diff_angle >= max_diff_angle || diff_angle <= min_diff_angle) {
			std::cout << "diff_angle is " << diff_angle << " does not lie in the accepted range (" << min_diff_angle << ", " << max_diff_angle << "). " << std::endl;
			return EXIT_FAILURE;
		}
		rotation(src, rot, rot_angle);
		std::cout << "size of input img: " << src.size() << std::endl;
		std::cout << "size of rotated img: " << rot.size() << std::endl;
		
		imwrite(rot_fn, rot);

		// eye center
		/* to wrap as a function*/
		int ec_x = (pt_anno_coords[2] + pt_anno_coords[0]) / 2;
		int ec_y = (pt_anno_coords[3] + pt_anno_coords[1]) / 2;
		std::cout << "eye center in original img: " << ec_x << ", " << ec_y << std::endl;
		/* to wrap as a function*/

		// eye center in rot
		float ec_x_rescale_d, ec_y_rescale_d;
		rescale_transform(ec_x, ec_y, rot_angle, src, rot, ec_x_rescale_d, ec_y_rescale_d);
		int ec_x_rescale = cvRound(ec_x_rescale_d);
		int ec_y_rescale = cvRound(ec_y_rescale_d);
		std::cout << "eye center in rotated img: " << ec_x_rescale << ", " << ec_y_rescale << std::endl;

		// mouth center
		/* to wrap as a function*/
		int mc_x = (pt_anno_coords[8] + pt_anno_coords[6]) / 2;
		int mc_y = (pt_anno_coords[9] + pt_anno_coords[7]) / 2;
		std::cout << "mouth center in original img: " << mc_x << ", " << mc_y << std::endl;
		/* to wrap as a function*/

		// mouth center in rot
		float mc_x_rescale_d, mc_y_rescale_d;
		rescale_transform(mc_x, mc_y, rot_angle, src, rot, mc_x_rescale_d, mc_y_rescale_d);
		int mc_x_rescale = cvRound(mc_x_rescale_d);
		int mc_y_rescale = cvRound(mc_y_rescale_d);
		std::cout << "mouth center in rotated img: " << mc_x_rescale << ", " << mc_y_rescale << std::endl;

		// resize rescale
		float resize_scale = ec_mc_y_const / (float)(mc_y_rescale - ec_y_rescale);
		std::cout << "resizing scale: " << resize_scale << std::endl;

		// img resize
		resize(rot, rescale, Size(), resize_scale, resize_scale);
		imwrite(rescale_fn, rescale);
		std::cout << "size of rescaled img: " << rescale.size() << std::endl;

		// img crop
		int ec_x_final = cvRound((ec_x_rescale - rot.cols / 2) * resize_scale + rescale.cols / 2);
		int ec_y_final = cvRound((ec_y_rescale - rot.rows / 2) * resize_scale + rescale.rows / 2);

		std::cout << "eye center in final (cropped) img: " << ec_x_final << ", " << ec_y_final << std::endl;

		int crop_x = ec_x_final - crop_size_const / 2;
		int crop_y = ec_y_final - ec_y_const;
		int crop_x_end = crop_x + crop_size_const;
		int crop_y_end = crop_y + crop_size_const;

		std::cout << "cropping region: " << crop_x << ", " << crop_y << ", " << crop_x_end << ", " << crop_y_end << std::endl;

		int starting_pt = 0;
		int crop_x_guard;
		int crop_y_guard;
		int crop_x_end_guard;
		int crop_y_end_guard;
		crop_x_guard = guard(crop_x, starting_pt, rescale.cols);
		crop_y_guard = guard(crop_y, starting_pt, rescale.rows);
		crop_x_end_guard = guard(crop_x_end, starting_pt, rescale.cols);
		crop_y_end_guard = guard(crop_y_end, starting_pt, rescale.rows);

		std::cout << "guarded region: " << crop_x_guard << ", " << crop_y_guard << ", " << crop_x_end_guard << ", " << crop_y_end_guard << std::endl;

		int crop_size_guard = min(crop_x_end_guard - crop_x_guard, crop_y_end_guard - crop_y_guard);
		std::cout << "crop_size_guard: " << crop_size_guard << std::endl;

		if (crop_size_guard < 128) {
			std::cout << "crop size is smaller than 128, take another photo" << std::endl;
			return EXIT_FAILURE;
		}

		crop_transform(rescale, crop, crop_x_guard, crop_y_guard, crop_size_guard);
		imwrite(crop_fn, crop);
	}
	
	return EXIT_SUCCESS;
}