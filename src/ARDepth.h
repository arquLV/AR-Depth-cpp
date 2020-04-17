#pragma once
#include <iostream>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/core.hpp>

#include "Eigen/Geometry"
#include "Eigen/Sparse"
#include <fstream>
#include <map>
#include <queue>
#include "util.h"
#include "ColmapReader.h"

class ARDepth{
public:
    ARDepth(const std::string& input_frames, const std::string& input_colmap, const bool& resize, const bool& visualize, const bool& edgesOnly)
            :input_frames(input_frames),
             input_colmap(input_colmap),
             resize(resize),
             visualize(visualize), 
			 edgesOnly(edgesOnly) {};

    ~ARDepth() = default;

	const int width_visualize = 720;
	const int height_visualize = 1280;

    const std::string input_frames;
    const std::string input_colmap;
	const bool edgesOnly;
    const bool resize;
    const bool visualize;
    const double tau_high = 0.784;
    const double tau_low = 0.196;
    const double tau_flow = 0.06;
    const int k_I = 5;
    const int k_T = 7;
    /*const int k_F = 31;*/
	const int k_F = 31;
    const double lambda_d = 1.3;
    const double lambda_t = 0.015;
    const double lambda_s = 1;
    const int num_solver_iterations = 500;
    const cv::Ptr<cv::DenseOpticalFlow> dis = cv::optflow::createOptFlow_DIS(2);

	cv::Mat AbsoluteMaximum(const std::vector<cv::Mat>& mats);

    cv::Mat GetFlow(const cv::Mat& image1, const cv::Mat& image2);

    std::pair<cv::Mat, cv::Mat> GetImageGradient(const cv::Mat& image);

    cv::Mat GetGradientMagnitude(const cv::Mat& img_grad_x, const cv::Mat& img_grad_y);

    std::pair<cv::Mat, cv::Mat> GetFlowGradientMagnitude(const cv::Mat& flow, const cv::Mat& img_grad_x, const cv::Mat& img_grad_y);

    cv::Mat GetSoftEdges(const cv::Mat& image, const std::vector<cv::Mat>& flows);

    cv::Mat Canny(const cv::Mat& soft_edges, const cv::Mat& image);

	void visualizeImg(const cv::Mat& raw_img, const cv::Mat& soft_edges, const cv::Mat& edges, const int frameNum);
    void visualizeImg(const cv::Mat& raw_img, const cv::Mat& raw_depth, const cv::Mat& filtered_depth, const cv::Mat& soft_edges, const cv::Mat& edges, const int frameNum);

    cv::Mat GetInitialization(const cv::Mat& sparse_points, const cv::Mat& last_depth_map);

    cv::Mat DensifyFrame(const std::vector<Reconstruction::SparseMap>& sparsePyramid, const cv::Mat& hard_edges, const cv::Mat& soft_edges, const cv::Mat& last_depth_map);

    template <typename T>
    T median(std::vector<T>& c);

    cv::Mat TemporalMedian(const std::deque<cv::Mat>& depth_maps);

    void run();

};