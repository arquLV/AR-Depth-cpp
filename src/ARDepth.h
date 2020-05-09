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
    ARDepth(const std::string& input_frames, const std::string& input_colmap, const std::string& input_scenes, const std::string& input_edges, const bool& resize, const bool& visualize, const bool& edgesOnly, const bool& precompEdges)
            :input_frames(input_frames),
             input_colmap(input_colmap),
			 input_scenes(input_scenes),
			 input_edges(input_edges),
             resize(resize),
             visualize(visualize), 
			 edgesOnly(edgesOnly),
			 precompEdges(precompEdges) {};

    ~ARDepth() = default;

	const int width_visualize = 720;
	const int height_visualize = 1280;

    const std::string input_frames;
	const std::string input_scenes;
    const std::string input_colmap;
	const std::string input_edges;
	const bool edgesOnly;
	const bool precompEdges;

    const bool resize;
    const bool visualize;
    const double tau_high = 0.1;
    const double tau_low = 0.07;
    const double tau_flow = 0.1;
    const int k_I = 5;
    const int k_T = 7;
    /*const int k_F = 31;*/
	const int k_F = 31;
    const double lambda_d = 1.1;
    const double lambda_t = 0.01;
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

	void visualizeImg(const cv::Mat& soft_edges, const cv::Mat& edges, const int frameNum);
    void visualizeImg(const cv::Mat& raw_img, const cv::Mat& scene_img, const cv::Mat& raw_depth, const cv::Mat& filtered_depth, const cv::Mat& soft_edges, const cv::Mat& edges, const double objectDepth, const int frameNum);

    cv::Mat GetInitialization(const cv::Mat& sparse_points, const cv::Mat& last_depth_map);

    cv::Mat DensifyFrame(const cv::Mat& sparse_points, const cv::Mat& confidence_map, const cv::Mat& hard_edges, const cv::Mat& soft_edges, const cv::Mat& last_depth_map);

    template <typename T>
    T median(std::vector<T>& c);

    cv::Mat TemporalMedian(const std::deque<cv::Mat>& depth_maps);

    void run();

};