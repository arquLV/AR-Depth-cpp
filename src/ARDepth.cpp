#include "ARDepth.h"

cv::Mat ARDepth::AbsoluteMaximum(const std::vector<cv::Mat>& mats) {
	cv::Mat output = cv::Mat::zeros(mats.at(0).size(), mats.at(0).depth());
	int height = output.rows;
	int width = output.cols;

	for (const auto& mat : mats) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (abs(mat.at<double>(y, x)) > abs(output.at<double>(y, x))) {
					output.at<double>(y, x) = mat.at<double>(y, x);
				}
			}
		}
	}

	return output;
}

cv::Mat ARDepth::GetFlow(const cv::Mat& image1, const cv::Mat& image2){
    cv::Mat flow, image1_, image2_;
    cv::cvtColor(image1, image1_, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, image2_, cv::COLOR_BGR2GRAY);
    dis->calc(image1_, image2_, flow);

    return flow;
}

std::pair<cv::Mat, cv::Mat> ARDepth::GetImageGradient(const cv::Mat& image){
    cv::Mat grad_x, grad_y;
    std::vector<cv::Mat> rgb_x, rgb_y;

    cv::Sobel(image, grad_x, CV_64F, 1, 0, 5);
    cv::Sobel(image, grad_y, CV_64F, 0, 1, 5);

    cv::split(grad_x, rgb_x);
    cv::split(grad_y, rgb_y);

    /*cv::Mat img_grad_x = cv::max(cv::max(rgb_x[0], rgb_x[1]), rgb_x[2]);
    cv::Mat img_grad_y = cv::max(cv::max(rgb_y[0], rgb_y[1]), rgb_y[2]);*/
	cv::Mat img_grad_x = AbsoluteMaximum(rgb_x);
	cv::Mat img_grad_y = AbsoluteMaximum(rgb_y);

    return std::make_pair(img_grad_x, img_grad_y);
}

cv::Mat ARDepth::GetGradientMagnitude(const cv::Mat& img_grad_x, const cv::Mat& img_grad_y){
    cv::Mat img_grad_magnitude;
    cv::sqrt(img_grad_x.mul(img_grad_x) + img_grad_y.mul(img_grad_y), img_grad_magnitude);

    return img_grad_magnitude;
};

std::pair<cv::Mat, cv::Mat> ARDepth::GetFlowGradientMagnitude(const cv::Mat& flow, const cv::Mat& img_grad_x, const cv::Mat& img_grad_y){
    std::vector<cv::Mat> grad_x, grad_y;
    cv::Mat tmp_x, tmp_y;
    cv::Sobel(flow, tmp_x, CV_64F, 1, 0, 5);
    cv::Sobel(flow, tmp_y, CV_64F, 0, 1, 5);
    cv::split(tmp_x, grad_x);
    cv::split(tmp_y, grad_y);

    /*cv::Mat flow_grad_x = cv::max(grad_x[0], grad_x[1]);
    cv::Mat flow_grad_y = cv::max(grad_y[0], grad_y[1]);*/
	cv::Mat flow_grad_x = AbsoluteMaximum(grad_x);
	cv::Mat flow_grad_y = AbsoluteMaximum(grad_y);

    cv::Mat flow_grad_magnitude;
    cv::sqrt(flow_grad_x.mul(flow_grad_x) + flow_grad_y.mul(flow_grad_y), flow_grad_magnitude);

    cv::Mat reliability = cv::Mat::zeros(flow.size(), flow.depth());

    int height = img_grad_x.rows;
    int width = img_grad_x.cols;

    for(int y=0; y<height; y++) {
        for(int x=1; x<width; x++){
            Eigen::Vector2d gradient_dir(img_grad_y.at<double>(y,x), (img_grad_x.at<double>(y,x)));
            if(gradient_dir.norm()==0){
                reliability.at<float>(y,x) = 0;
                continue;
            }
            gradient_dir /= gradient_dir.norm();
            Eigen::Vector2d center_pixel(y, x);
            Eigen::Vector2d p0 = center_pixel + gradient_dir;
            Eigen::Vector2d p1 = center_pixel - gradient_dir;

            if(p0[0]<0 || p1[0]<0 || p0[1]<0 || p1[1]<0
               || p0[0]>=height || p0[1]>=width || p1[0]>=height || p1[1]>=width){
                reliability.at<float>(y,x) = -1000;
                continue;
            }

            Eigen::Vector2d flow_p0(flow.at<cv::Vec2f>(int(p0[0]),int(p0[1]))[0], flow.at<cv::Vec2f>(int(p0[0]),int(p0[1]))[1]);
            Eigen::Vector2d flow_p1(flow.at<cv::Vec2f>(int(p1[0]),int(p1[1]))[0], flow.at<cv::Vec2f>(int(p1[0]),int(p1[1]))[1]);

            double f0 = flow_p0.dot(gradient_dir);
            double f1 = flow_p1.dot(gradient_dir);
            reliability.at<float>(y,x) = f1-f0;
        }
    }

    return std::make_pair(flow_grad_magnitude, reliability);
}

cv::Mat ARDepth::GetSoftEdges(const cv::Mat& image, const std::vector<cv::Mat>& flows){
    std::pair<cv::Mat, cv::Mat> img_grad = GetImageGradient(image);
    cv::Mat img_grad_magnitude = GetGradientMagnitude(img_grad.first, img_grad.second);

    cv::Mat flow_gradient_magnitude= cv::Mat::zeros(img_grad_magnitude.size(), img_grad_magnitude.depth());
    cv::Mat max_reliability = cv::Mat::zeros(img_grad_magnitude.size(), img_grad_magnitude.depth());


    int height = flows[0].rows;
    int width = flows[0].cols;

	std::cout << "Iterating flows" << std::endl;
    for(const auto& flow : flows){
        std::pair<cv::Mat, cv::Mat> FlowGradMag = GetFlowGradientMagnitude(flow, img_grad.first, img_grad.second);
        cv::Mat magnitude = FlowGradMag.first;
        cv::Mat reliability = FlowGradMag.second;

		//std::cout << flow_gradient_magnitude.elemSize1() << std::endl;
        for(int y=0; y<height; y++) {
            for (int x = 0; x<width; x++) {
                if(reliability.at<float>(y,x)>max_reliability.at<double>(y,x)){
                    flow_gradient_magnitude.at<double>(y,x) = magnitude.at<double>(y,x);

					max_reliability.at<double>(y, x) = reliability.at<float>(y, x); // ?
                }
            }
        }
    }
	std::cout << "Done iterating flows" << std::endl;

    cv::GaussianBlur(flow_gradient_magnitude, flow_gradient_magnitude, cv::Size(k_F, k_F), 0);
    flow_gradient_magnitude = flow_gradient_magnitude.mul(img_grad_magnitude);
	//cv:add(flow_gradient_magnitude, flow_gradient_magnitude.mul(img_grad_magnitude), flow_gradient_magnitude);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(flow_gradient_magnitude, &minVal, &maxVal, &minLoc, &maxLoc);
    flow_gradient_magnitude /= (maxVal * 0.7);
	flow_gradient_magnitude = cv::min(flow_gradient_magnitude, 1.0);

    return flow_gradient_magnitude;
}

cv::Mat ARDepth::Canny(const cv::Mat& soft_edges, const cv::Mat& image){
    cv::GaussianBlur(image, image, cv::Size(k_I, k_I), 0);
    cv::Mat grad_x, grad_y;
    std::vector<cv::Mat> rgb_x, rgb_y;
    cv::Sobel(image, grad_x, CV_64F, 1, 0, 5);
    cv::Sobel(image, grad_y, CV_64F, 0, 1, 5);
    cv::split(grad_x, rgb_x);
    cv::split(grad_y, rgb_y);

    std::vector<cv::Mat> merge;
    /*cv::Mat gx = cv::max(cv::max(rgb_x[0], rgb_x[1]), rgb_x[2]);
    cv::Mat gy = cv::max(cv::max(rgb_y[0], rgb_y[1]), rgb_y[2]);*/
	cv::Mat gx = AbsoluteMaximum(rgb_x);
	cv::Mat gy = AbsoluteMaximum(rgb_y);

    merge.push_back(gx);
    merge.push_back(gy);


    cv::Mat img_gradient;
    cv::merge(merge, img_gradient);

    int TG22 = 13573;
    gx = gx * pow(2,15); //CV_64FC1
    gy = gy * pow(2,15);

    cv::Mat mag = GetGradientMagnitude(gx, gy);//CV_64FC1
    std::queue<std::pair<int, int>> seeds;
    cv::Mat edges = cv::Mat::zeros(soft_edges.size(), soft_edges.depth());

    int height = img_gradient.rows-1;
    int width = img_gradient.cols-1;

    for(int y=1; y<height; y++){
        for(int x=1; x<width; x++){
            long int ax = static_cast<long int>(abs(gx.at<double>(y,x)));
            long int ay = static_cast<long int>(abs(gy.at<double>(y,x)))<<15;
            long int tg22x = ax * TG22;
            long int tg67x = tg22x + (ax<<16);
            double m = mag.at<double>(y,x);

			if (ay < tg22x) {
				if (m > mag.at<double>(y-1, x) && m >= mag.at<double>(y+1, x)) {
					if (m > tau_high && soft_edges.at<double>(y, x) > tau_flow) {
						seeds.push(std::make_pair(y, x));
						edges.at<double>(y, x) = 255;
					}
					else if (m > tau_low) {
						edges.at<double>(y, x) = 1;
					}
				}
			}
			else if (ay > tg67x) {
				if (m > mag.at<double>(y, x+1) && m >= mag.at<double>(y, x-1)) {
					if (m > tau_high && soft_edges.at<double>(y, x) > tau_flow) {
						seeds.push(std::make_pair(y, x));
						edges.at<double>(y, x) = 255;
					}
					else if (m > tau_low) {
						edges.at<double>(y, x) = 1;
					}
				}
			}
			else if ((int(gx.at<double>(y, x)) ^ int(gy.at<double>(y, x))) < 0) {
				if (m > mag.at<double>(y + 1, x - 1) && m >= mag.at<double>(y - 1, x + 1)) {
					if (m > tau_high && soft_edges.at<double>(y, x) > tau_flow) {
						seeds.push(std::make_pair(y, x));
						edges.at<double>(y, x) = 255;
					}
					else if (m > tau_low) {
						edges.at<double>(y, x) = 1;
					}
				}
			}
			else {
				if (m > mag.at<double>(y - 1, x - 1) && m >= mag.at<double>(y + 1, x + 1)) {
					if (m > tau_high && soft_edges.at<double>(y, x) > tau_flow) {
						seeds.push(std::make_pair(y, x));
						edges.at<double>(y, x) = 255;
					}
					else if (m > tau_low) {
						edges.at<double>(y, x) = 1;
					}
				}
			}



            /*if(ay < tg22x){
                if( m > mag.at<double>(y,x-1) && m >= mag.at<double>(y,x+1)) {
                    if (m > tau_high && soft_edges.at<double>(y,x) > tau_flow){
                        seeds.push(std::make_pair(y,x));
                        edges.at<double>(y,x) =255;
                    }else if (m > tau_low){
                        edges.at<double>(y,x) = 1;
                    }
                }
            } else if (ay > tg67x){
                if(m > mag.at<double>(y+1,x) && m >= mag.at<double>(y-1,x)){
                    if (m > tau_high && soft_edges.at<double>(y,x) > tau_flow){
                        seeds.push(std::make_pair(y,x));
                        edges.at<double>(y,x) =255;
                    }else if (m > tau_low){
                        edges.at<double>(y,x) = 1;
                    }
                }
            } else if ( (int(gx.at<double>(y,x)) ^ int(gy.at<double>(y,x))) < 0){
                if(m > mag.at<double>(y-1,x+1) && m >= mag.at<double>(y+1,x-1)) {
                    if (m > tau_high && soft_edges.at<double>(y,x) > tau_flow){
                        seeds.push(std::make_pair(y,x));
                        edges.at<double>(y,x) =255;
                    }else if (m > tau_low){
                        edges.at<double>(y,x) = 1;
                    }
                }
            } else {
                if(m > mag.at<double>(y-1,x-1) && m >= mag.at<double>(y+1,x+1)){
                    if (m > tau_high && soft_edges.at<double>(y,x) > tau_flow){
                        seeds.push(std::make_pair(y,x));
                        edges.at<double>(y,x) =255;
                    }else if (m > tau_low){
                        edges.at<double>(y,x) = 1;
                    }
                }
            }*/
        }
    }

    while(!seeds.empty()){
        std::pair<int, int> seed = seeds.front();
        seeds.pop();

        int y = seed.first;
        int x = seed.second;
        if (x < width && y < height && edges.at<double>(y+1,x+1) == 1) {
            edges.at<double>(y+1,x+1) = 255;
            //seeds.push(std::make_pair(x + 1, y + 1));
			seeds.push(std::make_pair(y + 1, x + 1));
        }
        if (x > 0 && y < height && edges.at<double>(y+1,x-1) == 1) {
            edges.at<double>(y+1,x-1) = 255;
            /*seeds.push(std::make_pair(x+1, y-1));*/
			seeds.push(std::make_pair(y+1, x-1));
        }
        if (y < height && edges.at<double>(y+1, x) == 1) {
            edges.at<double>(y+1, x) = 255;
            //seeds.push(std::make_pair(x+1, y));
			seeds.push(std::make_pair(y+1, x));
        }
        if (x < width && y > 0 && edges.at<double>(y-1,x+1) == 1) {
            edges.at<double>(y-1,x+1) = 255;
            //seeds.push(std::make_pair(x-1,y+1));
			seeds.push(std::make_pair(y-1, x+1));
        }
        if (x > 0 && y > 0 && edges.at<double>(y-1,x-1) == 1) {
            edges.at<double>(y-1,x-1) = 255;
            //seeds.push(std::make_pair(x-1,y-1));
			seeds.push(std::make_pair(y - 1, x - 1));
        }
        if (y > 0 && edges.at<double>(y-1, x) == 1) {
            edges.at<double>(y-1, x) = 255;
            //seeds.push(std::make_pair(x-1, y));
			seeds.push(std::make_pair(y-1, x));
        }
        if (x < width && edges.at<double>(y, x+1) == 1) {
            edges.at<double>(y, x+1) = 255;
            //seeds.push(std::make_pair(x, y+1));
			seeds.push(std::make_pair(y, x+1));
        }
        if (x > 0 && edges.at<double>(y, x-1) == 1) {
            edges.at<double>(y, x-1) = 255;
            //seeds.push(std::make_pair(x, y-1));
			seeds.push(std::make_pair(y, x-1));
        }

    }

    for(int y=1; y<height; y++){
        for(int x=1; x<width; x++){
            if(edges.at<double>(y,x)==1) edges.at<double>(y,x)=0;
        }
    }
    return edges;
}

cv::Mat ARDepth::GetInitialization(const cv::Mat& sparse_points, const cv::Mat& last_depth_map){
    cv::Mat initialization = sparse_points.clone();
    if(!last_depth_map.empty()){
        cv::Mat inv = 1.0/last_depth_map;
        inv.copyTo(initialization, last_depth_map>0);
    }

    int h = sparse_points.rows;
    int w = sparse_points.cols;
    double last_known = -1;
    double first_known = -1;

    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(sparse_points, &min, &max, &min_loc, &max_loc);

    for(int y=0; y< h; y++){
        for(int x=0; x< w; x++){
            if(sparse_points.at<double>(y,x) > 0){
                last_known = 1.0/sparse_points.at<double>(y,x);
            }else if(initialization.at<double>(y,x) > 0){
                last_known = initialization.at<double>(y,x);
            }
            if(first_known < 0){
                first_known = last_known;
            }
            initialization.at<double>(y,x) = last_known;
        }
    }

    cv::Mat first_known_mat = cv::Mat::ones(h, w, initialization.type())*first_known;
    cv::Mat mask = initialization<0;
    first_known_mat.copyTo(initialization, mask);

    return initialization;
}

cv::Mat ARDepth::DensifyFramePyramid(const std::vector<Reconstruction::SparseMap>& sparsePyramid, const cv::Mat& hard_edges, const cv::Mat& soft_edges, const cv::Mat& last_depth_map){
	
	std::vector<cv::Mat> hard_edges_pyramid;
	hard_edges_pyramid.emplace_back(hard_edges);

	std::vector<cv::Mat> soft_edges_pyramid;
	soft_edges_pyramid.emplace_back(soft_edges);

	std::vector<cv::Mat> last_depth_pyramid;
	if (!last_depth_map.empty()) {
		last_depth_pyramid.emplace_back(last_depth_map);
	}

	//Layer 0 = base size
	//Layer 3 = smallest
	for (int l = 1; l < 4; l++) {
		int w = sparsePyramid.at(l).first.cols;
		int h = sparsePyramid.at(l).first.rows;

		cv::Mat hard_edges_resized;
		cv::Mat soft_edges_resized;
		cv::resize(hard_edges, hard_edges_resized, cv::Size(w, h));
		cv::resize(soft_edges, soft_edges_resized, cv::Size(w, h));

		hard_edges_pyramid.emplace_back(hard_edges_resized);
		soft_edges_pyramid.emplace_back(soft_edges_resized);

		if (!last_depth_map.empty()) {
			cv::Mat last_depth_resized;
			cv::resize(last_depth_map, last_depth_resized, cv::Size(w, h));
			last_depth_pyramid.emplace_back(last_depth_resized);
		}
	}
	cv::Mat last_depth_small;
	if (!last_depth_map.empty()) {
		cv::resize(last_depth_map, last_depth_small, cv::Size(sparsePyramid.at(3).first.cols, sparsePyramid.at(3).first.rows));
	}
	cv::Mat topInitialization = GetInitialization(sparsePyramid.at(3).first, last_depth_small);
	Eigen::VectorXd x_vec;
	bool isTop = true;

	for (int iter = 0; iter < 5; iter++) {
		for (int layer = 3; layer >= 0; layer--) {
			int w = sparsePyramid.at(layer).first.cols;
			int h = sparsePyramid.at(layer).first.rows;
			int num_pixels = w * h;


			Eigen::SparseMatrix<double> A(num_pixels * 3, num_pixels);
			Eigen::VectorXd b = Eigen::VectorXd::Zero(num_pixels * 3);
			Eigen::VectorXd x0 = Eigen::VectorXd::Zero(num_pixels);
			int num_entries = 0;

			cv::Mat smoothness = cv::max(1 - soft_edges_pyramid.at(layer), 0);
			cv::Mat smoothness_x = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);
			cv::Mat smoothness_y = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);

			std::vector<Eigen::Triplet<double>> tripletList;

			for (int y = 1; y < h - 1; y++) {
				for (int x = 1; x < w - 1; x++) {
					int idx = x + y * w;

					if (isTop) {
						x0(idx) = topInitialization.at<double>(y, x);
					}
					else {
						x0(idx) = x_vec(x + y * w);
					}
					
					if (sparsePyramid.at(layer).first.at<double>(y, x) > 0.00) {
						double confidence = sparsePyramid.at(layer).second.at<double>(y, x);
						tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx, lambda_d * confidence));
						b(num_entries) = (1.0 / sparsePyramid.at(layer).first.at<double>(y, x)) * (lambda_d * confidence);
						num_entries++;
					}
					else if (!last_depth_map.empty() && last_depth_pyramid.at(layer).at<double>(y, x) > 0) {
						tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx, lambda_t));
						b(num_entries) = (1.0 / last_depth_pyramid.at(layer).at<double>(y, x)) * lambda_t;
						num_entries++;
					}

					double smoothnes_weight = lambda_s * std::min(smoothness.at<double>(y, x), smoothness.at<double>(y - 1, x));

					if (hard_edges.at<double>(y, x) == hard_edges.at<double>(y - 1, x)) {
						smoothness_x.at<double>(y, x) = smoothnes_weight;
						tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx - w, smoothnes_weight));
						tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx, -smoothnes_weight));
						b(num_entries) = 0;
						num_entries++;
					}

					smoothnes_weight = lambda_s * std::min(smoothness.at<double>(y, x), smoothness.at<double>(y, x - 1));

					if (hard_edges.at<double>(y, x) == hard_edges.at<double>(y, x - 1)) {
						smoothness_y.at<double>(y, x) = smoothnes_weight;
						tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx - 1, smoothnes_weight));
						tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx, -smoothnes_weight));
						b(num_entries) = 0;
						num_entries++;
					}
				}
			}


			A.setFromTriplets(tripletList.begin(), tripletList.end());

			Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;

			cg.compute(A.transpose() * A);
			cg.setMaxIterations(2);
			cg.setTolerance(1e-05);
			x_vec = cg.solveWithGuess(A.transpose() * b, x0);

			isTop = false;
		}
	}
/*
    int w = sparse_points.cols;
    int h = sparse_points.rows;
    int num_pixels = w * h;*/

   /* Eigen::SparseMatrix<double> A(num_pixels * 3, num_pixels);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(num_pixels * 3);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(num_pixels);
    int num_entries = 0;

    cv::Mat smoothness = cv::max(1-soft_edges, 0);
    cv::Mat smoothness_x = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);
    cv::Mat smoothness_y = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);

    cv::Mat initialization = GetInitialization(sparse_points, last_depth_map);

    std::vector<Eigen::Triplet<double>> tripletList;

    for(int y=1; y<h-1; y++){
        for(int x=1; x<w-1; x++){
            int idx = x+y*w;
            x0(idx) = initialization.at<double>(y,x);
            if(sparse_points.at<double>(y,x)>0.00){
				double confidence = confidence_map.at<double>(y, x);
                tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx, lambda_d * confidence));
                b(num_entries) = (1.0 / sparse_points.at<double>(y,x)) * (lambda_d * confidence);
                num_entries++;
            }
            else if(!last_depth_map.empty() && last_depth_map.at<double>(y,x)>0){
                tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx, lambda_t));
                b(num_entries) = (1.0 / last_depth_map.at<double>(y,x)) * lambda_t;
                num_entries++;
            }

            double smoothnes_weight = lambda_s * std::min(smoothness.at<double>(y,x), smoothness.at<double>(y-1,x));

            if(hard_edges.at<double>(y,x) == hard_edges.at<double>(y-1,x)){
                smoothness_x.at<double>(y,x) = smoothnes_weight;
                tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx-w, smoothnes_weight));
                tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx, -smoothnes_weight));
                b(num_entries) = 0;
                num_entries++;
            }

            smoothnes_weight = lambda_s * std::min(smoothness.at<double>(y,x), smoothness.at<double>(y,x-1));

            if(hard_edges.at<double>(y,x) == hard_edges.at<double>(y,x-1)){
                smoothness_y.at<double>(y,x) = smoothnes_weight;
                tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx-1, smoothnes_weight));
                tripletList.emplace_back(Eigen::Triplet<double>(num_entries, idx, -smoothnes_weight));
                b(num_entries) = 0;
                num_entries++;
            }
        }
    }


    A.setFromTriplets(tripletList.begin(), tripletList.end());

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;

    cg.compute(A.transpose()*A);
    cg.setMaxIterations(num_solver_iterations);
    cg.setTolerance(1e-05);
    Eigen::VectorXd x_vec = cg.solveWithGuess(A.transpose()*b, x0);*/

	int w = sparsePyramid.at(0).first.cols;
	int h = sparsePyramid.at(0).first.rows;
    cv::Mat depth = cv::Mat::zeros(h,w,CV_64FC1);
    for(int y=0; y<h; y++)
        for(int x=0; x<w; x++)
            depth.at<double>(y,x) = 1.0/(x_vec(x+y*w)+1e-7);

    return depth;
}

template <typename T>
T ARDepth::median(std::vector<T>& c)
{
    size_t n = c.size() / 2;
    std::nth_element(c.begin(), c.begin() + n, c.end());
    return c[n];
}

cv::Mat ARDepth::TemporalMedian(const std::deque<cv::Mat>& depth_maps){
    cv::Mat depth_map = depth_maps.front().clone();
    int w = depth_map.cols;
    int h = depth_map.rows;

    for(int y=0; y<h; y++){
        for(int x=0; x<w; x++){
            std::vector<double> values;
            for (auto itr = depth_maps.cbegin(); itr!=depth_maps.cend(); ++itr){
                values.emplace_back((*itr).at<double>(y,x));
            }
            if(values.size()>0){
                depth_map.at<double>(y,x) = median(values);
            } else {
                depth_map.at<double>(y,x) = 0;
            }
        }
    }

    return depth_map;
}

void ARDepth::visualizeImg(const cv::Mat& raw_img, const cv::Mat& soft_edges, const cv::Mat& edges, const int frameNum) {
	//Color image
	cv::Mat color_visual;
	cv::resize(raw_img, color_visual, cv::Size(width_visualize, height_visualize), 0, 0, cv::INTER_AREA);

	//Soft edges
	cv::Mat soft_edges_visual = cv::Mat::zeros(soft_edges.size(), soft_edges.depth());
	//soft_edges.copyTo(soft_edges_visual);
	soft_edges.convertTo(soft_edges_visual, CV_8U, 255);

	//Canny edges
	cv::Mat canny_visual = cv::Mat::zeros(edges.size(), edges.depth());
	edges.convertTo(canny_visual, CV_8U, 255);

	std::cout << "Saving images..." << std::endl;
	cv::imwrite("output/color/" + std::to_string(frameNum) + ".jpg", color_visual);
	cv::imwrite("output/soft_edges/" + std::to_string(frameNum) + ".jpg", soft_edges_visual);
	cv::imwrite("output/canny/" + std::to_string(frameNum) + ".jpg", canny_visual);
	cv::waitKey(1);
}
void ARDepth::visualizeImg(const cv::Mat& raw_img, const cv::Mat& raw_depth, const cv::Mat& filtered_depth, const cv::Mat& soft_edges, const cv::Mat& edges, const int frameNum){

   
    //Sparse depth map
    cv::Mat raw_depth_visual = cv::Mat::zeros(raw_depth.size(), raw_depth.depth());
    cv::Mat tmp = cv::Mat::ones(raw_depth.size(), raw_depth.depth())*255;
    tmp.copyTo(raw_depth_visual, raw_depth>0);
    cv::resize(raw_depth_visual, raw_depth_visual, cv::Size(width_visualize, height_visualize), 0, 0, cv::INTER_AREA);

    //Dense depth map
    double min_val, max_val;
    cv::Mat filtered_depthmap_visual = cv::Mat::zeros(filtered_depth.size(), filtered_depth.depth());
    filtered_depth.copyTo(filtered_depthmap_visual, filtered_depth<100);
    cv::minMaxLoc(filtered_depthmap_visual, &min_val, &max_val);
    filtered_depthmap_visual = 255 * (filtered_depthmap_visual - min_val) / (max_val - min_val);
    filtered_depthmap_visual.convertTo(filtered_depthmap_visual, CV_8U);
    cv::applyColorMap(filtered_depthmap_visual, filtered_depthmap_visual, 2); //COLORMAP_JET
    cv::resize(filtered_depthmap_visual, filtered_depthmap_visual, cv::Size(width_visualize, height_visualize));

    //Visualize
    //cv::imshow("Color", color_visual);
    //cv::imshow("Sparse depth", raw_depth_visual);
    //cv::imshow("Dense depth", filtered_depthmap_visual);


	cv::imwrite("output/sparse/" + std::to_string(frameNum) + ".jpg", raw_depth_visual);
	cv::imwrite("output/dense/" + std::to_string(frameNum) + ".jpg", filtered_depthmap_visual);
	visualizeImg(raw_img, soft_edges, edges, frameNum);
}

void ARDepth::run() {
    ColmapReader reader;
    Reconstruction recon = reader.ReadColmap(input_colmap, input_frames);
    //int skip_frames = recon.GetNeighboringKeyframes(recon.GetNeighboringKeyframes(recon.ViewIds()[15]).first).second;
	int skip_frames = 0;

    std::deque<cv::Mat> last_depths;
    cv::Mat last_depth;

    std::cout<<"Using the first "<<skip_frames<<" frames to initialize (these won't be saved)."<<std::endl;
    int count = 0;
    for(const auto& frame : recon.ViewIds()){

        std::vector<int> reference_frames = recon.GetReferenceFrames(frame);
        if(reference_frames.empty())
            continue;

        std::cout<<"==> Processing frame "<<recon.views[frame].name<<std::endl;
        cv::Mat base_img_small = recon.GetImage(frame, true);
		cv::Mat base_img = recon.GetImage(frame, false);

        std::vector<cv::Mat> flows;
        for(const auto& ref : reference_frames){
            //cv::Mat ref_img = recon.GetImage(ref, resize);
			cv::Mat ref_img = recon.GetImage(ref, true);
            flows.emplace_back(GetFlow(base_img_small, ref_img));
        }
		std::cout << "Getting soft edges" << std::endl;
        cv::Mat soft_edges = GetSoftEdges(base_img_small, flows);

		cv::Mat soft_edges_resized;
		cv::resize(soft_edges, soft_edges_resized, cv::Size(base_img.cols, base_img.rows));

		cv::Mat soft_edges_thresholded;
		cv::threshold(soft_edges_resized, soft_edges_thresholded, tau_flow, 1, CV_THRESH_BINARY);

		std::cout << "Canny edges..." << std::endl;
        //cv::Mat edges = Canny(soft_edges_resized, base_img);
		
		cv::Mat edges;
        cv::Canny(base_img, edges, 50, 200);
        edges.convertTo(edges, CV_64FC1);

		edges = edges.mul(soft_edges_thresholded);


        int last_keyframe = frame;
        if(!recon.views[frame].isKeyframe()){
            std::pair<int,int> neiboring_keyframes = recon.GetNeighboringKeyframes(frame);
            assert(neiboring_keyframes.first!=-1 && neiboring_keyframes.second!=-1);
            last_keyframe = neiboring_keyframes.first;
        }

		if (edgesOnly == false) {
			
			//cv::resize(edges, edges_resized, cv::Size(width_resize, height_resize));

			std::cout << "Densifying..." << std::endl;
			//std::pair<cv::Mat, cv::Mat> sparseDepth = recon.GetSparseDepthWithSize(last_keyframe, width_resize, height_resize);
			//cv::Mat depth = DensifyFrame(recon.GetSparseDepthMap(last_keyframe, resize), edges, soft_edges_resized, last_depth);
			auto sparsePyramid = recon.GetSparseDepthPyramid(last_keyframe, base_img.cols, base_img.rows, 4);

			cv::Mat depth = DensifyFramePyramid(sparsePyramid, edges, soft_edges_resized, last_depth);

			last_depths.push_back(depth);
			if (last_depths.size() > k_T)
				last_depths.pop_front();

			std::cout << "Temporal median..." << std::endl;
			cv::Mat filtered_depth = TemporalMedian(last_depths);
			last_depth = depth;
		}
		

		if (edgesOnly) {
			cv::Mat raw_img = recon.GetImage(frame, false);
			visualizeImg(raw_img, soft_edges, edges, count);
		}
		else if(visualize) {
            cv::Mat raw_img = recon.GetImage(frame, false);
            cv::Mat raw_depth = recon.GetSparseDepthMap(last_keyframe, false);
            visualizeImg(raw_img, raw_depth, last_depth, soft_edges, edges, count);
        }
        count++;
    }
}