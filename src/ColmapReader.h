#pragma once
#include <fstream>
#include "util.h"

class ColmapReader {
public:
    size_t split(const std::string &txt, std::vector<std::string> &strs, char ch);

    std::map<int, Camera> ReadColmapCamera(std::string filename);

    std::map<int, View> ReadColmapImages(std::string filename);

    std::map<int, Point> ReadColmapPoints(std::string filename);

	std::pair<std::map<int, Point>, std::map<int, Point>> ReadObjectData(std::string filename);

    Reconstruction ReadColmap(const std::string &poses_folder, const std::string &images_folder, const std::string &scenes_folder, const std::string& edges_folder);
};