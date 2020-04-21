#include "src/ARDepth.h"

int main(int argc, char* argv[]) {

	bool edgesOnly = false;
	std::string dataDir = "";

	for (int arg = 0; arg < argc; arg++) {
		std::string argVal = argv[arg];
		if (argVal == "--edgesOnly") {
			std::cout << "Edges only on" << std::endl;
			edgesOnly = true;
		}
		else if (argVal.substr(0, 7) == "--data=") {
			dataDir = argVal.substr(7) + "/";
			std::cout << "Data dir: " << dataDir << std::endl;
		}
	}
	
    std::string input_frames = "data/" + dataDir + "frames";
	std::string input_scenes = "data/" + dataDir + "sceneImgs";
    std::string input_colmap = "data/" + dataDir + "reconstruction";
    bool resize = true;
    bool visualize = true;
    ARDepth ardepth(input_frames, input_colmap, input_scenes, resize, visualize, edgesOnly);
    ardepth.run();

    return 0;
}