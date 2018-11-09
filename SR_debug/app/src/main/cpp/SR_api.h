#ifndef SR_API_
#define SR_API_

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class SRAPI{
public:
    SRAPI();
    ~SRAPI();
    void set_cpu_list(const int* cpu_list, int cpu_number);
    void addNetwork(const char* prototxt, const char* caffemodel);
    cv::Mat run(const cv::Mat& inputImage, int width, int height, const char* caffemodel);

private:
    int n_net;
    class Engine;
    std::map<std::string,std::shared_ptr<Engine>> engineMap;
};
#endif

