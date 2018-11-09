#include "SR_api.h"
#include "tengine_c_api.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include "cpu_device.h"

class SRAPI::Engine{
public:
    Engine(const char* prototxt, const char* caffemodel, int i){
        sprintf(model_name,"model_%d", i);
        if (load_model(model_name, "caffe", prototxt, caffemodel) < 0){
            throw "loading model error.\n";
        }

        char graph_name[10];
        sprintf(graph_name, "graph_%d",i);
        graph = create_runtime_graph(graph_name, model_name, NULL);
        prerun_graph(graph);
        input_tensor = get_graph_input_tensor(graph, 0, 0);


        get_tensor_shape(input_tensor, pre_input_dim, 4);
        output_tensor = get_graph_output_tensor(graph, 0,0);
//        memset(pre_input_dim, 0, sizeof(pre_input_dim));

    }
    int pre_input_dim[4];
    graph_t graph;
    char model_name[100];
    tensor_t input_tensor;
    tensor_t output_tensor;

    ~Engine(){
        put_graph_tensor(output_tensor);
        put_graph_tensor(input_tensor);
        postrun_graph(graph);
        destroy_runtime_graph(graph);
        remove_model(model_name);
    }
};

SRAPI::SRAPI(){
    n_net=0;
}

SRAPI::~SRAPI(){}

void SRAPI::set_cpu_list(const int* cpu_list, int cpu_number){
    set_working_cpu(cpu_list, cpu_number);
}

void SRAPI::addNetwork(const char* prototxt, const char* caffemodel){
    if (n_net == 0) {
        int error = init_tengine_library();
        if (error) {
            printf("error during init_tengine_library: %d\n", error);
            throw "init_tengine_library error.\n";
        }
    }
    std::string model_name(caffemodel);
    std::shared_ptr<Engine> engine(new Engine(prototxt, caffemodel, n_net++));
    engineMap[model_name] = engine;
}

cv::Mat SRAPI::run(const cv::Mat& inputImage, int width, int height, const char* caffemodel){
    std::string model_name(caffemodel);
    std::shared_ptr<Engine> engine = engineMap[model_name];
    graph_t graph=engine->graph;
    tensor_t input_tensor = engine->input_tensor;
    tensor_t output_tensor = engine->output_tensor;
    int* pre_input_dim = engine->pre_input_dim;


    int dims[] = {1, 1, height, width};
    bool IsPrerun = false;
    for(int i = 0; i < 4; ++i) {
        if (dims[i] != pre_input_dim[i]) IsPrerun = true;
        pre_input_dim[i] = dims[i];
    }

    if(IsPrerun){
        postrun_graph(graph);
        set_tensor_shape(input_tensor, dims, 4);
        prerun_graph(graph);
    }
    set_tensor_shape(input_tensor, dims, 4);
    set_tensor_buffer(input_tensor, inputImage.data, width * height * 4);
    prerun_graph(graph);
    // output
    int out_dim[4] = {0};
    get_tensor_shape(output_tensor, out_dim, 4);
    void* output = get_tensor_buffer(output_tensor);
    run_graph(graph, 1);
    cv::Mat outImage(cv::Size(out_dim[3], out_dim[2]), CV_32FC1, output);

    return outImage;
}


