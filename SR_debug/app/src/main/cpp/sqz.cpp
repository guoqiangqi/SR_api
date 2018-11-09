#include <unistd.h>
#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio>
#include <jni.h>

#include "tengine_c_api.h"


int sqzmain()
{
    const char * text_file="/storage/emulated/0/SR_api/TCL_SR.prototxt";
    const char * model_file="/storage/emulated/0/SR_api/TCL_SR.caffemodel";
    const char * model_name="sqz";
    int input_h = 227;
    int input_w = 227;
    int input_size = input_h * input_w * 1;



    // 1. init tengine lib
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;


    // 2. load model
    if(load_model(model_name,"caffe",text_file,model_file)<0)
        return 1;
    std::cout << "Load model successfully\n";


    // 3. creat graph
    graph_t graph = create_runtime_graph("graph0", model_name, NULL);
    if (!check_graph_valid(graph))
    {
        std::cout << "create graph0 failed\n";
        return 1;
    }
    std::cout << "graph created\n";


    // 4. set input_shape, allocate input_data
    float  * input_data=(float*) malloc (sizeof(float) * input_size);
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor=get_graph_input_tensor(graph , node_idx , tensor_idx );
    int dims[]={1,1,input_h,input_w};
    set_tensor_shape(input_tensor,dims,4);
    if(set_tensor_buffer(input_tensor,input_data,input_size*sizeof(float))<0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }
    prerun_graph(graph);

    // 5. which output_data you want to take
    tensor_t output_tensor=get_graph_output_tensor(graph, node_idx, tensor_idx);
    int data_size=get_tensor_buffer_size(output_tensor)/sizeof(float);
    // tensor_t mytensor = get_graph_tensor(graph, "tensorname");
    float *  output_data=(float *)(get_tensor_buffer(output_tensor));

    // 6. run, each time change your input_data
    int repeat_count=5;
    for(int i=0;i<repeat_count;i++)
    {
        // change your input data here
        for(int k= 0;k<input_size;k++)
        {
            input_data[k]= k%64+i;
        }
        // run
        run_graph(graph,1);

        //get output_data
        printf("data_size = %d, out_data[0]=%f out_data[2]=%f\n",data_size,output_data[0],output_data[2]);
    }

    // 7. free
    postrun_graph(graph);
    free(input_data);
    put_graph_tensor(output_tensor);
    put_graph_tensor(input_tensor);
    destroy_runtime_graph(graph);
    remove_model(model_name);
    release_tengine_library();
    std::cout << "ALL TEST DONE\n";

    return 1;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_tcl_animoji_1cpp_MainActivity_ssdmainFromJNI(JNIEnv *env, jobject instance) {

    return sqzmain();

}