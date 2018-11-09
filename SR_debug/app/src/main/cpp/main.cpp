#include <opencv2/opencv.hpp>
#include "SR_api.h"
#include <string>
#include <fstream>
#include <ostream>
#include <sys/time.h>
#include <chrono>
#include <stdlib.h>
#include "jni.h"

void write_image(const cv::Mat& im, const char * path){
    std::ofstream ofile(path, std::ofstream::binary);
    uchar* data = im.data;
    int rows = im.rows, cols = im.cols, channels = im.channels();
    ofile.write((char*)(&rows), sizeof(int));
    ofile.write((char*)(&cols), sizeof(int));
    ofile.write((char*)(&channels), sizeof(int));
    ofile.write((char*)(data), sizeof(float)*rows*cols*channels);
    ofile.close();
}

int main(int argc, char *argv[]){

    if (argc < 2){
        printf("[Usage]: %s proto_file model_file [cpu list, repeat_count]\n", argv[0]);
        printf("or\n");
        printf("[Usage]: %s file_list [cpu list, repeat_count]\n", argv[0]);
    }

    std::vector<std::string> prototxts;// = std::string(argv[1]);
    std::vector<std::string> caffemodels;// = std::string(argv[2]);
    int cpu_list[10];
    int cpu_num = 0;
    int repeat_count = 10;

    int idx = strlen(argv[1]) - 4;
    char ext[5];
    strncpy(ext, argv[1]+idx, 4);
    if (strcmp(ext, ".txt") == 0){
        printf("model_list: %s\n", argv[1]);
        if (argc < 2 || argc > 4){
            printf("[Usage]: %s proto_file model_file [cpu list, repeat_count]\n", argv[0]);
            printf("or\n");
            printf("[Usage]: %s file_list [cpu list, repeat_count]\n", argv[0]);
        }

        std::ifstream infile(argv[1]);

        std::string lineStr;
        while (true){
            if(!std::getline(infile,lineStr)) break;
            if (lineStr.length() < 4) continue;
            prototxts.push_back(lineStr);
            if(!std::getline(infile,lineStr)) {
                std::cout << "lineStr: " << lineStr << std::endl;
                std::cout << "file error 1: " << argv[1] << std::endl;
                throw "error code 1.\n";
            }
            if (lineStr.length() < 4) {
                std::cout << "lineStr: " << lineStr << std::endl;
                std::cout << "file error 2: " << argv[1] << std::endl;
                throw "error code 2.\n";
            }
            caffemodels.push_back(lineStr);
        }

        if (argc > 2) {
            std::cout << "cpu_list: " << argv[3] << std::endl;
            char* cpu_list_str = argv[3];
            char * p=strtok(cpu_list_str,",");
            while(p)
            {
                int cpu_id=strtoul(p,NULL,10);
                cpu_list[cpu_num++] = cpu_id;
                p=strtok(NULL,",");
            }
        }

        if(argc == 4){
            std::cout << "repeat_count: " << argv[4] << std::endl;
            repeat_count = atoi(argv[4]);
        }

    }
    else{
        prototxts.push_back(std::string(argv[1]));
        caffemodels.push_back(std::string(argv[2]));

        if (argc < 3 || argc > 5){
            printf("[Usage]: %s proto_file model_file [cpu list, repeat_count]\n", argv[0]);
            printf("or\n");
            printf("[Usage]: %s file_list [cpu list, repeat_count]\n", argv[0]);
        }

        if (argc > 3) {
            std::cout << "cpu_list: " << argv[3] << std::endl;
            char* cpu_list_str = argv[3];
            char * p=strtok(cpu_list_str,",");
            while(p)
            {
                int cpu_id=strtoul(p,NULL,10);
                cpu_list[cpu_num++] = cpu_id;
                p=strtok(NULL,",");
            }
        }

        if(argc == 5){
            std::cout << "repeat_count: " << argv[4] << std::endl;
            repeat_count = atoi(argv[4]);
        }
    }


    std::cout << std::endl << "......" << std::endl;
    std::chrono::high_resolution_clock::time_point t0, t1;

    SRAPI api;
    if(cpu_num > 0){
        api.set_cpu_list(cpu_list, cpu_num);
    }

    for (int nt = 0; nt < prototxts.size(); nt++){
        char img_path[50];
        char out_path[50];
        sprintf(img_path, "out_data/img_%d.bin", nt);
        sprintf(out_path, "out_data/out_%d.bin", nt);

        std::string prototxt = prototxts[nt];
        std::string caffemodel = caffemodels[nt];

        std::cout << "prototxt: " << prototxt << std::endl;
        std::cout << "caffemodel: " << caffemodel << std::endl;

        api.addNetwork(prototxt.c_str(),caffemodel.c_str());
        cv::Mat image_ = cv::imread("test.png",0);
        cv::Mat image;
        image_.convertTo(image, CV_32FC1, 1/255.0f);
        write_image(image, img_path);

        cv::Mat tmp_;
        cv::resize(image, tmp_, cv::Size(image.cols+1, image.rows));

        t0 = std::chrono::system_clock::now();
        for(int i =0; i < repeat_count; ++i) {
            api.run(tmp_, tmp_.cols, tmp_.rows, caffemodel.c_str());
            api.run(image, image.cols, tmp_.rows, caffemodel.c_str());
        }
        t1 = std::chrono::system_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count() * 1000;
        std::cout << "time1: " << dt/(repeat_count*2) << "ms" << std::endl;

        t0 = std::chrono::system_clock::now();
        for(int i = 0; i < repeat_count; ++i) api.run(image, image.cols, image.rows, caffemodel.c_str());
        t1 = std::chrono::system_clock::now();
        dt = std::chrono::duration<double>(t1 - t0).count() * 1000;
        std::cout << "time2: " << dt/repeat_count << "ms" << std::endl;

        cv::Mat out = api.run(image, image.cols, image.rows, caffemodel.c_str());
        write_image(out, out_path);
	std::cout << "img:   cols: " << out.cols << " rows: " << out.rows << std::endl;
        std::cout << "out:   cols: " << out.cols << " rows: " << out.rows << std::endl << std::endl << std::endl;
    }
}

int test(){
    std::string prototxt="/storage/emulated/0/SR_api/TCL_SR.prototxt";
    std::string caffemodel="/storage/emulated/0/SR_api/TCL_SR.caffemodel";
//    std::string prototxt="/sdcard/imgout/TCL_SR.prototxt";
//    std::string caffemodel="/sdcard/imgout/TCL_SR.caffemodel";

    std::string img_path = "/storage/emulated/0/SR_api/img.bin";
    std::string out_path = "/storage/emulated/0/SR_api/out.bin";

    SRAPI api;
    api.addNetwork(prototxt.c_str(),caffemodel.c_str());
    cv::Mat image_ = cv::imread("/storage/emulated/0/SR_api/test.png",0);
    cv::Mat image;
    image_.convertTo(image, CV_32FC1, 1/255.0f);
    write_image(image, img_path.c_str());
    cv::Mat out = api.run(image, image.cols, image.rows, caffemodel.c_str());
    write_image(out, out_path.c_str());

    return 1;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_tcl_animoji_1cpp_MainActivity_mainFromJNI(JNIEnv *env, jobject instance) {

   test();

}
