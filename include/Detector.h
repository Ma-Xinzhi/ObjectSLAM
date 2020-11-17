/*
* 2d目标检测
*/
#ifndef DETECTOR_H
#define DETECTOR_H

#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <deque>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>// gui 画图等
#include <opencv2/imgproc/imgproc.hpp>

#define LIB_API

// 检测结果类====
typedef struct Object
{
    cv::Rect_<float> rect;  // 边框
    float prob;             // 置信度
//    std::string object_name;// 物体类别名
    int object_id;           // 类别id
} Object;

struct bbox_t {
    unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;           // class of object - from range [0, classes-1]
    unsigned int track_id;         // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;   // counter of frames on which the object was detected
    float x_3d, y_3d, z_3d;        // center of object (in Meters) if ZED 3D Camera is used
};

struct image_t {
    int h;                        // height
    int w;                        // width
    int c;                        // number of chanels (3 - for RGB)
    float *data;                  // pointer to the image data
};

//extern "C" LIB_API int init(const char *configurationFilename, const char *weightsFilename, int gpu);
//extern "C" LIB_API int dispose();
//extern "C" LIB_API int get_device_count();
//extern "C" LIB_API int get_device_name(int gpu, char* deviceName);
//extern "C" LIB_API bool built_with_cuda();
//extern "C" LIB_API bool built_with_cudnn();
//extern "C" LIB_API bool built_with_opencv();

class Detector {
private:
    std::shared_ptr<void> detector_gpu_ptr;
    std::deque<std::vector<bbox_t>> prev_bbox_vec_deque;
    std::string _cfg_filename, _weight_filename;

public:
    LIB_API Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
    LIB_API ~Detector();

    LIB_API std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2, bool use_mean = false);
    LIB_API std::vector<bbox_t> detect(image_t img, float thresh = 0.2, bool use_mean = false);
    static LIB_API image_t load_image(std::string image_filename);
    static LIB_API void free_image(image_t m);
    LIB_API int get_net_width() const;
    LIB_API int get_net_height() const;
//    LIB_API int get_net_color_depth() const;

    std::vector<bbox_t> detect_resized(image_t img, int init_w, int init_h, float thresh = 0.2, bool use_mean = false);

    std::vector<Object> detect(cv::Mat mat, float thresh = 0.2, bool use_mean = false);

    std::shared_ptr<image_t> mat_to_image_resize(cv::Mat mat) const;

    static std::shared_ptr<image_t> mat_to_image(cv::Mat img_src);

    std::vector<std::string> objects_names_from_file(const std::string& filename);
    void draw_boxes(cv::Mat mat_img, const std::vector<Object>& result_vec, const std::vector<std::string>& obj_names);
    void show_console_result(const std::vector<Object>& result_vec, const std::vector<std::string>& obj_names, int frame_id = -1);

    cv::Scalar obj_id_to_color(int obj_id);

private:
    static image_t mat_to_image_custom(cv::Mat mat);

    static image_t make_empty_image(int w, int h, int c);

    static image_t make_image_custom(int w, int h, int c);

};
#endif