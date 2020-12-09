#include "Detector.h"

#include <cmath>
#include <iomanip>
#include <fstream>

#include <glog/logging.h>

std::vector<bbox_t> Detector::DetectResized(image_t img, int init_w, int init_h, float thresh, bool use_mean)
{
    if (img.data == NULL)
        throw std::runtime_error("Image is empty");
    auto detection_boxes = detect(img, thresh, use_mean);
    float wk = (float)init_w / img.w, hk = (float)init_h / img.h;
    for (auto &i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
    return detection_boxes;
}

Objects Detector::Detect(const cv::Mat& mat, const std::shared_ptr<KeyFrame>& pKF, float thresh, bool use_mean)
{
    if(mat.data == nullptr)
        throw std::runtime_error("Image is empty");
    auto image_ptr = mat_to_image_resize(mat);
    std::vector<bbox_t> bboxes = DetectResized(*image_ptr, mat.cols, mat.rows, thresh, use_mean);
    std::vector<Object> results;
    for(const bbox_t& item : bboxes){
        Object obj;
        obj.mBbox[0] = item.x;
        obj.mBbox[1] = item.y;
        obj.mBbox[2] = item.w;
        obj.mBbox[3] = item.h;
        obj.mProb = item.prob;
        obj.mObjectId = item.obj_id;
        obj.mpKeyFrame = pKF;
        results.push_back(obj);
    }
//    ShowConsoleResult(results);
    return results;
}

std::shared_ptr<image_t> Detector::mat_to_image_resize(const cv::Mat& mat) const
{
    if (mat.data == nullptr)
        return std::shared_ptr<image_t>(nullptr);

    cv::Size network_size = cv::Size(get_net_width(), get_net_height());
    cv::Mat det_mat;
    if (mat.size() != network_size)
        cv::resize(mat, det_mat, network_size);
    else
        det_mat = mat;  // only reference is copied

    return mat_to_image(det_mat);
}

std::shared_ptr<image_t> Detector::mat_to_image(const cv::Mat& img_src)
{
    cv::Mat img;
    if (img_src.channels() == 4) cv::cvtColor(img_src, img, cv::COLOR_RGBA2BGR);
    else if (img_src.channels() == 3) cv::cvtColor(img_src, img, cv::COLOR_RGB2BGR);
    else if (img_src.channels() == 1) cv::cvtColor(img_src, img, cv::COLOR_GRAY2BGR);
    else std::cerr << " Warning: img_src.channels() is not 1, 3 or 4. It is = " << img_src.channels() << std::endl;
    std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) { free_image(*img); delete img; });
    *image_ptr = mat_to_image_custom(img);
    return image_ptr;
}

cv::Scalar Detector::obj_id_to_color(int obj_id) {
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    int const offset = obj_id * 123457 % 6;
    int const color_scale = 150 + (obj_id * 123457) % 100;
    cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
    color *= color_scale;
    return color;
}

void Detector::DrawBoxes(cv::Mat mat_img, const Objects& result_vec, const std::vector<std::string>& obj_names)
{
    for (auto &i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.mObjectId);
        cv::Rect2f rect(i.mBbox[0], i.mBbox[1], i.mBbox[2], i.mBbox[3]);
        cv::rectangle(mat_img, rect, color, 1);
        std::string obj_name = obj_names[i.mObjectId];
        cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_PLAIN, 1.2, 2, nullptr);
        int max_width = (text_size.width > rect.width + 2) ? text_size.width : (rect.width + 2);
        max_width = std::max(max_width, (int)rect.width + 2);
        //max_width = std::max(max_width, 283);

//            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.rect.x - 1, 0), std::max((int)i.rect.y - 35, 0)),
//                          cv::Point2f(std::min((int)i.rect.x + max_width, mat_img.cols - 1), std::min((int)i.rect.y, mat_img.rows - 1)),
//                          color, 1, 8, 0);
        cv::putText(mat_img, obj_name, cv::Point2f(rect.x, rect.y - 6), cv::FONT_HERSHEY_PLAIN, 1.2, color, 2);
    }
}

void Detector::ShowConsoleResult(const Objects& result_vec, const std::vector<std::string>& obj_names) {
    for (auto &i : result_vec) {
        LOG(INFO) << obj_names[i.mObjectId] << " - "
                  << "Obj_id = " << i.mObjectId << ",  x = " << i.mBbox[0] << ", y = " << i.mBbox[1]
                  << ", w = " << i.mBbox[2] << ", h = " << i.mBbox[3]
                  << std::setprecision(3) << ", prob = " << i.mProb << std::endl;
    }
}

std::vector<std::string> Detector::GetObjectNamesFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open())
        return file_lines;

    for(std::string line; getline(file, line);)
        file_lines.push_back(line);
    LOG(INFO) << "Object names loaded \n";
    return file_lines;
}

image_t Detector::mat_to_image_custom(const cv::Mat& mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image_t im = make_image_custom(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}

image_t Detector::make_empty_image(int w, int h, int c)
{
    image_t out;
    out.data = nullptr;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image_t Detector::make_image_custom(int w, int h, int c)
{
    image_t out = make_empty_image(w, h, c);
    out.data = (float *)calloc(h*w*c, sizeof(float));
    return out;
}


