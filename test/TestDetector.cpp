#include "Detector.h"

int main(int argc, char *argv[])
{
    std::string names_file = "../cfg/coco.names";
    std::string cfg_file = "../cfg/yolov4.cfg";
//    std::string  cfg_file = "../cfg/yolov3.cfg";
    std::string  weights_file = "../model/yolov4.weights";
//    std::string  weights_file = "../model/yolov3.weights";
    std::string filename = "../test/image/dishes.png";

    Detector detector(cfg_file, weights_file);

    std::vector<std::string> obj_names;
    obj_names = detector.GetObjectNamesFromFile(names_file);

    // image file
    // to achive high performance for multiple images do these 2 lines in another thread
    cv::Mat mat_img = cv::imread(filename);
//        auto det_image = detector.mat_to_image_resize(mat_img);

    auto start = std::chrono::steady_clock::now();
    std::vector<Object> result_vec = detector.Detect(mat_img);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> spent = end - start;
    std::cout << " Time: " << spent.count() << " secs \n";

    //result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
    detector.DrawBoxes(mat_img, result_vec, obj_names);
    cv::imshow("window name", mat_img);
//        std::vector<std::string> filenamesplit=split(filename,'/');
//        std::string endname=filenamesplit[filenamesplit.size()-1];
//        endname.replace(endname.end()-4,endname.end(),"_yolov4_out.jpg");
//        std::string outputfile="../detect_result/"+endname;
//        imwrite(outputfile, mat_img);
    detector.ShowConsoleResult(result_vec, obj_names);
    cv::waitKey(0);

//    filename.clear();

    return 0;
}
