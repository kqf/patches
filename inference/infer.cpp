#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>

class Localizer {
public:
    Localizer(const std::string& onnx_file, int patch_size = 120)
        : size_(patch_size)
    {
        net_ = cv::dnn::readNetFromONNX(onnx_file);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT); // CPU
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);       // CPU
    }

    cv::Rect predict(const cv::Mat& frame, const cv::Rect& bbox) {
        // Crop ROI
        cv::Mat roi = frame(bbox).clone();

        // Preprocess: resize, convert to float, HWC -> CHW
        cv::Mat blob = cv::dnn::blobFromImage(
            roi, 1.0, cv::Size(size_, size_), cv::Scalar(), false, false, CV_32F
        );

        net_.setInput(blob);
        cv::Mat output = net_.forward(); // shape: [1,4]

        float* data = reinterpret_cast<float*>(output.data);
        cv::Rect2f local(data[0], data[1], data[2]-data[0], data[3]-data[1]);
        return toGlobal(local, bbox);
    }

private:
    int size_;
    cv::dnn::Net net_;

    // Map normalized local bbox -> absolute global coordinates
    cv::Rect toGlobal(const cv::Rect2f& local, const cv::Rect& patch) {
        int gx = static_cast<int>(patch.x + local.x * patch.width);
        int gy = static_cast<int>(patch.y + local.y * patch.height);
        int gw = static_cast<int>(local.width * patch.width);
        int gh = static_cast<int>(local.height * patch.height);
        return cv::Rect(gx, gy, gw, gh);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./run.bin <model.onnx>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    int patch_size = 120;

    cv::Mat frame = cv::imread("debug.jpg");
    if (frame.empty()) {
        std::cerr << "Failed to load image:" << std::endl;
        return 1;
    }

    std::cout <<  "Worked" << std::endl;
    // Example input bounding box (x, y, width, height)
    cv::Rect bbox(705, 154, 40, 40);

    Localizer localizer(model_path, patch_size);

    auto start = std::chrono::high_resolution_clock::now();
    cv::Rect pred_bbox = localizer.predict(frame, bbox);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Predicted bbox (global coordinates): " << pred_bbox << std::endl;
    std::cout << "Inference time: " << elapsed_ms << " ms" << std::endl;

    cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2);  // original
    cv::rectangle(frame, pred_bbox, cv::Scalar(0, 255, 0), 2); // predicted
    cv::imwrite("predicted.jpg", frame);
    std::cout << "Saved visualization: predicted.jpg" << std::endl;
    return 0;
}
