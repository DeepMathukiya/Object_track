#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include<iostream>
#include "inference.h" // Include your inference header file


using namespace cv;
using namespace std;



std::mutex frame_mutex;
cv::Mat latest_frame;
std::atomic<bool> running (true);

void frame_reader(cv::VideoCapture& cap) {
    while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            running = false;
            break;
        }
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame = frame.clone();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // avoid busy waiting
    }
}

cv::Mat process_frame(Inference inf , const cv::Mat frame) {
    // This function can be used to preprocess the frame if needed
    // For now, we just return the original frame
    cv::Mat frame_copy = frame.clone();
      std::vector<Detection> output = inf.runInference(frame_copy);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;
        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame_copy, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            // cv::rectangle(frame_copy, textBox, color, cv::FILLED);
            cv::putText(frame_copy, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
    return frame_copy;
}

void frame_processor() {
    auto last_time = std::chrono::high_resolution_clock::now();
    
    bool runOnGPU = true;
    Inference inf( "../models/yolov8s.onnx", cv::Size(640, 480), "../classes/classes.txt", runOnGPU);

    while (running) {
       
        cv::Mat frame_copy;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (latest_frame.empty()) {
                continue; // No new frame to process
            }
            frame_copy = latest_frame.clone();
        }
        // Perform inference
        // Inference starts here...
        frame_copy = process_frame(inf, frame_copy);
        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame_copy, frame_copy, cv::Size(frame_copy.cols*scale, frame_copy.rows*scale));
        cv::imshow("Inference", frame_copy);


        auto current_time = std::chrono::high_resolution_clock::now();
        double fps = 1.0 / std::chrono::duration<double>(current_time - last_time).count();
        std::cout << "FPS: " << fps << std::endl;
        last_time = current_time;
        if (cv::waitKey(1) == 'q') {
            running = false;
            break;
        }

        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}


int main(int argc, char** argv) {
    // Load class names


    const char* path = argv[1];

    // Open video / image
    VideoCapture cap(path); // Use camera, or replace with "video.mp4" or "image.jpg"
    if (!cap.isOpened()) {
          std::cout << "Error: Could not open video file." << std::endl;
        return -1;
    } else {
        std::cout << "Video file opened successfully!" << std::endl;
    }
    int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));  // Get total number of frames
    double fps = cap.get(cv::CAP_PROP_FPS);  // Get frames per second (FPS)
    std::cout << "Total frames: " << frame_count << ", FPS: " << fps << std::endl;
 
    std::thread reader_thread(frame_reader, std::ref(cap));
    std::thread processor_thread(frame_processor);

    reader_thread.join();
    processor_thread.join();


    cap.release();
    destroyAllWindows();
    return 0;
}
