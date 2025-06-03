#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include<iostream>
#include "inference.h" // Include your inference header file
#include<sort.h>
using sort::Sort;

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

cv::Mat process_frame(Inference inf , const cv::Mat frame,Sort::Ptr mot) {
    // This function can be used to preprocess the frame if needed
    // For now, we just return the original frame
    cv::Mat frame_copy = frame.clone();
      std::vector<Detection> output = inf.runInference(frame_copy);

        int det = output.size();
        cv::Mat detections(det, 6, CV_32F);
        vector<cv::Scalar> colors;
        std::cout << "Number of detections:" << det << std::endl;
        for (int i = 0; i < det; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            colors.push_back(detection.color);
            float xc = detection.box.x + detection.box.width / 2.0f;
            float yc = detection.box.y + detection.box.height / 2.0f;
            float w = detection.box.width;
            float h = detection.box.height;
            float score = detection.confidence;
            float class_id = static_cast<float>(detection.class_id);

            detections.at<float>(i, 0) = xc;
            detections.at<float>(i, 1) = yc;
            detections.at<float>(i, 2) = w;
            detections.at<float>(i, 3) = h;
            detections.at<float>(i, 4) = score;
            detections.at<float>(i, 5) = class_id;

            // Detection box
            // cv::rectangle(frame_copy, box, colors[i], 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            // cv::rectangle(frame_copy, textBox, color, cv::FILLED);
            cv::putText(frame_copy, classString, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        
        }


        cv::Mat tracked = mot->update(detections);
        for (int i = 0; i < tracked.rows; i++)
        {
        float xc, yc, w, h, score, dx, dy;
        int trackerId;
        xc = tracked.at<float>(i, 0);
        yc = tracked.at<float>(i, 1);
        w = tracked.at<float>(i, 2);
        h = tracked.at<float>(i, 3);
        dx = tracked.at<float>(i, 6);
        dy = tracked.at<float>(i, 7);
        trackerId = int(tracked.at<float>(i, 8));
        // You can also extract velocity and tracker_id if needed
        // float vx = tracked.at<float>(i, 6);
        // float vy = tracked.at<float>(i, 7);
        // int tracker_id = static_cast<int>(tracked.at<float>(i, 8));
        std::string strl = "Id: " + std::to_string(trackerId);
        cv::rectangle(frame_copy, cv::Rect(xc - w/2, yc - h/2, w, h), colors[i], 2);
        cv::putText(frame_copy,strl ,cv::Point(xc - w/2, yc - h/2 -30),
                    cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2);
        cv::arrowedLine(frame_copy, cv::Point(xc, yc), cv::Point(xc + 5 * dx, yc + 5 * dy),
                       colors[i], 4);
        }
    return frame_copy;
}

void frame_processor() {
    auto last_time = std::chrono::high_resolution_clock::now();
    
    bool runOnGPU = true;
    Inference inf( "../models/yolov8s.onnx", cv::Size(640, 480), "../classes/classes.txt", runOnGPU);
    Sort::Ptr mot = std::make_shared<Sort>(1, 3, 0.3f);

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
        frame_copy = process_frame(inf, frame_copy,mot);
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
