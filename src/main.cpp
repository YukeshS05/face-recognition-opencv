// ================================================================
//  Real-Time Face Detection & Recognition System
//  Phase 2: Known = Green Box + Name | Unknown = Red Box
//  Author  : Yukesh
// ================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <ctime>

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::face;

// ── Paths ────────────────────────────────────────────────────────
const std::string CASCADE_PATH =
    "D:/face-recognition-opencv/data/haarcascade_frontalface_default.xml";
const std::string KNOWN_FACES_DIR =
    "D:/face-recognition-opencv/data/known_faces/";

// ── Tuning ───────────────────────────────────────────────────────
// Lower = stricter match. Raise if known faces show as Unknown.
const double CONFIDENCE_THRESHOLD = 80.0;
const Size   FACE_SIZE(200, 200);

// ── Colors ───────────────────────────────────────────────────────
const Scalar GREEN (  0, 255,   0);
const Scalar RED   (  0,   0, 255);
const Scalar WHITE (255, 255, 255);
const Scalar BLACK (  0,   0,   0);
const Scalar YELLOW(  0, 255, 255);
const int    FONT = FONT_HERSHEY_SIMPLEX;

// ── Draw a filled label above the bounding box ───────────────────
void drawLabel(Mat& frame, const std::string& text,
               Point origin, Scalar color)
{
    int baseline = 0;
    Size sz = getTextSize(text, FONT, 0.6, 2, &baseline);
    Point p1 = origin + Point(0,  baseline + 2);
    Point p2 = origin + Point(sz.width + 6, -sz.height - 6);
    rectangle(frame, p1, p2, color, FILLED);
    putText(frame, text, origin + Point(3, -2),
            FONT, 0.6, BLACK, 2);
}

// ── Load images from known_faces/ and build training data ────────
bool loadTrainingData(CascadeClassifier& cascade,
                      std::vector<Mat>&  images,
                      std::vector<int>&  labels,
                      std::map<int, std::string>& labelMap)
{
    if (!fs::exists(KNOWN_FACES_DIR)) {
        std::cerr << "[WARN] known_faces folder not found: "
                  << KNOWN_FACES_DIR << "\n";
        return false;
    }

    int id = 0;
    for (const auto& entry : fs::directory_iterator(KNOWN_FACES_DIR))
    {
        if (!entry.is_directory()) continue;

        std::string name = entry.path().filename().string();
        labelMap[id]     = name;
        int count        = 0;

        std::cout << "[TRAIN] Loading: " << name << " ... ";

        for (const auto& imgEntry :
             fs::directory_iterator(entry.path()))
        {
            Mat img = imread(imgEntry.path().string());
            if (img.empty()) continue;

            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);

            std::vector<Rect> faces;
            cascade.detectMultiScale(gray, faces, 1.1, 5,
                                     0, Size(60, 60));

            for (const auto& face : faces) {
                Mat roi;
                resize(gray(face), roi, FACE_SIZE);
                images.push_back(roi);
                labels.push_back(id);
                ++count;
            }
        }

        std::cout << count << " face(s) loaded.\n";
        if (count == 0)
            std::cout << "  [WARN] No faces found in "
                      << name << "'s folder!\n";
        ++id;
    }

    return !images.empty();
}

// ================================================================
int main()
{
    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "   Face Detection & Recognition System\n";
    std::cout << "==============================================\n\n";

    // ── Load Haar Cascade ────────────────────────────────────
    CascadeClassifier faceCascade;
    if (!faceCascade.load(CASCADE_PATH)) {
        std::cerr << "[ERROR] Cannot load cascade file!\n"
                  << "        " << CASCADE_PATH << "\n";
        return -1;
    }
    std::cout << "[OK] Haar cascade loaded.\n";

    // ── Train LBPH Recognizer ────────────────────────────────
    auto recognizer = LBPHFaceRecognizer::create();
    std::map<int, std::string> labelMap;
    bool canRecognize = false;

    std::vector<Mat> trainImages;
    std::vector<int> trainLabels;

    std::cout << "[INFO] Scanning known_faces folder...\n";

    if (loadTrainingData(faceCascade,
                         trainImages, trainLabels, labelMap))
    {
        std::cout << "[TRAIN] Training on "
                  << trainImages.size() << " face image(s)...\n";
        recognizer->train(trainImages, trainLabels);
        std::cout << "[OK] Training complete!\n";
        canRecognize = true;

        std::cout << "\n[People registered]\n";
        for (auto& p : labelMap)
            std::cout << "  ID " << p.first
                      << "  →  " << p.second << "\n";
    }
    else {
        std::cout << "[WARN] No training data found.\n"
                  << "       Running in Detection-Only mode.\n"
                  << "       Add photos to data/known_faces/<Name>/\n";
    }

    // ── Open Webcam ──────────────────────────────────────────
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open webcam!\n";
        return -1;
    }
    cap.set(CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "\n[OK] Webcam open. Press Q or ESC to quit.\n\n";

    Mat  frame, gray;
    std::vector<Rect> faces;
    double fps       = 0.0;
    int64  tickStart = getTickCount();

    // ── Main Loop ────────────────────────────────────────────
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // Detect faces
        faceCascade.detectMultiScale(
            gray, faces, 1.1, 5, 0, Size(80, 80));

        // ── Process Each Face ─────────────────────────────
        for (const auto& faceRect : faces)
        {
            std::string name   = "Unknown";
            Scalar      color  = RED;
            double      conf   = 0.0;

            if (canRecognize)
            {
                Mat roi;
                resize(gray(faceRect), roi, FACE_SIZE);

                int predLabel = -1;
                recognizer->predict(roi, predLabel, conf);

                if (predLabel >= 0 &&
                    conf < CONFIDENCE_THRESHOLD)
                {
                    name  = labelMap[predLabel];
                    color = GREEN;
                }
            }
            else {
                // Detection-only mode
                name  = "Face Detected";
                color = GREEN;
            }

            // Bounding box
            rectangle(frame, faceRect, color, 2);

            // Label with confidence
            std::string displayText = name;
            if (canRecognize)
                displayText += "  " +
                    std::to_string((int)conf) + "%";

            drawLabel(frame, displayText,
                      Point(faceRect.x, faceRect.y - 5),
                      color);
        }

        // ── HUD ──────────────────────────────────────────
        // FPS
        int64 now = getTickCount();
        fps = getTickFrequency() / (double)(now - tickStart);
        tickStart = now;

        putText(frame,
            "FPS: " + std::to_string((int)fps),
            Point(10, 35), FONT, 0.8, YELLOW, 2);

        // Face count
        putText(frame,
            "Faces: " + std::to_string(faces.size()),
            Point(10, 70), FONT, 0.8, WHITE, 2);

        // Mode
        std::string modeStr = canRecognize
            ? "Mode: Recognition (" +
              std::to_string(labelMap.size()) + " people)"
            : "Mode: Detection Only";
        putText(frame, modeStr,
            Point(frame.cols/2 - 180, 35),
            FONT, 0.7, WHITE, 2);

        // Title
        putText(frame, "Press Q to Quit",
            Point(frame.cols - 200, 35),
            FONT, 0.6, WHITE, 1);

        imshow("Face Recognition System", frame);

        char key = (char)waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;

        // Press S to save screenshot
        if (key == 's' || key == 'S') {
            std::string filename = "screenshot_" +
            std::to_string(time(nullptr)) + ".jpg";
            imwrite(filename, frame);
            std::cout << "[OK] Screenshot saved: " << filename << "\n";
        }
    }

    cap.release();
    destroyAllWindows();
    std::cout << "[OK] Exited cleanly.\n";
    return 0;
}