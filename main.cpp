#include <iostream>
#include <vector>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


void WhiteBalance(Mat& mat) {
    double discard_ratio = 0.05;
    int hists[3][256];
    memset(hists, 0, 3 * 256 * sizeof(int));

    for (int y = 0; y < mat.rows; ++y) {
        uchar* ptr = mat.ptr<uchar>(y);
        for (int x = 0; x < mat.cols; ++x) {
            for (int j = 0; j < 3; ++j) {
                hists[j][ptr[x * 3 + j]] += 1;
            }
        }
    }

    // cumulative hist
    int total = mat.cols * mat.rows;
    int vmin[3], vmax[3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 255; ++j) {
            hists[i][j + 1] += hists[i][j];
        }
        vmin[i] = 0;
        vmax[i] = 255;
        while (hists[i][vmin[i]] < discard_ratio * total)
            vmin[i] += 1;
        while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
            vmax[i] -= 1;
        if (vmax[i] < 255 - 1)
            vmax[i] += 1;
    }

    for (int y = 0; y < mat.rows; ++y) {
        uchar* ptr = mat.ptr<uchar>(y);
        for (int x = 0; x < mat.cols; ++x) {
            for (int j = 0; j < 3; ++j) {
                int val = ptr[x * 3 + j];
                if (val < vmin[j])
                    val = vmin[j];
                if (val > vmax[j])
                    val = vmax[j];
                ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));
            }
        }
    }
}

void UnsharpMasking(Mat& img, int sigma, int amount) {
    // improved UM
    Mat positive_g;
    GaussianBlur(img, positive_g, Size(), sigma, sigma);

    Mat negative;
    Mat negative_g;
    img.copyTo(negative);
    negative = 255 - negative;  // inverse grayscale
    GaussianBlur(negative, negative_g, Size(), sigma, sigma);

    addWeighted(img, 1 + amount, positive_g, -amount, 0, img);
    addWeighted(negative, amount, negative_g, -amount, 0, negative);
    add(img, negative, img);
}

void ContrastAdjustment(Mat& img, int position, float factor) {
    for (int row = 0; row < img.rows; ++row) {
        uchar* src = img.ptr<uchar>(row);

        for (int col = 0; col < img.cols; ++col) {
            *src++ = saturate_cast<uchar>(factor * (*src - position) + position);
        }
    }
}

void AdvancedMorphology(Mat& src, int elem, int size, int mode, int times) {
    // mode: 2, Opening. mode: 3, Closing.
    int type = 0;
    if (elem == 0) { type = MORPH_RECT; }
    else if (elem == 1) { type = MORPH_CROSS; }
    else if (elem == 2) { type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement(type,
        Size(2 * size + 1, 2 * size + 1),
        Point(size, size));

    morphologyEx(src, src, mode, element, Point(-1, -1), times);
}

void CreateMask(Mat& mask, int W, int H, int background) {
    if (background == 0) {
        mask = Mat::zeros(Size(W, H), CV_8U);
    }
    else {
        mask = Mat::ones(Size(W, H), CV_8U);
    }
    Point center(55, 285);
    int radius = 220;
    double startangle = -85;
    double endangle = 85;

    ellipse(mask, center, Size(radius, radius), 0,
            startangle, endangle, Scalar(!background), 41, 0);

    mask(Rect(549, 83, 70, 65)) = !background;
    mask(Rect(549, 165, 93, 105)) = !background;
    mask(Rect(549, 289, 87, 95)) = !background;
    mask(Rect(549, 399, 70, 79)) = !background;
}

int main() {
    string data_path = "E:\\Dataset\\defects-detection\\test\\";
    vector<int> total {7, 25};

    Mat img1;
    Mat img2;

    int X = 1200;  // ROI parameters
    int Y = 750;
    int W = 700;
    int H = 550;
    Mat roi1;
    Mat roi2;
    Mat roi1_sub1;
    Mat roi1_sub2;
    Mat roi2_sub1;
    Mat roi2_sub2;

    Mat dst1;
    Mat dst2;
    Mat dst;

    Mat goodsample_area = Mat::ones(Size(W, H), CV_8U);

    Mat mask1;
    CreateMask(mask1, W, H, 0);
    Mat mask2;
    CreateMask(mask2, W, H, 1);

    clock_t start, stop;
    int insidemaskareastandard = 0;
    int outsidemaskareastandard = 0;
    int insidemaskerror = 0;
    int outsidemaskerror = 0;
    int correct = 0;
    
    for (int times = 0; times < 2; ++times) {
        for (int num = 0; num < total[times]; ++num) {
            if (times == 0) {
                img1 = imread(data_path + "sample_good\\" + to_string(num + 1) + "_after.jpg");
                img2 = imread(data_path + "sample_good\\" + to_string(num + 1) + "_before.jpg");
            }
            else {
                img1 = imread(data_path + "sample_bad\\" + to_string(num + 1) + "b_after.jpg");
                img2 = imread(data_path + "sample_bad\\" + to_string(num + 1) + "b_before.jpg");
            }

            // start clock
            start = clock();

            // select ROI
            roi1 = img1(Rect(X, Y, W, H));
            roi2 = img2(Rect(X, Y, W, H));

            // remove noise
            medianBlur(roi1, roi1, 3);
            medianBlur(roi2, roi2, 3);

            // white balance
            WhiteBalance(roi1);
            WhiteBalance(roi2);

            // grayscale
            cvtColor(roi1, roi1, COLOR_BGR2GRAY);
            cvtColor(roi2, roi2, COLOR_BGR2GRAY);

            // divide into two parts
            roi1_sub1 = roi1(Rect(0, 0, 400, H));
            roi1_sub2 = roi1(Rect(400, 0, 300, H));
            roi2_sub1 = roi2(Rect(0, 0, 400, H));
            roi2_sub2 = roi2(Rect(400, 0, 300, H));

            // unsharp masking
            UnsharpMasking(roi1_sub1, 3, 1);
            UnsharpMasking(roi1_sub2, 3, 1);
            UnsharpMasking(roi2_sub1, 3, 1);
            UnsharpMasking(roi2_sub2, 3, 1);

            // contrast adjustment
            ContrastAdjustment(roi1_sub1, 60, 5);
            ContrastAdjustment(roi1_sub2, 145, 3);
            ContrastAdjustment(roi2_sub1, 60, 5);
            ContrastAdjustment(roi2_sub2, 145, 3);

            // subtract
            subtract(roi1_sub1, roi2_sub1, dst1);
            subtract(roi1_sub2, roi2_sub2, dst2);

            // threshold
            threshold(dst1, dst1, 8, 255, THRESH_BINARY);
            threshold(dst2, dst2, 8, 255, THRESH_BINARY);

            // concatenate two parts
            hconcat(dst1, dst2, dst);

            // opening morphology
            AdvancedMorphology(dst, 1, 1, 2, 2);
            
            // closing morphology
            AdvancedMorphology(dst, 1, 1, 3, 2);

            if (times == 0) {
                // end clock
                stop = clock();
                cout << "The " << num + 1 << " good sample" << '\n';
                cout << "Computational time: ";
                cout << double(stop - start) / CLOCKS_PER_SEC << " s" << '\n';

                // IOU calculation
                Mat goodsample = dst.mul(mask1);  // mask filter
                Mat groundtrue = imread(data_path + "sample_good_Groundtruth\\" + to_string(num + 1) + "_correct.jpg", IMREAD_GRAYSCALE);
                groundtrue = groundtrue(Rect(X, Y, W, H));
                groundtrue.convertTo(groundtrue, goodsample.type());
                groundtrue = groundtrue.mul(mask1);

                float gtarea = 0;
                float resultarea = 0;
                float commonarea = 0;
                for (int row = 0; row < groundtrue.rows; ++row)
                {
                    uchar* gt = groundtrue.ptr<uchar>(row);
                    uchar* res = goodsample.ptr<uchar>(row);

                    for (int col = 0; col < groundtrue.cols; ++col)
                    {
                        if (*gt >= 128) {
                            gtarea += 1;
                        }
                        if (*res >= 128) {
                            resultarea += 1;
                        }
                        if ((*gt >= 128) && (*res >= 128)) {
                            commonarea += 1;
                        }

                        gt++;
                        res++;
                    }
                }
                cout << "IOU: "<< commonarea * 100 / (gtarea + resultarea - commonarea) << " %" << '\n';
            
                // common area of good samples
                for (int row = 0; row < goodsample_area.rows; ++row)
                {
                    uchar* src1 = goodsample_area.ptr<uchar>(row);
                    uchar* src2 = dst.ptr<uchar>(row);

                    for (int col = 0; col < goodsample_area.cols; ++col)
                    {
                        if ((*src1 > 0) && (*src2 > 0)) {
                            *src1 = 255;
                        }
                        else {
                            *src1 = 0;
                        }

                        src1++;
                        src2++;
                    }
                }

                if (num == total[times] - 1) {
                    // check inside mask area standard
                    Mat inside_area = goodsample_area.mul(mask1);
                    for (int row = 0; row < inside_area.rows; ++row)
                    {
                        uchar* src = inside_area.ptr<uchar>(row);

                        for (int col = 0; col < inside_area.cols; ++col)
                        {
                            if (*src > 0) {
                                insidemaskareastandard += 1;
                            }

                            src++;
                        }
                    }

                    // check outside mask area standard
                    Mat outside_area = goodsample_area.mul(mask2);
                    for (int row = 0; row < outside_area.rows; ++row)
                    {
                        uchar* src = outside_area.ptr<uchar>(row);

                        for (int col = 0; col < outside_area.cols; ++col)
                        {
                            if (*src > 0) {
                                outsidemaskareastandard += 1;
                            }

                            src++;
                        }
                    }
                }
            }
            else {
                // check the area inside the mask
                Mat inside_area_ref = goodsample_area.mul(mask1);
                Mat inside_area = dst.mul(mask1);
                for (int row = 0; row < inside_area_ref.rows; ++row)
                {
                    uchar* src1 = inside_area_ref.ptr<uchar>(row);
                    uchar* src2 = inside_area.ptr<uchar>(row);

                    for (int col = 0; col < inside_area_ref.cols; ++col)
                    {
                        if ((*src1 == 255) && (*src2 == 0)) {
                            insidemaskerror += 1;
                        }
                        
                        src1++;
                        src2++;
                    }
                }

                // check the area outside the mask
                Mat outside_area_ref = goodsample_area.mul(mask2);
                Mat outside_area = dst.mul(mask2);
                for (int row = 0; row < outside_area_ref.rows; ++row)
                {
                    uchar* src1 = outside_area_ref.ptr<uchar>(row);
                    uchar* src2 = outside_area.ptr<uchar>(row);

                    for (int col = 0; col < outside_area_ref.cols; ++col)
                    {
                        if ((*src1 == 0) && (*src2 == 255)) {
                            outsidemaskerror += 1;
                        }

                        src1++;
                        src2++;
                    }
                }

                // end clock
                stop = clock();
                cout << "The " << num + 1 << " bad sample" << '\n';
                cout << "Computational time: ";
                cout << double(stop - start) / CLOCKS_PER_SEC << " s" << '\n';

                // display detection result
                if ((insidemaskerror + outsidemaskerror) >= 0.1*(insidemaskareastandard + outsidemaskareastandard)) {
                    correct += 1;
                    cout << "Result: " << "Bad" << '\n';
                }
                else {
                    cout << "Result: " << "Good" << '\n';
                }

                insidemaskerror = 0;
                outsidemaskerror = 0;
            }
            cout << '\n';
        }
    }
    cout << "Bad Sample Accuracy: " << correct*100/total[1] << " %" << '\n';
    
    cin.get();
    return 0;
}
